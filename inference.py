# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from argparse import ArgumentParser
import gc
import os
import threading
import time

import numpy as np
import torch
import torchaudio

from diffwave.params import AttrDict, params as base_params
from diffwave.model import DiffWave


models = {}


def _process_rss_bytes():
  """Return the current process RSS in bytes on Linux."""
  try:
    with open('/proc/self/statm', 'r') as statm:
      resident_pages = int(statm.readline().split()[1])
    return resident_pages * os.sysconf('SC_PAGE_SIZE')
  except (OSError, ValueError, IndexError):
    return None


def _synchronize_device(device):
  device = torch.device(device)
  if device.type == 'cuda' and torch.cuda.is_available():
    torch.cuda.synchronize(device)


def _cuda_memory_allocated(device):
  device = torch.device(device)
  if device.type == 'cuda' and torch.cuda.is_available():
    return torch.cuda.memory_allocated(device)
  return None


def _cuda_memory_reserved(device):
  device = torch.device(device)
  if device.type == 'cuda' and torch.cuda.is_available():
    return torch.cuda.memory_reserved(device)
  return None


def _mib(num_bytes):
  return num_bytes / (1024 ** 2)


class _PeakRssMonitor:
  """Sample process RSS while an inference call is running."""
  def __init__(self, poll_interval=0.005):
    self.poll_interval = poll_interval
    self.before = None
    self.peak = None
    self.after = None
    self._stop_event = threading.Event()
    self._thread = None

  def _sample(self):
    rss = _process_rss_bytes()
    if rss is not None and (self.peak is None or rss > self.peak):
      self.peak = rss
    return rss

  def _run(self):
    while not self._stop_event.wait(self.poll_interval):
      self._sample()

  def start(self):
    initial_rss = _process_rss_bytes()
    if initial_rss is None:
      return
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()
    # Take the baseline after creating the sampler thread so its small
    # bookkeeping allocation is not attributed to inference.
    self.before = _process_rss_bytes() or initial_rss
    self.peak = self.before

  def stop(self):
    if self._thread is not None:
      self._stop_event.set()
      self._thread.join()
    self.after = self._sample()


def _model_cache_key(model_dir, device):
  return os.path.abspath(model_dir), str(torch.device(device))


def _predict(spectrogram=None, model_dir=None, params=None, device=torch.device('cuda'), fast_sampling=False):
  device = torch.device(device)
  model_key = _model_cache_key(model_dir, device)

  # Lazy load model.
  if model_key not in models:
    _synchronize_device(device)
    load_started_at = time.perf_counter()

    if os.path.exists(f'{model_dir}/weights.pt'):
      checkpoint = torch.load(f'{model_dir}/weights.pt', map_location=device)
    else:
      checkpoint = torch.load(model_dir, map_location=device)
    model = DiffWave(AttrDict(base_params)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    models[model_key] = model

    # Release the temporary checkpoint before finishing the load timing.
    del checkpoint
    gc.collect()
    _synchronize_device(device)

    load_time = time.perf_counter() - load_started_at
    print(f'[model] load_time={load_time:.3f}s', flush=True)

  model = models[model_key]
  _synchronize_device(device)
  inference_started_at = time.perf_counter()
  model.params.override(params)
  with torch.no_grad():
    # Change in notation from the DiffWave paper for fast sampling.
    # DiffWave paper -> Implementation below
    # --------------------------------------
    # alpha -> talpha
    # beta -> training_noise_schedule
    # gamma -> alpha
    # eta -> beta
    training_noise_schedule = np.array(model.params.noise_schedule)
    inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)

    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)


    if not model.params.unconditional:
      if len(spectrogram.shape) == 2:# Expand rank 2 tensors by adding a batch dimension.
        spectrogram = spectrogram.unsqueeze(0)
      spectrogram = spectrogram.to(device)
      audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
    else:
      audio = torch.randn(1, params.audio_len, device=device)
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

    for n in range(len(alpha) - 1, -1, -1):
      c1 = 1 / alpha[n]**0.5
      c2 = beta[n] / (1 - alpha_cum[n])**0.5
      audio = c1 * (audio - c2 * model(audio, torch.tensor([T[n]], device=audio.device), spectrogram).squeeze(1))
      if n > 0:
        noise = torch.randn_like(audio)
        sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
        audio += sigma * noise
      audio = torch.clamp(audio, -1.0, 1.0)

  _synchronize_device(device)
  inference_time = time.perf_counter() - inference_started_at
  print(f'[inference] time={inference_time:.3f}s', flush=True)
  return audio, model.params.sample_rate


def predict(spectrogram=None, model_dir=None, params=None, device=torch.device('cuda'), fast_sampling=False):
  """Run inference and report peak memory relative to the call baseline."""
  device = torch.device(device)
  cold_start = _model_cache_key(model_dir, device) not in models
  scope = 'model_load+inference' if cold_start else 'inference_only'
  _synchronize_device(device)

  ram_monitor = _PeakRssMonitor()
  try:
    ram_monitor.start()

    vram_before = _cuda_memory_allocated(device)
    vram_reserved_before = _cuda_memory_reserved(device)
    cuda_device = torch.device(device)
    if vram_before is not None:
      torch.cuda.reset_peak_memory_stats(cuda_device)

    result = _predict(
        spectrogram=spectrogram,
        model_dir=model_dir,
        params=params,
        device=device,
        fast_sampling=fast_sampling)
    _synchronize_device(device)
  finally:
    ram_monitor.stop()

  memory_stats = [f'scope={scope}', f'cold_start={str(cold_start).lower()}']
  if ram_monitor.before is not None and ram_monitor.peak is not None:
    memory_stats.append(f'RAM_RSS_peak_delta={_mib(ram_monitor.peak - ram_monitor.before):+.2f} MiB')
    memory_stats.append(f'RAM_RSS_before={_mib(ram_monitor.before):.2f} MiB')
    memory_stats.append(f'RAM_RSS_peak={_mib(ram_monitor.peak):.2f} MiB')
    if ram_monitor.after is not None:
      memory_stats.append(f'RAM_RSS_after={_mib(ram_monitor.after):.2f} MiB')
  else:
    memory_stats.append('RAM_RSS_peak=unavailable')

  if vram_before is not None:
    vram_allocated_peak = torch.cuda.max_memory_allocated(cuda_device)
    vram_reserved_peak = torch.cuda.max_memory_reserved(cuda_device)
    vram_after = _cuda_memory_allocated(cuda_device)
    vram_reserved_after = _cuda_memory_reserved(cuda_device)
    memory_stats.append(f'VRAM_allocated_peak_delta={_mib(vram_allocated_peak - vram_before):+.2f} MiB')
    memory_stats.append(f'VRAM_allocated_before={_mib(vram_before):.2f} MiB')
    memory_stats.append(f'VRAM_allocated_peak={_mib(vram_allocated_peak):.2f} MiB')
    memory_stats.append(f'VRAM_allocated_after={_mib(vram_after):.2f} MiB')
    memory_stats.append(f'VRAM_reserved_peak_delta={_mib(vram_reserved_peak - vram_reserved_before):+.2f} MiB')
    memory_stats.append(f'VRAM_reserved_peak={_mib(vram_reserved_peak):.2f} MiB')
    memory_stats.append(f'VRAM_reserved_after={_mib(vram_reserved_after):.2f} MiB')

  # print(f"[end_to_end_memory] {', '.join(memory_stats)}", flush=True)
  return result


def main(args):
  device = torch.device(args.device)
  print(f'[device] device={device.type}', flush=True)
  spec_names = os.listdir(args.input_dir)
  os.makedirs(args.output_dir, exist_ok=True)
  for spec_name in spec_names:

    spectrogram_path = os.path.join(args.input_dir, spec_name)
    spectrogram = torch.from_numpy(np.load(spectrogram_path))

    
    for i in range(args.num_samples):
      audio, sr = predict(
          spectrogram,
          model_dir=args.model_dir,
          fast_sampling=args.fast,
          params=base_params,
          device=device)
      output_name = spec_name.removesuffix(".spec.npy")
      torchaudio.save(f'{args.output_dir}/{output_name}', audio.cpu(), sample_rate=sr)
      del audio


if __name__ == '__main__':
  parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('--model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')
  parser.add_argument('--input_dir', '-s',
      help='path to a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('--output_dir', '-dir', default='./output',
      help='output file name')
  # parser.add_argument('--output', '-o', default='output.wav',
  #     help='output file name')
  parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda',
      help='inference device (use cpu to avoid allocating GPU memory)')
  parser.add_argument('--num_samples', '-n', default=1, type=int,
      help='number of audio samples to generate')
  parser.add_argument('--fast', '-f', action='store_true',
      help='fast sampling procedure')
  main(parser.parse_args())
