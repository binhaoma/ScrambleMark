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
import re
import numpy as np
import os
import torch
import torch.nn as nn
import os
import glob
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import glob
from dataset import from_path, from_gtzan
from model import DiffWave
from params import AttrDict
import pickle

def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


class DiffWaveLearner:
  def __init__(self, model_dir, model, dataset, optimizer, params, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.optimizer = optimizer
    self.params = params
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.is_master = True

    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta)
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    self.loss_fn = nn.L1Loss()
    self.summary_writer = None

  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'])
    else:
      self.model.load_state_dict(state_dict['model'])
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']

  # def save_to_checkpoint(self, filename='weights'):
  #     # 1. 删除旧 checkpoint
  #     pattern = os.path.join(self.model_dir, f'{filename}-*.pt')
  #     old_checkpoints = glob.glob(pattern)
      
  #     for ckpt in old_checkpoints:
  #         os.remove(ckpt)

  #     # 2. 保存新的
  #     save_basename = f'{filename}-{self.step}.pt'
  #     save_name = f'{self.model_dir}/{save_basename}'
  #     link_name = f'{self.model_dir}/{filename}.pt'

  #     torch.save(self.state_dict(), save_name)

  #     # 3. 更新 latest link
  #     if os.name == 'nt':
  #         torch.save(self.state_dict(), link_name)
  #     else:
  #         if os.path.exists(link_name) or os.path.islink(link_name):
  #             os.remove(link_name)
  #         os.symlink(save_basename, link_name)

  # def save_to_checkpoint(self, filename='weights'):
  #   save_basename = f'{filename}-{self.step}.pt'
  #   save_name = f'{self.model_dir}/{save_basename}'
  #   link_name = f'{self.model_dir}/{filename}.pt'
  #   torch.save(self.state_dict(), save_name)
  #   if os.name == 'nt':
  #     torch.save(self.state_dict(), link_name)
  #   else:
  #     if os.path.exists(link_name) or os.path.islink(link_name):
  #           os.remove(link_name)   # 关键：无论文件还是link都删
  #     # if os.path.islink(link_name):
  #     #   os.unlink(link_name)
  #     os.symlink(save_basename, link_name)

  # def save_to_checkpoint(self, filename='weights'):
  #   save_basename = f'{filename}-{self.step}.pt'
  #   save_name = os.path.join(self.model_dir, save_basename)
  #   tmp_name = save_name + '.tmp'
  #   link_name = os.path.join(self.model_dir, f'{filename}.pt')

  #   # 先写入临时文件
  #   torch.save(self.state_dict(), tmp_name)

  #   # 写完后原子替换成正式 checkpoint
  #   os.replace(tmp_name, save_name)

  #   if os.name == 'nt':
  #     torch.save(self.state_dict(), link_name)
  #   else:
  #     if os.path.exists(link_name) or os.path.islink(link_name):
  #       os.remove(link_name)
  #     os.symlink(save_basename, link_name)



  def save_to_checkpoint(self, filename='weights', keep=2):
      save_basename = f'{filename}-{self.step}.pt'
      save_name = os.path.join(self.model_dir, save_basename)

      os.makedirs(self.model_dir, exist_ok=True)

      print(f"[checkpoint] before sync step={self.step}", flush=True)

      # 调试阶段保留：用来提前暴露前面训练里的 CUDA 异步错误
      if torch.cuda.is_available():
          torch.cuda.synchronize()

      print(f"[checkpoint] start save: {save_name}", flush=True)

      state = {
          k: v.detach().cpu() if torch.is_tensor(v) else v
          for k, v in self.state_dict().items()
      }

      torch.save(state, save_name)

      print(f"[checkpoint] finish save: {save_name}", flush=True)
      print(f"[checkpoint] done step={self.step}", flush=True)



  # def restore_from_checkpoint(self, filename='weights'):
  #   try:
  #     checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
  #     self.load_state_dict(checkpoint)
  #     return True
  #   except FileNotFoundError:
  #     return False
  def restore_from_checkpoint(self, filename='weights'):
    import glob
    import re
    def try_load_checkpoint(path):
  # 文件不存在
      if not os.path.exists(path):
        return False

      # 文件为空，直接跳过
      if os.path.getsize(path) == 0:
        print(f'[Skip] Empty checkpoint: {path}')
        return False

      try:
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint)
        print(f'[Restore] Loaded checkpoint: {path}, step={self.step}')
        return True

      except (EOFError, RuntimeError, OSError, pickle.UnpicklingError, KeyError) as e:
        print(f'[Skip] Bad checkpoint: {path}')
        print(f'       Reason: {repr(e)}')
        return False


    # 1. 先扫描 weights-*.pt
    pattern = os.path.join(self.model_dir, f'{filename}-*.pt')
    ckpt_paths = glob.glob(pattern)

    ckpts = []
    step_pattern = re.compile(rf'{re.escape(filename)}-(\d+)\.pt$')

    for path in ckpt_paths:
      base = os.path.basename(path)
      match = step_pattern.match(base)
      if match:
        step = int(match.group(1))
        ckpts.append((step, path))

    # 2. step 从大到小，优先加载最新的带步数 checkpoint
    ckpts.sort(key=lambda x: x[0], reverse=True)

    for step, path in ckpts:
      if try_load_checkpoint(path):
        return True


    # 3. 如果没有任何有效的 weights-*.pt，再尝试 weights.pt
    latest_path = os.path.join(self.model_dir, f'{filename}.pt')
    if try_load_checkpoint(latest_path):
      return True


    print('[Restore] No valid checkpoint found. Train from scratch.')
    return False

    # def try_load_checkpoint(path):
    #   # 文件不存在
    #   if not os.path.exists(path):
    #     return False

    #   # 文件为空，直接跳过
    #   if os.path.getsize(path) == 0:
    #     print(f'[Skip] Empty checkpoint: {path}')
    #     return False

    #   try:
    #     checkpoint = torch.load(path, map_location='cpu')
    #     self.load_state_dict(checkpoint)
    #     print(f'[Restore] Loaded checkpoint: {path}, step={self.step}')
    #     return True

    #   except (EOFError, RuntimeError, OSError, pickle.UnpicklingError, KeyError) as e:
    #     print(f'[Skip] Bad checkpoint: {path}')
    #     print(f'       Reason: {repr(e)}')
    #     return False

    # # 1. 先尝试 weights.pt
    # latest_path = os.path.join(self.model_dir, f'{filename}.pt')
    # if try_load_checkpoint(latest_path):
    #   return True

    # # 2. 如果 weights.pt 坏了，就扫描 weights-*.pt
    # pattern = os.path.join(self.model_dir, f'{filename}-*.pt')
    # ckpt_paths = glob.glob(pattern)

    # ckpts = []
    # step_pattern = re.compile(rf'{re.escape(filename)}-(\d+)\.pt$')

    # for path in ckpt_paths:
    #   base = os.path.basename(path)
    #   match = step_pattern.match(base)
    #   if match:
    #     step = int(match.group(1))
    #     ckpts.append((step, path))

    # # 3. step 从大到小，找最新的有效 checkpoint
    # ckpts.sort(key=lambda x: x[0], reverse=True)

    # for step, path in ckpts:
    #   if try_load_checkpoint(path):
    #     # 修复 weights.pt，让它重新指向有效 checkpoint
    #     link_name = os.path.join(self.model_dir, f'{filename}.pt')
    #     save_basename = os.path.basename(path)

    #     if os.name != 'nt':
    #       if os.path.exists(link_name) or os.path.islink(link_name):
    #         os.remove(link_name)
    #       os.symlink(save_basename, link_name)
    #       print(f'[Fix] Updated latest symlink: {link_name} -> {save_basename}')
    #     else:
    #       torch.save(self.state_dict(), link_name)
    #       print(f'[Fix] Updated latest checkpoint file: {link_name}')

    #     return True

    print('[Restore] No valid checkpoint found. Train from scratch.')
    return False
    
    
# for features in tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:
  def train(self, max_steps=None):
    device = next(self.model.parameters()).device
    while True:
      for features in tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:
        if max_steps is not None and self.step >= max_steps:
          return
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        loss = self.train_step(features)
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          if self.step % 1000 == 0:
            self._write_summary(self.step, features, loss)
          if self.step % 5000*len(self.dataset) == 0:
            self.save_to_checkpoint()
        self.step += 1

  def train_step(self, features):
    for param in self.model.parameters():
      param.grad = None

    audio = features['audio']
    spectrogram = features['spectrogram']

    N, T = audio.shape
    device = audio.device
    self.noise_level = self.noise_level.to(device)

    with self.autocast:
      t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
      noise_scale = self.noise_level[t].unsqueeze(1)
      noise_scale_sqrt = noise_scale**0.5
      noise = torch.randn_like(audio)
      noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise

      predicted = self.model(noisy_audio, t, spectrogram)
      loss = self.loss_fn(noise, predicted.squeeze(1))

    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss

  def _write_summary(self, step, features, loss):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)
    if not self.params.unconditional:
      writer.add_image('feature/spectrogram', torch.flip(features['spectrogram'][:1], [1]), step)
    writer.add_scalar('train/loss', loss, step)
    writer.add_scalar('train/grad_norm', self.grad_norm, step)
    writer.flush()
    self.summary_writer = writer


def _train_impl(replica_id, model, dataset, args, params):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

  learner = DiffWaveLearner(args.model_dir, model, dataset, opt, params, fp16=args.fp16)
  learner.is_master = (replica_id == 0)
  learner.restore_from_checkpoint()
  learner.train(max_steps=args.max_steps)


def train(args, params):
  if args.data_dirs[0] == 'gtzan':
    dataset = from_gtzan(params)
  else:
    dataset = from_path(args.data_dirs, params)
  model = DiffWave(params).cuda()
  _train_impl(0, model, dataset, args, params)


def train_distributed(replica_id, replica_count, port, args, params):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(port)
  torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)
  if args.data_dirs[0] == 'gtzan':
    dataset = from_gtzan(params, is_distributed=True)
  else:
    dataset = from_path(args.data_dirs, params, is_distributed=True)
  device = torch.device('cuda', replica_id)
  torch.cuda.set_device(device)
  model = DiffWave(params).to(device)
  model = DistributedDataParallel(model, device_ids=[replica_id])
  _train_impl(replica_id, model, dataset, args, params)
