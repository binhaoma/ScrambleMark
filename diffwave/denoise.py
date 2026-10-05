#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch audio denoising with 7 classic (non-deep) methods.

Methods:
  1) Spectral Subtraction
  2) Wiener Filtering
  3) Spectral Gating (soft gate)
  4) MMSE-STSA (Ephraim–Malah spectral amplitude estimator)
  5) Wavelet Thresholding
  6) Kalman Filtering (AR(1) speech model)
  7) Non-Local Means (1-D, patch-based)

Python: 3.9
"""

import os
import sys
import argparse
from pathlib import Path
import math

import numpy as np
import soundfile as sf
import librosa
import scipy.signal as spsig
import scipy.fft as spfft
import pywt
import torch

# ---------------------------
# I/O helpers
# ---------------------------

def load_audio(path, target_sr=None, mono=True):
    """Read audio with soundfile, optionally resample, convert to mono."""
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim == 2 and mono:
        y = np.mean(y, axis=1)
    if (target_sr is not None) and (sr != target_sr):
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    y = np.asarray(y, dtype=np.float32)
    return y, sr


def save_audio(path, y, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)


def stft(y, n_fft=1024, hop=256, win="hann", center=True):
    w = spsig.get_window(win, n_fft, fftbins=True).astype(np.float32)
    return librosa.stft(y, n_fft=n_fft, hop_length=hop, window=w, center=center)


def istft(S, hop=256, win="hann", length=None, center=True):
    n_fft = (S.shape[0] - 1) * 2
    w = spsig.get_window(win, n_fft, fftbins=True).astype(np.float32)
    return librosa.istft(S, hop_length=hop, window=w, length=length, center=center)


def magphase(S):
    return np.abs(S), np.exp(1j * np.angle(S))


# ---------------------------
# Noise profile estimation
# ---------------------------

def estimate_noise_psd(M, frames_for_noise):
    """
    Estimate noise magnitude & power from first N 'noise' frames (simple, stationary).
    M: magnitude spectrogram (freq x time)
    frames_for_noise: int, number of initial frames assumed noise-only
    """
    N = min(frames_for_noise, M.shape[1])
    if N <= 0:
        N = min(6, M.shape[1])  # fallback
    noise_mag = np.maximum(np.mean(M[:, :N], axis=1, keepdims=True), 1e-8)
    noise_psd = noise_mag ** 2
    return noise_mag, noise_psd


# ---------------------------
# 1) Spectral Subtraction
# ---------------------------

def denoise_spectral_subtraction(y, sr, n_fft=1024, hop=256, noise_frames=6,
                                 oversub=1.0, floor=0.02):
    """
    Basic spectral subtraction with oversubtraction and spectral floor (to reduce musical noise).
    """
    S = stft(y, n_fft=n_fft, hop=hop)
    M, P = magphase(S)
    noise_mag, _ = estimate_noise_psd(M, noise_frames)
    # subtract
    M_hat = M - oversub * noise_mag
    M_hat = np.where(M_hat > floor * noise_mag, M_hat, floor * noise_mag)
    S_hat = M_hat * P
    y_hat = istft(S_hat, hop=hop, length=len(y))
    return y_hat.astype(np.float32)

import numpy as np

def adaptive_comb_filter(signal, fs, f0, alpha):
    # Compute the comb filter coefficients
    a = np.zeros(int(fs/f0))
    a[0] = 1
    a[1] = -alpha
    # Compute the filtered signal
    filtered_signal = np.convolve(signal, a, mode='same')
    return filtered_signal



# ---------------------------
# 2) Wiener Filtering
# ---------------------------

def denoise_wiener(y, sr, n_fft=1024, hop=256, noise_frames=6, alpha_dd=0.98):

    if y.shape[0] == 1:
        y = y.squeeze(0)
    y = np.array(y)

    S = stft(y, n_fft=n_fft, hop=hop)
    M, P = magphase(S)

    _, Npsd = estimate_noise_psd(M, noise_frames)
    Ypsd = np.maximum(M**2, 1e-12)

    gamma = Ypsd / np.maximum(Npsd, 1e-12)
    xi_prev = np.maximum(gamma[:, [0]] - 1.0, 0.0)

    M_hat = np.zeros_like(M)

    for t in range(M.shape[1]):

        if t == 0:
            xi = xi_prev
        else:
            xi = alpha_dd * (G_prev**2) * gamma[:, [t-1]] + \
                 (1 - alpha_dd) * np.maximum(gamma[:, [t]] - 1.0, 0.0)

        # Wiener filter gain
        G = xi / (xi + 1.0)

        # Enhance magnitude
        M_hat[:, t:t+1] = G * M[:, t:t+1]

        # save gain for next frame
        G_prev = G

    S_hat = M_hat * P
    y_hat = istft(S_hat, hop=hop, length=len(y))

    return torch.tensor(y_hat).unsqueeze(0), sr


# ---------------------------
# 3) Spectral Gating (soft)
# ---------------------------

def denoise_spectral_gate(y, sr, n_fft=1024, hop=256, noise_frames=6,
                          thresh_k=1.0, min_gain=0.05, smooth=True):
    """
    Soft gate by comparing magnitude to (noise_mean + k*noise_std).
    """
    S = stft(y, n_fft=n_fft, hop=hop)
    M, P = magphase(S)
    N_mag, _ = estimate_noise_psd(M, noise_frames)
    # crude std estimate
    N_std = np.std(M[:, :min(noise_frames, M.shape[1])], axis=1, keepdims=True) + 1e-8
    thresh = N_mag + thresh_k * N_std

    ratio = M / np.maximum(thresh, 1e-8)
    if smooth:
        gain = np.clip(ratio, min_gain, 1.0)
    else:
        gain = (ratio >= 1.0).astype(np.float32)
        gain[gain == 0] = min_gain

    M_hat = gain * M
    S_hat = M_hat * P
    y_hat = istft(S_hat, hop=hop, length=len(y))
    return y_hat.astype(np.float32)


# ---------------------------
# 4) MMSE-STSA (Ephraim–Malah)
# ---------------------------

def _mmse_stsa_gain(xi, gamma):
    """
    Ephraim–Malah MMSE-STSA gain function.
    xi: a priori SNR
    gamma: a posteriori SNR
    """
    # nu = gamma * xi / (1 + xi)
    nu = (gamma * xi) / (1.0 + xi + 1e-12)
    # Approximate gain using the classic approximation:
    # G = xi / (1 + xi) * exp(0.5 * E1(nu))   (E1 is exponential integral of order 1)
    # We use a stable approximation: G_MMSE ≈ sqrt(pi)/2 * sqrt(nu) * exp(-nu/2) * ((1 + nu) * I0(nu/2) + nu * I1(nu/2)) / gamma
    # To avoid special functions dependencies, use a simpler practical approximation:
    # G ≈ xi / (1 + xi) * np.exp(0.5 * np.exp(-nu))
    G = (xi / (1.0 + xi + 1e-12)) * np.exp(0.5 * np.exp(-np.clip(nu, 0, 40)))
    return np.clip(G, 0.0, 1.0).astype(np.float32)

def mmse_stsa(y, n_fft=1024, hop=256, noise_frames=6, alpha_dd=0.98):
    y = np.array(y)
    if y.shape[0] == 1:
        y = y.squeeze(0)
    S = stft(y, n_fft=n_fft, hop=hop)
    M, P = magphase(S)
    _, Npsd = estimate_noise_psd(M, noise_frames)
    Ypsd = np.maximum(M**2, 1e-12)

    gamma = Ypsd / np.maximum(Npsd, 1e-12)
    xi_prev = np.maximum(gamma - 1.0, 0.0)

    M_hat = np.zeros_like(M)
    for t in range(M.shape[1]):
        if t == 0:
            xi = np.maximum(gamma[:, [t]] - 1.0, 0.0)
        else:
            xi = alpha_dd * (M_hat[:, [t-1]]**2) / np.maximum(Npsd, 1e-12) + (1 - alpha_dd) * np.maximum(gamma[:, [t]] - 1.0, 0.0)

        G = _mmse_stsa_gain(xi, gamma[:, [t]])
        M_hat[:, t:t+1] = G * M[:, t:t+1]

    S_hat = M_hat * P
    y_hat = istft(S_hat, hop=hop, length=len(y))
    return y_hat.astype(np.float32)


# ---------------------------
# 5) Wavelet Thresholding
# ---------------------------

def denoise_wavelet(y, wavelet="db8", level=None, mode="soft"):
    """
    Wavelet denoising with universal threshold (VisuShrink), sigma via MAD.
    """
    # DWT
    coeffs = pywt.wavedec(y, wavelet=wavelet, level=level)
    # Noise sigma from finest detail coeffs
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail)) / 0.6745 + 1e-9
    thr = sigma * math.sqrt(2 * math.log(len(y)))
    # Threshold all detail coeffs
    new_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        new_coeffs.append(pywt.threshold(c, thr, mode=mode))
    y_hat = pywt.waverec(new_coeffs, wavelet=wavelet)
    y_hat = y_hat[:len(y)]
    return y_hat.astype(np.float32)


# ---------------------------
# 6) Kalman Filtering (AR(1) speech model)
# ---------------------------

def denoise_kalman(y, sr, ar_a=0.95, q=1e-4, r=None):
    """
    Simple scalar Kalman filter assuming:
      x_k = a * x_{k-1} + w_k,  w ~ N(0, q)
      y_k = x_k + v_k,          v ~ N(0, r)
    r is noise variance estimated from first 0.25 s if not given.
    """
    y = y.astype(np.float32)
    if r is None:
        n_init = max(1, int(0.25 * sr))
        r = float(np.var(y[:n_init])) * 0.3 + 1e-8

    # Init
    x_hat = 0.0
    P = 1.0

    out = np.zeros_like(y)
    for k in range(len(y)):
        # Predict
        x_hat = ar_a * x_hat
        P = ar_a * P * ar_a + q

        # Update
        K = P / (P + r)
        x_hat = x_hat + K * (y[k] - x_hat)
        P = (1 - K) * P
        out[k] = x_hat

    return out.astype(np.float32)


# ---------------------------
# 7) Non-Local Means (1-D, patch-based)
# ---------------------------

def denoise_nlm_1d(y, patch_size=21, search_radius=200, h=None, step=1):
    """
    1-D patch-based Non-Local Means (simplified). This can be slow on long files.
    - patch_size: odd
    - search_radius: samples to each side for similar patches
    - h: filtering parameter (if None, derived from noise estimate)
    """
    y = y.astype(np.float32)
    N = len(y)
    half = patch_size // 2
    if patch_size % 2 == 0:
        patch_size += 1
        half = patch_size // 2

    # Simple noise std estimate from high-frequency energy
    hf = y - spsig.medfilt(y, kernel_size=5)
    sigma = float(np.std(hf)) + 1e-8
    if h is None:
        h = 0.6 * sigma

    out = np.zeros(N, dtype=np.float32)
    wsum = np.zeros(N, dtype=np.float32)

    # Pre-pad
    y_pad = np.pad(y, (half, half), mode='reflect')

    for i in range(0, N, step):
        i0 = i
        # reference patch
        ref = y_pad[i0:i0 + 2 * half + 1]

        jmin = max(i0 - search_radius, 0)
        jmax = min(i0 + search_radius, N - 1)

        # Accumulate
        for j in range(jmin, jmax + 1, step):
            nbr = y_pad[j:j + 2 * half + 1]
            if nbr.shape[0] != ref.shape[0]:
                continue
            d2 = np.sum((ref - nbr) ** 2) / patch_size
            w = math.exp(-d2 / (h * h + 1e-12))
            out[i] += w * y[j]
            wsum[i] += w

    # Fill remaining zero weights (edges or if step>1)
    nz = wsum > 0
    out[nz] = out[nz] / wsum[nz]
    out[~nz] = y[~nz]

    # Optional light smoothing to reduce isolated spikes
    out = spsig.medfilt(out, kernel_size=3).astype(np.float32)
    return out


# ---------------------------
# Batch runner
# ---------------------------

def process_file(path, out_root, args):
    y, sr = load_audio(path, target_sr=args.sr, mono=True)

    # Common STFT params
    n_fft = args.n_fft
    hop = args.hop
    noise_frames = args.noise_frames

    # 1) Spectral Subtraction
    y_ss = denoise_spectral_subtraction(
        y, sr, n_fft=n_fft, hop=hop, noise_frames=noise_frames,
        oversub=args.ss_oversub, floor=args.ss_floor
    )
    save_audio(out_root / "spectral_subtraction" / path.name, y_ss, sr)

    # 2) Wiener
    y_wi = denoise_wiener(
        y, sr, n_fft=n_fft, hop=hop, noise_frames=noise_frames, alpha_dd=args.dd_alpha
    )
    save_audio(out_root / "wiener" / path.name, y_wi, sr)

    # 3) Spectral Gate
    y_gate = denoise_spectral_gate(
        y, sr, n_fft=n_fft, hop=hop, noise_frames=noise_frames,
        thresh_k=args.gate_k, min_gain=args.gate_min_gain, smooth=True
    )
    save_audio(out_root / "spectral_gate" / path.name, y_gate, sr)

    # 4) MMSE-STSA
    y_mmse = denoise_mmse_stsa(
        y, sr, n_fft=n_fft, hop=hop, noise_frames=noise_frames, alpha_dd=args.dd_alpha
    )
    save_audio(out_root / "mmse_stsa" / path.name, y_mmse, sr)

    # 5) Wavelet
    y_wav = denoise_wavelet(
        y, wavelet=args.wavelet, level=args.wavelet_level, mode=args.wavelet_mode
    )
    save_audio(out_root / "wavelet" / path.name, y_wav, sr)

    # 6) Kalman
    y_kal = denoise_kalman(
        y, sr, ar_a=args.kalman_a, q=args.kalman_q, r=None
    )
    save_audio(out_root / "kalman" / path.name, y_kal, sr)

    # 7) NLM (can be slow; keep conservative defaults)
    y_nlm = denoise_nlm_1d(
        y, patch_size=args.nlm_patch, search_radius=args.nlm_search, h=args.nlm_h, step=args.nlm_step
    )
    save_audio(out_root / "nlm" / path.name, y_nlm, sr)


def find_audio_files(in_dir):
    exts = (".wav", ".flac", ".ogg", ".mp3", ".m4a")
    return [p for p in Path(in_dir).rglob("*") if p.suffix.lower() in exts]


import numpy as np

def compressor(x, threshold=-20.0, ratio=2.0, attack=0.01, release=0.1, fs=48000, epsilon=1e-9):
    """
    一个简单的数字动态范围压缩器实现。

    参数:
    x (numpy.ndarray): 输入的单通道音频信号。
    threshold (float): 压缩开始的阈值，单位 dB (通常为负值)。
    ratio (float): 压缩比 (例如 2.0 代表 2:1)。
    attack (float): 增益降低的启动时间，单位秒。
    release (float): 增益恢复的释放时间，单位秒。
    fs (int): 采样率，单位 Hz。
    epsilon (float): 用于防止除以零的小阈值。

    返回:
    numpy.ndarray: 压缩后的输出音频信号 y。
    """
    
    # 1. 参数初始化和转换
    
    # 将 dB 阈值转换为幅度值 T = 10^(threshold/20)
    T = 10 ** (threshold / 20.0) 
    
    # 初始化输出数组和动态增益
    y = np.zeros_like(x)
    gain = 1.0  # 初始增益为 1.0 (0 dB)
    
    # 计算增益平滑系数 (Alpha)
    # alpha_a 和 alpha_r 是一阶低通滤波器的系数
    alpha_a = np.exp(-1.0 / (fs * attack))
    alpha_r = np.exp(-1.0 / (fs * release))

    # 2. 信号处理循环
    
    for i in range(len(x)):
        
        # a. 瞬时电平检测 (Level Detection)
        level = abs(x[i])
        
        # b. 瞬时增益计算 (Ideal Gain Calculation)
        if level > T:
            # 信号超过阈值，需要压缩
            # desired = T + (level - T) / ratio
            # g = desired / level
            
            # 使用对数域（dB）计算增益更常见，但这里保持幅度域计算：
            # 超过阈值部分的幅度压缩：level_excess = level - T
            # 目标输出幅度：T + (level_excess / ratio)
            
            # 检查 level 是否极小，防止除以零或数值溢出
            if level < epsilon:
                 g = 1.0
            else:
                 desired_level = T + (level - T) / ratio
                 g = desired_level / level 
            
        else:
            # 信号未超过阈值，增益为 1.0 (不压缩)
            g = 1.0
            
        # c. 增益平滑 (Gain Smoothing)
        # 根据瞬时目标增益 g 和当前增益 gain 的关系，选择 Attack 或 Release
        if g < gain:
            # 目标增益 g 更小 (需要压缩): 使用较快的 Attack 时间
            gain = alpha_a * gain + (1.0 - alpha_a) * g
        else:
            # 目标增益 g 更大 (需要恢复): 使用较慢的 Release 时间
            gain = alpha_r * gain + (1.0 - alpha_r) * g
            
        # d. 增益应用 (Apply Gain)
        y[i] = x[i] * gain
        
    return y

from scipy import signal
def biquad_peak_optimized(x, fs, f0 = 4000, Q= 1, gain_db= 2):
    """
    使用 SciPy.signal.lfilter 实现的高效双二阶峰值 (Bell) 滤波器。

    参数:
    x (np.ndarray): 输入音频信号数组。
    fs (float): 采样率 (Hz)。
    f0 (float): 中心频率 (Hz)。
    Q (float): 品质因数，控制带宽。
    gain_db (float): 在 f0 处的增益 (dB)。正值为提升，负值为衰减。

    返回:
    np.ndarray: 滤波后的输出信号数组。
    """
    
    # 1. 系数准备
    
    # 将增益 (dB) 转换为线性振幅比 A
    A = 10**(gain_db / 40.0)
    # 计算归一化角频率
    w0 = 2 * np.pi * f0 / fs
    # 计算带宽控制参数 alpha
    alpha = np.sin(w0) / (2.0 * Q)
    
    # 2. 计算双二阶滤波器系数
    
    # 分子系数 (b)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    
    # 分母系数 (a)
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A

    # 3. 归一化和滤波
    
    # 将系数归一化，确保 a[0] = 1，这是 lfilter 的要求
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    
    # 使用 lfilter 函数进行滤波
    y = signal.lfilter(b, a, x)
    
    return y


def main():
    ap = argparse.ArgumentParser(description="Batch denoise audio with 7 traditional methods.")
    ap.add_argument("--in_dir", type=Path, default="./test/", help="Directory containing audio files")
    ap.add_argument("--out_dir", type=Path, default="./test_denoise/", help="Directory to write outputs")
    ap.add_argument("--sr", type=int, default=None, help="Resample to this sample rate (e.g., 16000). Leave None to keep original")
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--noise_frames", type=int, default=6, help="Initial frames assumed noise-only for spectral methods")

    # Spectral Subtraction params
    ap.add_argument("--ss_oversub", type=float, default=1.0, help="Oversubtraction factor")
    ap.add_argument("--ss_floor", type=float, default=0.02, help="Spectral floor as fraction of noise magnitude")

    # Decision-directed alpha (Wiener/MMSE)
    ap.add_argument("--dd_alpha", type=float, default=0.98, help="Decision-directed smoothing (0..1)")

    # Gate params
    ap.add_argument("--gate_k", type=float, default=1.0, help="k in noise_mean + k*noise_std")
    ap.add_argument("--gate_min_gain", type=float, default=0.05, help="Minimum gain to avoid holes")

    # Wavelet params
    ap.add_argument("--wavelet", default="db8")
    ap.add_argument("--wavelet_level", type=int, default=None, help="None lets pywt choose max level")
    ap.add_argument("--wavelet_mode", choices=["soft", "hard"], default="soft")

    # Kalman params
    ap.add_argument("--kalman_a", type=float, default=0.95, help="AR(1) coefficient")
    ap.add_argument("--kalman_q", type=float, default=1e-4, help="Process noise variance")

    # NLM params (warning: can be slow)
    ap.add_argument("--nlm_patch", type=int, default=21, help="Odd patch size")
    ap.add_argument("--nlm_search", type=int, default=200, help="Search radius in samples")
    ap.add_argument("--nlm_h", type=float, default=None, help="Filtering parameter (None -> auto from noise)")
    ap.add_argument("--nlm_step", type=int, default=1, help="Stride for speed/quality tradeoff")

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_audio_files(in_dir)
    if not files:
        print(f"No audio files found in: {in_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} files. Processing...")
    for p in files:
        rel = p.relative_to(in_dir)
        out_root = out_dir / rel.parent
        process_file(p, out_root, args)
        print(f"Processed: {rel}")

    print("Done.")


if __name__ == "__main__":
    main()
