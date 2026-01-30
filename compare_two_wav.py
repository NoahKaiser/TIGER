import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags, resample_poly

def to_mono(x: np.ndarray) -> np.ndarray:
    # x: (n,) or (n, ch)
    if x.ndim == 1:
        return x
    return x.mean(axis=1)

def rms(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.mean(x * x)) + eps)

def load_wav(path: str):
    x, sr = sf.read(path, always_2d=False)
    x = to_mono(np.asarray(x, dtype=np.float64))
    return x, sr

def resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    # rational resample using polyphase filtering
    # up/down as integers:
    from math import gcd
    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return resample_poly(x, up, down)

def normalized_xcorr_max(a: np.ndarray, b: np.ndarray):
    """
    Returns:
      max_corr: max normalized cross-correlation (can be negative if phase-inverted)
      lag_samples: lag (in samples) at which max occurs (positive => b lags behind a)
    """
    # remove DC offset
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)

    # full cross-correlation
    c = correlate(a0, b0, mode="full", method="auto")
    lags = correlation_lags(len(a0), len(b0), mode="full")

    # normalize to [-1, 1] (roughly): divide by energy
    denom = (np.linalg.norm(a0) * np.linalg.norm(b0)) + 1e-12
    c_norm = c / denom

    idx = int(np.argmax(np.abs(c_norm)))  # best match regardless of sign
    best = float(c_norm[idx])
    lag = int(lags[idx])
    return best, lag

def pearson_corr_zero_lag(a: np.ndarray, b: np.ndarray):
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    denom = (np.linalg.norm(a0) * np.linalg.norm(b0)) + 1e-12
    return float(np.dot(a0, b0) / denom)

def main(wav1: str, wav2: str, target_sr: int | None = None, max_seconds: float | None = None):
    x1, sr1 = load_wav(wav1)
    x2, sr2 = load_wav(wav2)

    # choose SR
    if target_sr is None:
        target_sr = sr1
    x1 = resample_to(x1, sr1, target_sr)
    x2 = resample_to(x2, sr2, target_sr)

    # optionally limit duration (faster for long files)
    if max_seconds is not None:
        nmax = int(max_seconds * target_sr)
        x1 = x1[:nmax]
        x2 = x2[:nmax]

    # match lengths (trim to min)
    n = min(len(x1), len(x2))
    x1 = x1[:n]
    x2 = x2[:n]

    # compute
    best_corr, lag_samp = normalized_xcorr_max(x1, x2)
    lag_sec = lag_samp / target_sr
    zero_lag_corr = pearson_corr_zero_lag(x1, x2)

    print(f"Samplerate used: {target_sr} Hz")
    print(f"Compared samples: {n} ({n/target_sr:.2f} s)")
    print(f"Zero-lag correlation: {zero_lag_corr:.6f}")
    print(f"Best normalized cross-correlation (|max|): {best_corr:.6f}")
    print(f"Lag at best match: {lag_samp} samples ({lag_sec:.6f} s)")
    print("Interpretation:")
    print(" - Values near 1.0: very similar (same polarity)")
    print(" - Values near -1.0: very similar but phase-inverted")
    print(" - Values near 0: little similarity")

if __name__ == "__main__":
    # example:
    # main("a.wav", "b.wav", target_sr=16000, max_seconds=30)
    import sys
    if len(sys.argv) < 3:
        print("Usage: python compare_wavs.py <wav1> <wav2> [target_sr] [max_seconds]")
        sys.exit(1)
    wav1, wav2 = sys.argv[1], sys.argv[2]
    target_sr = int(sys.argv[3]) if len(sys.argv) >= 4 else None
    max_seconds = float(sys.argv[4]) if len(sys.argv) >= 5 else None
    main(wav1, wav2, target_sr=target_sr, max_seconds=max_seconds)
