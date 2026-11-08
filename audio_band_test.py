"""Reset-corrected, band-level pitch check for the neural audio model (no GPU needed).

Inputs: the stored held-out loss sequences of the audio Transformer
  audio_losses_60_rerun.npy        (real held-out audio)
  audio_shuf_losses_60_rerun.npy   (same model, shuffled held-out audio)
Both come from audio_reset_corrected_check.ipynb (fixed seeds).

Usage:  python audio_band_test.py real.npy shuffled.npy
"""
import sys
import numpy as np

W = 255                    # context-reset period (256-position context -> 255 predictions per window)
BAND = (58.0, 91.0)        # pitch band in samples at 8 kHz = 10th-90th percentile of the autocorrelation
                           # tracker's frame-level F0 over all 60 recordings (87.9-137.9 Hz), fixed in advance
RANGE = (40.0, 200.0)      # reference range of periods, in samples


def resid(x):
    """Remove the position-dependent mean cost (kills the period-W component and all its harmonics)."""
    L = x[:(len(x) // W) * W].reshape(-1, W)
    return (L - L.mean(axis=0, keepdims=True)).ravel()


def spectrum(x):
    s = x - x.mean()
    sp = np.abs(np.fft.rfft(s))
    f = np.fft.rfftfreq(len(s))
    return 1.0 / f[1:], sp[1:]          # period (samples), magnitude


def ratio(per, a, lo, hi):
    inside = (per >= lo) & (per <= hi)
    ref = (per >= RANGE[0]) & (per <= RANGE[1]) & ~inside
    return a[inside].mean() / a[ref].mean()


def report(name, x):
    per, a = spectrum(resid(x))
    obs = ratio(per, a, *BAND)
    width = BAND[1] - BAND[0]
    sliding = np.array([ratio(per, a, lo, lo + width) for lo in np.arange(RANGE[0], RANGE[1] - width, 1.0)])
    edges = [40, 50, 58, 75, 91, 120, 200.01]
    ref = a[(per >= RANGE[0]) & (per <= RANGE[1])].mean()
    prof = [a[(per >= l) & (per < h)].mean() / ref for l, h in zip(edges[:-1], edges[1:])]
    print(f"{name:9s} band ratio (58-91 samples vs rest of 40-200): {obs:.3f}")
    print(f"{'':9s} sliding same-width bands: min {sliding.min():.3f}, max {sliding.max():.3f}")
    print(f"{'':9s} coarse profile {edges[:-1]} -> {[round(v, 2) for v in prof]}")


if __name__ == "__main__":
    real = np.load(sys.argv[1]).astype(float)
    shuf = np.load(sys.argv[2]).astype(float)
    report("real", real)
    report("shuffled", shuf)
