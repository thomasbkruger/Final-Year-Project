"""simple_fft.py - averaged FFT of one Acquisition.py capture.

One FFT per record, magnitudes averaged over the records. No Welch, no
segmenting: the whole record is one transform.

    python simple_fft.py data/20260928_110022_B1_48mA.npz
    python simple_fft.py data/....npz --plot fft.png
"""
import sys

import numpy as np

XINC, YINC, YORIG, YREF = 4, 7, 8, 9    # preamble columns (Keysight X-Series)


def averaged_fft(path):
    """Returns f (Hz) and the record-averaged single-sided amplitude spectrum (V)."""
    with np.load(path) as d:
        raw = d["raw"].astype(float)            # (n_records, n_samples), ADC codes
        pre = d["preamble"].astype(float)

    v = (raw - pre[:, YREF, None]) * pre[:, YINC, None] + pre[:, YORIG, None]
    fs = 1.0 / pre[0, XINC]

    n = v.shape[1]
    w = np.hanning(n)
    spec = np.fft.rfft((v - v.mean(axis=1, keepdims=True)) * w, axis=1)
    amp = np.abs(spec) * 2 / w.sum()            # single-sided amplitude, volts
    return np.fft.rfftfreq(n, 1 / fs), amp.mean(axis=0)


if __name__ == "__main__":
    path = sys.argv[1]
    f, amp = averaged_fft(path)

    print(f"{path}: {len(f)} bins, df = {f[1]:.4g} Hz, up to {f[-1]/1e6:.4g} MHz")
    peak = 1 + np.argmax(amp[1:])               # skip DC
    print(f"largest component: {amp[peak]*1e6:.4g} uV at {f[peak]/1e6:.4g} MHz")

    if "--plot" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = sys.argv[sys.argv.index("--plot") + 1]
        plt.figure(figsize=(9, 5))
        plt.loglog(f[1:], amp[1:] * 1e6, lw=0.8)
        plt.xlabel("frequency (Hz)")
        plt.ylabel("amplitude (uV)")
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"plot -> {out}")
