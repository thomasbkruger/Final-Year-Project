"""peak_vs_rpm.py - Doppler peak frequency against target RPM, C1 series.

Works on the FFT amplitude spectra from psd_rpm.py (5 kHz bins), one capture
per RPM (--capture, as there). For each RPM the stationary reference (C1_ref,
mean of its captures) is subtracted in power, leaving the Doppler excess, and:

  * peak location = power-weighted centroid of the excess over the bins where
    it stays above 20 % of its maximum. The bin of maximum alone is not used:
    the 50 RPM peak has two lobes and the argmax lands on whichever is taller.
  * the full width at half maximum is reported in the table (not plotted),
    taken between the outermost half-maximum bins inside that 20 % region, so
    a two-lobed peak gets the width of both lobes rather than just the taller.

The excess is smoothed over 5 bins (25 kHz) only to find those edges; the
centroid itself uses the unsmoothed excess.

A line through the origin, f = k * RPM, is fitted (0 RPM must give 0 Hz). The
slope's line-of-sight speed (f_D = 2 v / lambda) is printed, not plotted.

    python peak_vs_rpm.py
    python peak_vs_rpm.py --capture 2 --out peak_vs_rpm_cap2.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from psd_rpm import C_GRID, C_INK, C_MUTED, C_SURFACE, CAT, collect

WAVELENGTH = 1550e-9        # m, DFB laser (System Design)
C_POINT = CAT[0]


def peak_of(f: np.ndarray, excess: np.ndarray, f_min: float = 20e3,
            smooth: int = 5) -> dict:
    """Centroid and FWHM edges of the strongest peak in the excess spectrum."""
    sm = np.convolve(excess, np.ones(smooth) / smooth, mode="same")
    i = int(np.argmax(np.where(f >= f_min, sm, -np.inf)))

    lo = hi = i
    while lo > 0 and sm[lo - 1] > 0.2 * sm[i]:
        lo -= 1
    while hi < len(sm) - 1 and sm[hi + 1] > 0.2 * sm[i]:
        hi += 1
    r = slice(lo, hi + 1)
    centroid = float((f[r] * excess[r]).sum() / excess[r].sum())
    above = lo + np.flatnonzero(sm[r] > 0.5 * sm[i])
    h_lo, h_hi = above[0], above[-1]
    return {"centroid": centroid, "argmax": float(f[i]),
            "fwhm_lo": float(f[h_lo]), "fwhm_hi": float(f[h_hi])}


def fit_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares slope of y = k x through the origin, its standard error, R^2."""
    k = float((x * y).sum() / (x * x).sum())
    ss_res = ((y - k * x) ** 2).sum()
    se = float(np.sqrt(ss_res / (len(x) - 1) / (x * x).sum()))
    ss_tot = ((y - y.mean()) ** 2).sum()
    return k, se, float(1 - ss_res / ss_tot)


def plot(rows: list[dict], k: float, se: float, out: Path, series: str) -> None:
    """series is the display name; selection happens before this is called."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rpm = np.array([r["rpm"] for r in rows], float)
    fc = np.array([r["centroid"] for r in rows]) / 1e6

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    xs = np.array([0, rpm.max() * 1.05])

    ax.plot(xs, k * xs / 1e6, color=C_MUTED, lw=1.2, ls=(0, (5, 4)), zorder=2,
            label=f"fit through origin: {k:.0f} ± {se:.0f} Hz/RPM")
    ax.plot(rpm, fc, "o", ms=7, color=C_POINT, mec=C_SURFACE, mew=1.5, zorder=3,
            label="peak centroid")
    ax.set_xlabel("target speed (RPM)", color=C_INK)
    ax.set_ylabel("Doppler peak frequency (MHz)", color=C_INK)
    ax.set_xlim(0, xs[1])
    ax.set_ylim(0, None)
    ax.set_title(f"{series} Doppler peak vs target speed", loc="left",
                 fontsize=11, color=C_INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=C_INK)

    ax.grid(True, color=C_GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"plot -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--series", default="C1",
                    help="tag prefix that selects the captures")
    ap.add_argument("--label", default="C",
                    help="name for the series in the figure; kept separate "
                         "from --series so selection stays exact")
    ap.add_argument("--ref", default="C1_ref")
    ap.add_argument("--date", help="YYYYmmdd; default is the latest date with RPM captures")
    ap.add_argument("--capture", type=int, default=1,
                    help="which capture of each RPM to use, in time order")
    ap.add_argument("--out", type=Path, default=Path("peak_vs_rpm.png"))
    args = ap.parse_args()

    # nperseg only affects the PSD, which is not used here.
    d = collect(args.data_dir, args.series, args.ref, args.date, nperseg=12500)
    if not d["rpm"] or not d["ref"]:
        print(f"need {args.series} RPM captures and {args.ref} captures on one date")
        return 1
    f = d["f"]["fft"]
    ref = np.mean([e["fft"] for e in d["ref"]], axis=0)

    rows = []
    for rpm, caps in d["rpm"].items():
        if not 1 <= args.capture <= len(caps):
            print(f"{rpm} RPM has {len(caps)} capture(s); cannot pick #{args.capture}")
            return 1
        e = caps[args.capture - 1]
        rows.append({"rpm": rpm, "capture": e["time"], **peak_of(f, e["fft"] - ref)})

    rpm = np.array([r["rpm"] for r in rows], float)
    fc = np.array([r["centroid"] for r in rows])
    k, se, r2 = fit_origin(rpm, fc)

    print(f"{args.series} on {d['date']}, capture {args.capture}, FFT df = {f[1]/1e3:g} kHz")
    print(f"  {'RPM':>5} {'centroid':>10} {'argmax':>9} {'FWHM':>17} {'Hz/RPM':>8} {'from fit':>9}")
    for r in rows:
        print(f"  {r['rpm']:5d} {r['centroid']/1e3:8.1f} k {r['argmax']/1e3:7.1f} k "
              f"{r['fwhm_lo']/1e3:7.0f}-{r['fwhm_hi']/1e3:<7.0f}k "
              f"{r['centroid']/r['rpm']:8.1f} "
              f"{100*(r['centroid']-k*r['rpm'])/(k*r['rpm']):+8.2f}%")
    v = k * WAVELENGTH / 2
    print(f"  fit: {k:.1f} +/- {se:.1f} Hz/RPM, R^2 = {r2:.6f}")
    print(f"  -> {v*1e3:.4f} mm/s line-of-sight per RPM, "
          f"effective r cos(theta) = {v*60/(2*np.pi)*1e3:.2f} mm")

    table = args.out.with_suffix(".csv")
    with table.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"table -> {table}")

    plot(rows, k, se, args.out, args.label or args.series)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
