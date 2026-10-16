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
RIG_RADIUS = 15e-3          # m, beam spot from the disc axis (Results, rig C)
RIG_ANGLE_DEG = 45.0        # collimator to motor axis
# Expected shift per RPM: v_LOS = (2 pi N / 60) d sin(alpha), f_D = 2 v / lambda.
HZ_PER_RPM = 2 / WAVELENGTH * 2 * np.pi / 60 * RIG_RADIUS * np.sin(np.radians(RIG_ANGLE_DEG))
C_POINT = CAT[0]
C_EXPECTED = CAT[1]


def peak_of(f: np.ndarray, excess: np.ndarray, f_min: float = 20e3,
            smooth: int = 5, locate: np.ndarray | None = None) -> dict:
    """Centroid and FWHM edges of the strongest peak in the excess spectrum.

    locate, if given, is the spectrum the peak is found in (the capture over
    its reference, say); the centroid and edges are still taken on the excess.
    Without it the largest absolute excess wins, which near DC can be drift
    or vibration rather than the Doppler return.
    """
    kernel = np.ones(smooth) / smooth
    sm = np.convolve(excess, kernel, mode="same")
    seek = sm if locate is None else np.convolve(locate, kernel, mode="same")
    i = int(np.argmax(np.where(f >= f_min, seek, -np.inf)))

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
    """series is the display name; selection happens before this is called.

    Untitled: the report caption names the figure. The expected shift from
    the rig geometry is drawn against the measurement, and the right axis
    reads the peak frequency as line-of-sight speed, v = f_D lambda / 2.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import figstyle
    figstyle.use()

    rpm = np.array([r["rpm"] for r in rows], float)
    fc = np.array([r["centroid"] for r in rows]) / 1e6

    fig, ax = plt.subplots(figsize=(figstyle.TEXT_WIDTH, 3.6))
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    xs = np.array([0, rpm.max() * 1.05])

    ax.plot(xs, k * xs / 1e6, color=C_MUTED, lw=1.2, ls=(0, (5, 4)), zorder=2,
            label=f"fit to measured: {k:.0f} ± {se:.0f} Hz/RPM")
    line, = ax.plot(xs, HZ_PER_RPM * xs / 1e6, color=C_EXPECTED, lw=1.6,
                    ls=(0, (1, 1.5)), zorder=2)
    marks, = ax.plot(rpm, HZ_PER_RPM * rpm / 1e6, "o", ms=7, mfc=C_SURFACE,
                     mec=C_EXPECTED, mew=1.8, zorder=3)
    ax.plot(rpm, fc, "o", ms=6, color=C_POINT, mec=C_SURFACE, mew=1.2, zorder=4,
            label="measured centroid")
    ax.set_xlabel("target speed (RPM)", color=C_INK)
    ax.set_ylabel("Doppler peak frequency (MHz)", color=C_INK)
    ax.set_xlim(0, xs[1])
    ax.set_ylim(0, max(fc.max(), k * xs[1] / 1e6) * 1.03)

    handles, labels = ax.get_legend_handles_labels()
    handles.insert(1, (line, marks))
    labels.insert(1, f"expected: {HZ_PER_RPM:.0f} Hz/RPM")
    # Above the lines, where the top left of the plot is empty; the labels are
    # kept short so the legend stays clear of the 2000 RPM points.
    ax.legend(handles, labels, frameon=False, loc="upper left", labelcolor=C_INK,
              fontsize=figstyle.FS_NOTE)

    sec = ax.secondary_yaxis("right", functions=(lambda f: f * 1e6 * WAVELENGTH / 2,
                                                 lambda v: v * 2 / WAVELENGTH / 1e6))
    sec.set_ylabel(r"line-of-sight velocity $v_{\mathrm{LOS}}$ (m/s)", color=C_INK)
    sec.spines["right"].set_color(C_GRID)
    sec.tick_params(colors=C_MUTED)

    ax.grid(True, color=C_GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED)

    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
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
        # Found by ratio to the reference: at 1000 RPM on 9 Oct a near-DC
        # excess matches the Doppler peak in absolute power, but it is 3x the
        # reference where the Doppler peak is 60x.
        rows.append({"rpm": rpm, "capture": e["time"],
                     **peak_of(f, e["fft"] - ref, locate=e["fft"] / ref)})

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
