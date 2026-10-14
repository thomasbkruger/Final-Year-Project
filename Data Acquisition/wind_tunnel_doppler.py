"""wind_tunnel_doppler.py - D series: PSD overlay and Doppler peak vs tunnel speed.

Uses the 12 Oct wind-tunnel captures, one per speed setpoint (the earliest when
a speed was captured more than once), leaving out the mist runs. The 9 Oct
3 m/s capture is added as well, plotted at 3.5 m/s: it was logged as 3 m/s, but
its peak sits where 3.5 m/s does on the line through the other speeds (3.5-3.8
m/s was the range given for it; 3.5 fits best).

These dense-fog captures are compared with the mist runs (tags ending _mist),
which are picked the same way, one per speed.

Four figures, in the style of psd_rpm.py and peak_vs_rpm.py:

  * psd_windtunnel_overlay.png - every dense-fog speed's Welch PSD on one
    linear axis;
  * psd_windtunnel_overlay_floor.png - the same PSDs, each divided by its own
    running-median floor (below) and shown in dB, so the floors' different
    levels drop out and every Doppler peak stands up from 0 dB;
  * peak_vs_windtunnel_speed.png - Doppler peak against setpoint for dense fog
    and mist, with a least-squares line through the origin (dense fog only)
    and the predicted shift for the rig geometry,
    f_D = (2 / lambda) cos(35 deg) U = 1.057 MHz per m/s. The right axis
    converts the peak frequency to tunnel speed with that same factor;
  * psd_5mps_mist_averaging.png - the 5 m/s mist PSD relative to its floor,
    from one capture, two averaged and all three averaged, side by side: the
    mist return is weak enough that averaging visibly lifts it out of the noise.

There is no no-flow reference on 12 Oct, and the 9 Oct one sits above the flow
captures (the floor moved), so the peak is not found by subtracting a
reference. Each PSD is instead divided by its own running median over
BASELINE_BINS bins - wide enough that the Doppler bump barely moves it - and
the peak is located on that ratio, its centroid taken on the excess over the
median with peak_vs_rpm.peak_of.

A setpoint tagged as approximate ("15ishmps") is plotted with a hollow marker
and left out of the fit. The 12 Oct captures once tagged 15ishmps were retagged
21mps, which is where their ~24.6 MHz peak puts them.

    python wind_tunnel_doppler.py
    python wind_tunnel_doppler.py --nperseg 12500
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from acquire_windtunnel import psd_of
from noise_vs_rpm import measure
from peak_vs_rpm import WAVELENGTH, fit_origin, peak_of
from psd_rpm import C_GRID, C_INK, C_MUTED, C_SURFACE, CAT, style_axes

BEAM_ANGLE_DEG = 35.0       # beam to horizontal flow, from the rig geometry
HZ_PER_MPS = 2 / WAVELENGTH * np.cos(np.radians(BEAM_ANGLE_DEG))   # 1.057 MHz
BASELINE_BINS = 61          # running-median width; 3 MHz at 50 kHz bins
F_MIN = 1e6                 # peak search starts here, above the 1/f end
F_MAX = 40e6                # and stops here: a fixed bump near 60 MHz, close to
                            # Nyquist, outranks the weak mist returns
AVERAGING_TAG = "5mps_mist"  # the captures compared in psd_5mps_mist_averaging

# 5mps, 11-3mps (11.3 m/s), 15ishmps (an approximate setpoint, ~15 m/s)
SPEED_TAG = re.compile(r"^(\d+)(?:-(\d+))?(ish)?mps$")

# Captures from other dates, with the speed to plot them at.
EXTRA = {"20261009_135015_3mps": 3.5}


def speed_of(tag: str) -> tuple[float, bool] | None:
    """Setpoint in m/s from a tag, and whether it was only approximate."""
    m = SPEED_TAG.match(tag)
    if not m:
        return None
    whole, frac, ish = m.groups()
    return float(f"{whole}.{frac}" if frac else whole), bool(ish)


def captures(data_dir: Path, date: str, tag: str) -> list[Path]:
    """Every capture from `date` tagged exactly `tag`, in time order."""
    return [p for p in sorted(data_dir.glob(f"{date}_*_{tag}.npz"))
            if p.stem[16:] == tag]


def select(data_dir: Path, date: str, mist: bool = False) -> list[dict]:
    """One capture per speed from `date` (the earliest), either the dense-fog
    runs plus EXTRA or, with mist=True, the mist runs alone."""
    chosen: dict[float, dict] = {}
    for p in sorted(data_dir.glob(f"{date}_*.npz")):   # time order
        if p.stem.endswith(("_psd", "_volts")):
            continue
        tag = p.stem[16:]
        if tag.endswith("_mist") != mist:
            continue
        s = speed_of(tag.removesuffix("_mist"))
        if s is None or s[0] in chosen:
            continue
        chosen[s[0]] = {"path": p, "speed": s[0], "approx": s[1], "extra": False}
    if not mist:
        for stem, speed in EXTRA.items():
            chosen[speed] = {"path": data_dir / f"{stem}.npz", "speed": speed,
                             "approx": False, "extra": True}
    return [chosen[k] for k in sorted(chosen)]


def locate(f: np.ndarray, psd: np.ndarray) -> dict:
    """Doppler peak of one PSD, against its own running-median floor."""
    base = median_filter(psd, size=BASELINE_BINS, mode="nearest")
    # Zero, not the ratio's ~1 baseline, above F_MAX, so nothing there can win.
    p = peak_of(f, psd - base, f_min=F_MIN,
                locate=np.where(f <= F_MAX, psd / base, 0))
    j = int(np.argmin(np.abs(f - p["centroid"])))
    return {**p, "floor": base,
            "ratio_dB": float(10 * np.log10(psd[j] / base[j]))}


def speed_label(r: dict) -> str:
    return f"{'~' if r['approx'] else ''}{r['speed']:g} m/s"


def finish(ax, fig, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"plot -> {out}")


def plot_overlay(rows: list[dict], f: np.ndarray, out: Path, title: str,
                 y_of=lambda r: r["psd"], ylabel: str | None = None) -> None:
    """Every speed on one linear axis; y_of picks what is plotted per row."""
    import matplotlib.pyplot as plt

    # The date's own captures take the categorical slots in speed order; one
    # from another date is drawn dashed in muted ink, which also keeps the
    # count within the eight slots rather than inventing a ninth hue.
    own = [r for r in rows if not r["extra"]]
    if len(own) > len(CAT):
        raise ValueError(f"{len(own)} speeds but only {len(CAT)} categorical "
                         "colours; split the figure rather than inventing hues")
    colour = dict(zip((id(r) for r in own), CAT))
    m = f > 0
    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    for r in rows:
        style = ({"color": C_MUTED, "ls": (0, (5, 4))} if r["extra"]
                 else {"color": colour[id(r)]})
        ax.plot(f[m], y_of(r)[m], lw=1.4, zorder=3,
                label=f"{speed_label(r)} PSD", **style)
    fmax = max(r["centroid"] for r in rows) * 1.25
    ax.set_xlim(0, fmax)
    # Ranged to the spectrum above F_MIN: the near-DC spike would otherwise set
    # the scale and flatten the Doppler peaks.
    shown = (f >= F_MIN) & (f <= fmax)
    lo = min(y_of(r)[shown].min() for r in rows)
    hi = max(y_of(r)[shown].max() for r in rows)
    ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    style_axes(ax, "psd", title, ylabel=ylabel)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=C_INK,
                    bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                    title="tunnel speed setpoint", alignment="left")
    leg.get_title().set_fontsize(9)
    leg.get_title().set_color(C_MUTED)
    finish(ax, fig, out)
    plt.close(fig)


def plot_peaks(rows: list[dict], mist: list[dict], k: float, se: float,
               out: Path, label: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    every = rows + mist
    xs = np.array([0, max(r["speed"] for r in every) * 1.05])

    ax.plot(xs, HZ_PER_MPS * xs / 1e6, color=CAT[1], lw=2, zorder=2,
            label=f"predicted: {HZ_PER_MPS/1e6:.3f} MHz per m/s "
                  f"(cos {BEAM_ANGLE_DEG:g}°)")
    ax.plot(xs, k * xs / 1e6, color=C_MUTED, lw=1.2, ls=(0, (5, 4)), zorder=2,
            label=f"fit through origin, dense fog: {k/1e6:.3f} ± {se/1e6:.3f} "
                  "MHz per m/s")
    fit = [r for r in rows if not r["approx"]]
    out_of_fit = [r for r in rows if r["approx"]]
    ax.plot([r["speed"] for r in fit], [r["centroid"] / 1e6 for r in fit], "o",
            ms=7, color=CAT[0], mec=C_SURFACE, mew=1.5, zorder=3,
            label="peak centroid, dense fog")
    if out_of_fit:
        ax.plot([r["speed"] for r in out_of_fit],
                [r["centroid"] / 1e6 for r in out_of_fit], "o", ms=7,
                mfc=C_SURFACE, mec=CAT[0], mew=1.8, zorder=3,
                label="peak centroid, dense fog, approximate setpoint (not fitted)")
    if mist:
        ax.plot([r["speed"] for r in mist], [r["centroid"] / 1e6 for r in mist],
                "D", ms=7, color=CAT[2], mec=C_SURFACE, mew=1.5, zorder=4,
                label="peak centroid, mist")

    ax.set_xlabel("wind-tunnel speed setpoint (m/s)", color=C_INK)
    ax.set_ylabel("Doppler peak frequency (MHz)", color=C_INK)
    ax.set_xlim(0, xs[1])
    ax.set_ylim(0, max(max(r["centroid"] for r in every), k * xs[1]) / 1e6 * 1.05)
    # Same quantity in other units, not a second measure: U = f_D / 1.057 MHz.
    sec = ax.secondary_yaxis("right", functions=(lambda f: f * 1e6 / HZ_PER_MPS,
                                                 lambda u: u * HZ_PER_MPS / 1e6))
    sec.set_ylabel("tunnel speed from Doppler peak (m/s)", color=C_INK)
    sec.spines["right"].set_color(C_GRID)
    sec.tick_params(colors=C_MUTED, labelsize=9)

    ax.set_title(f"{label} Doppler peak vs wind-tunnel speed setpoint", loc="left",
                 fontsize=11, color=C_INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=C_INK)
    ax.grid(True, color=C_GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    finish(ax, fig, out)
    plt.close(fig)


def plot_averaging(caps: list[dict], f: np.ndarray, out: Path, title: str,
                   fmax: float = 15e6) -> list[dict]:
    """One panel per number of captures averaged (1, 2, ... all, in time
    order), each PSD relative to its own floor. Returns each panel's peak."""
    import matplotlib.pyplot as plt

    n = len(caps)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.6), sharey=True)
    fig.patch.set_facecolor(C_SURFACE)
    m = (f >= F_MIN) & (f <= fmax)
    panels = []
    for i, ax in enumerate(np.atleast_1d(axes)):
        used = caps[:i + 1]
        # Averaged in linear power, weighted by records: one long capture.
        records = sum(c["n"] for c in used)
        psd = sum(c["psd"] * c["n"] for c in used) / records
        p = locate(f, psd)
        db = 10 * np.log10(psd / p["floor"])
        # Scatter about the floor in the same local noise band as
        # noise_vs_windspeed.txt, so the figure and the report agree.
        n = measure(f, psd, None, peak=p, floor=p["floor"])
        sd = n["rel_sd_dB"]
        sig = (n["rel_peak_dB"] - n["rel_mean_dB"]) / sd
        panels.append({"captures": len(used), "records": records,
                       "centroid": p["centroid"], "ratio_dB": p["ratio_dB"],
                       "noise_sd_dB": sd, "sigmas": sig})

        ax.set_facecolor(C_SURFACE)
        ax.axvline(p["centroid"], color=C_MUTED, lw=1, ls=(0, (2, 3)), zorder=2)
        ax.plot(f[m], db[m], color=CAT[0], lw=1.4, zorder=3)
        # Stats go in the title: a spur at 12.5 MHz leaves no clear corner.
        style_axes(ax, "psd", f"{len(used)} capture{'s' * (len(used) > 1)}"
                              f"{' averaged' if len(used) > 1 else ''} "
                              f"({records} records)\n"
                              f"peak {p['ratio_dB']:.2f} dB at "
                              f"{p['centroid']/1e6:.2f} MHz, floor sd {sd:.3f} dB "
                              f"({sig:.1f} sd)",
                   ylabel="PSD / local noise floor (dB)")
        if i:
            ax.set_ylabel("")                     # shared with the first panel
        ax.set_xlim(0, fmax)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=C_INK)
    finish(ax, fig, out)
    plt.close(fig)
    return panels


def measure_all(rows: list[dict], nperseg: int, f: np.ndarray | None = None
                ) -> np.ndarray:
    """PSD and Doppler peak of each row's capture, added to the row."""
    for r in rows:
        d = psd_of(r["path"], nperseg)
        if f is not None and not np.array_equal(d["f"], f):
            raise ValueError(f"{r['path'].name}: frequency axis differs")
        f = d["f"]
        r.update(psd=d["psd"], n=d["n"], capture=r["path"].stem[:15],
                 **locate(f, d["psd"]))
    return f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--date", default="20261012")
    ap.add_argument("--label", default="D", help="name for the series in the figures")
    ap.add_argument("--nperseg", type=int, default=2500,
                    help="Welch segment length; 2500 gives 50 kHz bins, as the "
                         "captures were first processed with")
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()

    rows = select(args.data_dir, args.date)
    if not rows:
        print(f"no wind-tunnel captures in {args.data_dir} on {args.date}")
        return 1
    mist = select(args.data_dir, args.date, mist=True)
    f = measure_all(rows, args.nperseg)
    measure_all(mist, args.nperseg, f)
    for r in rows:
        r["fog"] = "dense fog"
    for r in mist:
        r["fog"] = "mist"

    fit = [r for r in rows if not r["approx"]]
    k, se, r2 = fit_origin(np.array([r["speed"] for r in fit]),
                           np.array([r["centroid"] for r in fit]))

    print(f"{args.label}: {len(rows)} dense-fog and {len(mist)} mist captures, "
          f"Welch nperseg={args.nperseg} (df = {f[1]/1e3:g} kHz)")
    print(f"  {'U':>6} {'fog':>9} {'capture':>16} {'centroid':>10} {'argmax':>10} "
          f"{'FWHM':>13} {'above floor':>11} {'U from peak':>11}")
    for r in rows + mist:
        print(f"  {speed_label(r):>8} {r['fog']:>9} {r['capture']:>16} "
              f"{r['centroid']/1e6:7.3f} MHz {r['argmax']/1e6:7.3f} MHz "
              f"{r['fwhm_lo']/1e6:5.2f}-{r['fwhm_hi']/1e6:<5.2f}MHz "
              f"{r['ratio_dB']:8.2f} dB {r['centroid']/HZ_PER_MPS:8.2f} m/s")
    cos_eff = k * WAVELENGTH / 2
    print(f"  fit (dense fog, approximate setpoints excluded): "
          f"{k/1e6:.4f} +/- {se/1e6:.4f} "
          f"MHz per m/s, R^2 = {r2:.5f}")
    print(f"  predicted {HZ_PER_MPS/1e6:.4f} MHz per m/s; measured/predicted = "
          f"{k/HZ_PER_MPS:.3f}; effective beam angle "
          f"{np.degrees(np.arccos(min(cos_eff, 1))):.1f} deg")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = args.out_dir / "peak_vs_windtunnel_speed.csv"
    keys = ["speed", "fog", "approx", "capture", "centroid", "argmax", "fwhm_lo",
            "fwhm_hi", "ratio_dB"]
    with table.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows({key: r[key] for key in keys} for r in rows + mist)
    print(f"table -> {table}")

    import matplotlib
    matplotlib.use("Agg")
    plot_overlay(rows, f, args.out_dir / "psd_windtunnel_overlay.png",
                 f"{args.label} wind-tunnel PSD")
    plot_overlay(rows, f, args.out_dir / "psd_windtunnel_overlay_floor.png",
                 f"{args.label} wind-tunnel PSD relative to its own noise floor",
                 y_of=lambda r: 10 * np.log10(r["psd"] / r["floor"]),
                 ylabel="PSD / local noise floor (dB)")
    plot_peaks(rows, mist, k, se, args.out_dir / "peak_vs_windtunnel_speed.png",
               args.label)

    avg = [{"path": p} for p in captures(args.data_dir, args.date, AVERAGING_TAG)]
    if len(avg) > 1:
        measure_all(avg, args.nperseg, f)
        tag = AVERAGING_TAG.replace("mps_mist", " m/s mist")
        panels = plot_averaging(
            avg, f, args.out_dir / f"psd_{AVERAGING_TAG}_averaging.png",
            f"{args.label} {tag}: PSD relative to its own noise floor, "
            "averaging successive captures")
        print(f"  {AVERAGING_TAG} averaging:")
        for p in panels:
            print(f"    {p['captures']} capture(s), {p['records']:5d} records: "
                  f"peak {p['ratio_dB']:.3f} dB at {p['centroid']/1e6:.3f} MHz, "
                  f"floor sd {p['noise_sd_dB']:.3f} dB "
                  f"-> {p['sigmas']:.1f} sd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
