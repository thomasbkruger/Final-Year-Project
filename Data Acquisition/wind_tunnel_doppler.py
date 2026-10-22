"""wind_tunnel_doppler.py - D series: PSD overlay and Doppler peak vs tunnel speed.

Uses the 12 Oct wind-tunnel captures, one per speed setpoint (the earliest when
a speed was captured more than once), leaving out the mist runs. The 9 Oct
3.1 m/s capture is added as well.

These dense-fog captures are compared with the mist runs (tags ending _mist),
which are picked the same way, one per speed.

Six figures, in the style of psd_rpm.py and peak_vs_rpm.py:

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
  * psd_windtunnel_cases.png - a grid with one panel per test case (dense fog
    and mist), each the raw PSD with its floor and located peak, for an
    appendix;
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

--set rotated runs the same analysis on the 16 Oct captures taken with the rig
rotated (tags ending _rotated), writing the peak table, the overlay pair and
the peak plot with a _rotated suffix. Only captures at 0.2 V/div are used -
the first few that morning were taken at 2 and 10 V/div, where the ADC step is
larger than the signal - and repeats at one speed are averaged in power rather
than the earliest taken, since 21 m/s needs its four captures to clear 5 sd.

    python wind_tunnel_doppler.py
    python wind_tunnel_doppler.py --set rotated
    python wind_tunnel_doppler.py --nperseg 12500
"""
from __future__ import annotations

import argparse
import csv
import json
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
# Repeated captures compared in psd_<tag>_averaging.png: (date, tag, the upper
# frequency shown). Both are weak returns that averaging lifts out of the noise.
AVERAGING = (("20261012", "5mps_mist", 15e6),
             ("20261016", "21mps_rotated", 30e6))

# Mist cases shown in the appendix grid, psd_windtunnel_cases.png, each with
# all its captures averaged - one 5 m/s mist capture alone is below threshold.
GRID_MIST = ("5mps_mist",)

# 5mps, 11-8mps (11.8 m/s), 15ishmps (an approximate setpoint, ~15 m/s)
SPEED_TAG = re.compile(r"^(\d+)(?:-(\d+))?(ish)?mps$")

# Captures from other dates, with the speed to plot them at.
EXTRA = {"20261009_135015_3-1mps": 3.1}

# The two rig configurations. suffix: what the tags end in; vdiv: the only
# vertical scale accepted (None for any); average: repeats at a speed averaged
# (else the earliest kept); name: what the legends and reports call the set;
# out: added to output file names.
SETS = {
    "unrotated": {"date": "20261012", "suffix": "", "extra": EXTRA, "vdiv": None,
                  "average": False, "mist": True, "name": "dense fog", "out": ""},
    "rotated": {"date": "20261016", "suffix": "_rotated", "extra": {}, "vdiv": 0.2,
                "average": True, "mist": False, "name": "rotated",
                "out": "_rotated"},
}


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


def scale_of(p: Path) -> float | None:
    """The vertical scale a capture was taken at, V/div, as the scope read it
    back; None while the capture is still being written (no .json yet)."""
    meta = p.with_suffix(".json")
    if not meta.exists():
        return None
    return float(json.loads(meta.read_text())["scope_readback"]["volts_per_div"])


def select(data_dir: Path, date: str, mist: bool = False, suffix: str = "",
           extra: dict | None = None, vdiv: float | None = None,
           average: bool = False) -> list[dict]:
    """One row per speed from `date`: the dense-fog runs plus `extra` or, with
    mist=True, the mist runs alone. Only tags ending in `suffix` are taken,
    and only captures at `vdiv` V/div when it is given. Each row's "paths" are
    the captures at that speed in time order - all of them with average=True,
    else just the earliest."""
    chosen: dict[float, dict] = {}
    for p in sorted(data_dir.glob(f"{date}_*.npz")):   # time order
        if p.stem.endswith(("_psd", "_volts")):
            continue
        tag = p.stem[16:]
        if not tag.endswith(suffix):
            continue
        tag = tag.removesuffix(suffix) if suffix else tag
        if tag.endswith("_mist") != mist:
            continue
        s = speed_of(tag.removesuffix("_mist"))
        scale = scale_of(p)
        # 0 m/s is a no-flow reference, not a speed: it has no Doppler peak.
        if (s is None or s[0] == 0 or scale is None
                or (vdiv is not None and scale != vdiv)):
            continue
        row = chosen.setdefault(s[0], {"paths": [], "speed": s[0],
                                       "approx": s[1], "extra": False})
        if average or not row["paths"]:
            row["paths"].append(p)
    if not mist:
        for stem, speed in (extra or {}).items():
            chosen[speed] = {"paths": [data_dir / f"{stem}.npz"], "speed": speed,
                             "approx": False, "extra": True}
    for row in chosen.values():
        row["path"] = row["paths"][0]
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
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"plot -> {out}")


def draw_overlay(ax, rows: list[dict], f: np.ndarray, y_of=lambda r: r["psd"],
                 pad: float = 0.1) -> None:
    """Every speed on one linear axis; y_of picks what is plotted per row."""
    # The date's own captures take the categorical slots in speed order. One
    # more series - a capture from another date, or a ninth speed - is drawn
    # dashed in muted ink, rather than inventing a ninth hue.
    own = [r for r in rows if not r["extra"]]
    colour = dict(zip((id(r) for r in own), CAT))
    if len(rows) - len(colour) > 1:
        raise ValueError(f"{len(rows)} series but only {len(CAT)} categorical "
                         "colours and one dashed; split the figure rather than "
                         "inventing hues")
    m = f > 0
    ax.set_facecolor(C_SURFACE)
    for r in rows:
        style = ({"color": colour[id(r)]} if id(r) in colour
                 else {"color": C_MUTED, "ls": (0, (5, 4))})
        ax.plot(f[m], y_of(r)[m], lw=1.2, zorder=3,
                label=speed_label(r), **style)
    fmax = max(r["centroid"] for r in rows) * 1.25
    ax.set_xlim(0, fmax)
    # Ranged to the spectrum above F_MIN: the near-DC spike would otherwise set
    # the scale and flatten the Doppler peaks.
    shown = (f >= F_MIN) & (f <= fmax)
    lo = min(y_of(r)[shown].min() for r in rows)
    hi = max(y_of(r)[shown].max() for r in rows)
    ax.set_ylim(lo - pad * (hi - lo), hi + pad * (hi - lo))


def plot_overlay(rows: list[dict], f: np.ndarray, out: Path, title: str,
                 y_of=lambda r: r["psd"], ylabel: str | None = None) -> None:
    import matplotlib.pyplot as plt

    import figstyle

    fig, ax = plt.subplots(figsize=(figstyle.TEXT_WIDTH, 3.8))
    fig.patch.set_facecolor(C_SURFACE)
    draw_overlay(ax, rows, f, y_of)
    style_axes(ax, "psd", title, ylabel=ylabel)
    leg = ax.legend(frameon=False, loc="upper left", labelcolor=C_INK,
                    bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                    title="tunnel speed setpoint", alignment="left")
    leg.get_title().set_color(C_MUTED)
    finish(ax, fig, out)
    plt.close(fig)


def plot_overlay_pair(rows: list[dict], f: np.ndarray, out: Path) -> None:
    """The raw and floor-normalised overlays side by side, with the legend
    shared beneath them. Sized for the full text width (the report includes
    it slightly wider than \\linewidth)."""
    import matplotlib.pyplot as plt

    import figstyle

    fig, (raw, rel) = plt.subplots(1, 2, figsize=(7.0, 4.1))
    fig.patch.set_facecolor(C_SURFACE)
    # Raw PSD in units of 1e-12 V^2/Hz, so the scale sits in the axis label
    # rather than in an offset above the axis.
    draw_overlay(raw, rows, f, y_of=lambda r: r["psd"] / 1e-12, pad=0.02)
    style_axes(raw, "psd", ylabel=r"PSD ($10^{-12}$ V$^2$/Hz)")
    draw_overlay(rel, rows, f, y_of=lambda r: 10 * np.log10(r["psd"] / r["floor"]),
                 pad=0.02)
    style_axes(rel, "psd", ylabel="PSD / local noise floor (dB)")
    for ax, title in ((raw, "Raw signal PSD"), (rel, "Normalised to noise floor")):
        ax.set_title(title, loc="left", color=C_INK)
        ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1, 10_000_000))

    handles, labels = raw.get_legend_handles_labels()
    leg = fig.legend(handles, labels, frameon=False, ncol=5,
                     loc="lower center", labelcolor=C_INK, columnspacing=1.2,
                     handlelength=1.8, title="tunnel speed setpoint",
                     fontsize=figstyle.FS_NOTE, title_fontsize=figstyle.FS_NOTE)
    leg.get_title().set_color(C_MUTED)
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.5)
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"plot -> {out}")
    plt.close(fig)


def plot_case_grid(rows: list[dict], mist: list[dict], f: np.ndarray,
                   out: Path, ncols: int = 3) -> None:
    """One panel per test case, for the appendix: the raw PSD, the running-
    median floor its peak was found against, and the located centroid. Each
    panel is scaled to its own data, since the floors differ by up to 2x and
    the mist peaks are only a few per cent high."""
    import matplotlib.pyplot as plt

    import figstyle

    cases = ([(r, "dense fog", CAT[0]) for r in rows]
             + [(r, "mist", CAT[2]) for r in mist])
    nrows = -(-(len(cases) + 1) // ncols)       # one spare slot for the legend
    # Not sharex: it would hide the ticks of a column's last panel whenever
    # that panel is not on the bottom row. The x-limits are set alike below.
    # The full text width, and as tall as an appendix page allows.
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figstyle.TEXT_WIDTH + 0.4, 2.2 * nrows))
    fig.patch.set_facecolor(C_SURFACE)
    axes = axes.ravel()
    fmax = max(r["centroid"] for r, _, _ in cases) * 1.25
    m = (f >= F_MIN) & (f <= fmax)
    scale = 1e-12                                # plotted in 1e-12 V^2/Hz

    for i, (ax, (r, fog, colour)) in enumerate(zip(axes, cases)):
        ax.set_facecolor(C_SURFACE)
        ax.plot(f[m], r["psd"][m] / scale, color=colour, lw=1.2, zorder=3,
                label=f"PSD, {fog}")
        ax.plot(f[m], r["floor"][m] / scale, color=C_MUTED, lw=1.2,
                ls=(0, (5, 4)), zorder=4, label="median noise floor")
        ax.axvline(r["centroid"], color=C_INK, lw=1, ls=(0, (1, 2)), zorder=2,
                   label="peak centroid")
        lo, hi = r["psd"][m].min() / scale, r["psd"][m].max() / scale
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        ax.set_xlim(0, fmax)
        # Titles kept short enough to fit a third of the page; the colour and
        # the legend say dense fog or mist in full.
        date = f", {int(r['capture'][6:8])} Oct" if r["extra"] else ""
        short = "fog" if fog == "dense fog" else fog
        avg = f" ({r['n_caps']} avg.)" if r.get("n_caps", 1) > 1 else ""
        style_axes(ax, "psd", f"{speed_label(r)} {short}{date}{avg}\n"
                              f"{r['centroid']/1e6:.2f} MHz, "
                              f"+{r['ratio_dB']:.2f} dB",
                   ylabel=r"PSD ($10^{-12}$ V$^2$/Hz)")
        ax.title.set_size(figstyle.FS_NOTE)
        ax.set_xticks(range(0, int(fmax) + 1, 10_000_000))
        ax.tick_params(labelsize=figstyle.FS_NOTE, which="both")
        ax.xaxis.label.set_size(figstyle.FS_LEGEND)
        ax.yaxis.label.set_size(figstyle.FS_LEGEND)
        if i % ncols:
            ax.set_ylabel("")                     # left column carries it
        if i < len(cases) - ncols:
            ax.set_xlabel("")                     # bottom panel of each column
            ax.tick_params(labelbottom=False)

    # Both PSD colours, then the floor and centroid entries once.
    entries = {}
    for ax in (axes[0], axes[len(cases) - 1]):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(lab, h)
    order = sorted(entries, key=lambda lab: not lab.startswith("PSD"))
    handles, labels = [entries[lab] for lab in order], order
    spare = axes[len(cases)]
    spare.axis("off")
    spare.legend(handles, labels, frameon=False, fontsize=figstyle.FS_NOTE,
                 loc="center", labelcolor=C_INK, handlelength=1.6,
                 borderaxespad=0)
    for ax in axes[len(cases) + 1:]:
        ax.axis("off")
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"plot -> {out}")
    plt.close(fig)


def plot_peaks(rows: list[dict], mist: list[dict], k: float, se: float,
               out: Path, name: str = "dense fog") -> None:
    import matplotlib.pyplot as plt

    import figstyle

    fig, ax = plt.subplots(figsize=(figstyle.TEXT_WIDTH, 3.8))
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    every = rows + mist
    xs = np.array([0, max(r["speed"] for r in every) * 1.05])

    ax.plot(xs, HZ_PER_MPS * xs / 1e6, color=CAT[1], lw=2, zorder=2,
            label=f"predicted: {HZ_PER_MPS/1e6:.3f} MHz per m/s")
    ax.plot(xs, k * xs / 1e6, color=C_MUTED, lw=1.2, ls=(0, (5, 4)), zorder=2,
            label=f"fit, {name}: {k/1e6:.3f} ± {se/1e6:.3f} MHz per m/s")
    fit = [r for r in rows if not r["approx"]]
    out_of_fit = [r for r in rows if r["approx"]]
    ax.plot([r["speed"] for r in fit], [r["centroid"] / 1e6 for r in fit], "o",
            ms=7, color=CAT[0], mec=C_SURFACE, mew=1.5, zorder=4,
            label=f"peak centroid, {name}")
    if out_of_fit:
        ax.plot([r["speed"] for r in out_of_fit],
                [r["centroid"] / 1e6 for r in out_of_fit], "o", ms=7,
                mfc=C_SURFACE, mec=CAT[0], mew=1.8, zorder=4,
                label=f"peak centroid, {name}, approximate setpoint (not fitted)")
    if mist:
        ax.plot([r["speed"] for r in mist], [r["centroid"] / 1e6 for r in mist],
                # Under the dense-fog circles, which it would hide at 5 m/s.
                "D", ms=7, color=CAT[2], mec=C_SURFACE, mew=1.5, zorder=3,
                label="peak centroid, mist")

    ax.set_xlabel("wind-tunnel speed setpoint (m/s)", color=C_INK)
    ax.set_ylabel("Doppler peak frequency (MHz)", color=C_INK)
    ax.set_xlim(0, xs[1])
    ax.set_ylim(0, max(max(r["centroid"] for r in every), k * xs[1]) / 1e6 * 1.03)
    # Same quantity in other units, not a second measure: U = f_D / 1.057 MHz.
    sec = ax.secondary_yaxis("right", functions=(lambda f: f * 1e6 / HZ_PER_MPS,
                                                 lambda u: u * HZ_PER_MPS / 1e6))
    sec.set_ylabel("tunnel speed from\nDoppler peak (m/s)", color=C_INK)
    sec.spines["right"].set_color(C_GRID)
    sec.tick_params(colors=C_MUTED)

    # Above the lines, where the top left of the plot is empty; the labels are
    # kept short so the legend stays clear of the 21 m/s point.
    ax.legend(frameon=False, loc="upper left", labelcolor=C_INK,
              fontsize=figstyle.FS_NOTE)
    ax.grid(True, color=C_GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED)
    finish(ax, fig, out)
    plt.close(fig)


def plot_averaging(caps: list[dict], f: np.ndarray, out: Path,
                   fmax: float = 15e6) -> list[dict]:
    """One panel per number of captures averaged (1, 2, ... all, in time
    order), each PSD relative to its own floor. Returns each panel's peak."""
    import matplotlib.pyplot as plt

    import figstyle

    # Up to three panels fit across the page; more go two to a row, since
    # four side by side run their titles into each other.
    ncols = len(caps) if len(caps) <= 3 else 2
    nrows = -(-len(caps) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 3.3 * nrows), sharey=True)
    fig.patch.set_facecolor(C_SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(caps):]:
        ax.axis("off")
    m = (f >= F_MIN) & (f <= fmax)
    panels = []
    for i, ax in enumerate(axes[:len(caps)]):
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
        ax.axvline(p["centroid"], color=C_MUTED, lw=1, ls=(0, (2, 3)), zorder=2,
                   label="peak centroid")
        # The detection threshold of noise_vs_windspeed.txt: the noise band's
        # mean plus five of its standard deviations, in the same dB units.
        ax.axhline(n["rel_mean_dB"] + 5 * sd, color=CAT[1], lw=1.2,
                   ls=(0, (5, 4)), zorder=4, label=r"mean + 5$\sigma$ threshold")
        ax.plot(f[m], db[m], color=CAT[0], lw=1.4, zorder=3)
        # Stats go in the title: a spur at 12.5 MHz leaves no clear corner.
        # Kept to two short lines so three panels fit across the page; the
        # centroid's frequency is marked by the dotted line.
        style_axes(ax, "psd", f"{len(used)} capture{'s' * (len(used) > 1)}, "
                              f"{records} rec.\n"
                              f"peak +{p['ratio_dB']:.2f} dB\n"
                              f"floor $\\sigma$ = {sd:.3f} dB",
                   ylabel="PSD / local noise floor (dB)")
        ax.title.set_size(figstyle.FS_NOTE)
        if i % ncols:
            ax.set_ylabel("")                     # shared with the row's first
        if i < len(caps) - ncols:
            ax.set_xlabel("")                     # the bottom row carries it
        ax.set_xlim(0, fmax)
        # 5 MHz ticks crowd a narrow panel past 15 MHz.
        ax.set_xticks(range(0, int(fmax) + 1,
                            5_000_000 if fmax <= 15e6 else 10_000_000))
    # One legend for every panel, beneath them: the panels have no free corner.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(labels), loc="lower center",
               labelcolor=C_INK, fontsize=figstyle.FS_NOTE, handlelength=1.8)
    fig.tight_layout(rect=(0, 0.35 / (3.3 * nrows + 0.35), 1, 1))
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"plot -> {out}")
    plt.close(fig)
    return panels


def measure_all(rows: list[dict], nperseg: int, f: np.ndarray | None = None
                ) -> np.ndarray:
    """PSD and Doppler peak of each row, added to the row. A row with several
    "paths" gets their PSDs averaged in linear power, weighted by records -
    one long capture."""
    for r in rows:
        paths = r["paths"] if "paths" in r else [r["path"]]
        runs = []
        for p in paths:
            d = psd_of(p, nperseg)
            if f is not None and not np.array_equal(d["f"], f):
                raise ValueError(f"{p.name}: frequency axis differs")
            f = d["f"]
            runs.append(d)
        n = sum(d["n"] for d in runs)
        psd = sum(d["psd"] * d["n"] for d in runs) / n
        capture = paths[0].stem[:15] + (f"..{paths[-1].stem[9:15]}"
                                        if len(paths) > 1 else "")
        r.update(psd=psd, n=n, n_caps=len(paths), capture=capture,
                 **locate(f, psd))
    return f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--set", choices=SETS, default="unrotated",
                    help="which rig configuration to analyse")
    ap.add_argument("--date", help="YYYYmmdd; default is the set's own date")
    ap.add_argument("--label", default="D", help="name for the series in the figures")
    ap.add_argument("--nperseg", type=int, default=2500,
                    help="Welch segment length; 2500 gives 50 kHz bins, as the "
                         "captures were first processed with")
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()

    cfg = SETS[args.set]
    date = args.date or cfg["date"]
    pick = {"suffix": cfg["suffix"], "vdiv": cfg["vdiv"], "average": cfg["average"]}
    rows = select(args.data_dir, date, extra=cfg["extra"], **pick)
    if not rows:
        print(f"no {args.set} wind-tunnel captures in {args.data_dir} on {date}")
        return 1
    mist = select(args.data_dir, date, mist=True, **pick) if cfg["mist"] else []
    f = measure_all(rows, args.nperseg)
    measure_all(mist, args.nperseg, f)
    for r in rows:
        r["fog"] = cfg["name"]
    for r in mist:
        r["fog"] = "mist"

    fit = [r for r in rows if not r["approx"]]
    k, se, r2 = fit_origin(np.array([r["speed"] for r in fit]),
                           np.array([r["centroid"] for r in fit]))

    print(f"{args.label}, {args.set}: {len(rows)} {cfg['name']} and {len(mist)} mist "
          f"speeds, Welch nperseg={args.nperseg} (df = {f[1]/1e3:g} kHz)")
    print(f"  {'U':>6} {'fog':>9} {'capture':>22} {'centroid':>10} {'argmax':>10} "
          f"{'FWHM':>13} {'above floor':>11} {'U from peak':>11}")
    for r in rows + mist:
        print(f"  {speed_label(r):>8} {r['fog']:>9} {r['capture']:>22} "
              f"{r['centroid']/1e6:7.3f} MHz {r['argmax']/1e6:7.3f} MHz "
              f"{r['fwhm_lo']/1e6:5.2f}-{r['fwhm_hi']/1e6:<5.2f}MHz "
              f"{r['ratio_dB']:8.2f} dB {r['centroid']/HZ_PER_MPS:8.2f} m/s")
    cos_eff = k * WAVELENGTH / 2
    print(f"  fit ({cfg['name']}, approximate setpoints excluded): "
          f"{k/1e6:.4f} +/- {se/1e6:.4f} "
          f"MHz per m/s, R^2 = {r2:.5f}")
    print(f"  predicted {HZ_PER_MPS/1e6:.4f} MHz per m/s; measured/predicted = "
          f"{k/HZ_PER_MPS:.3f}; effective beam angle "
          f"{np.degrees(np.arccos(min(cos_eff, 1))):.1f} deg")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = cfg["out"]
    table = args.out_dir / f"peak_vs_windtunnel_speed{out}.csv"
    keys = ["speed", "fog", "approx", "capture", "n_caps", "centroid", "argmax",
            "fwhm_lo", "fwhm_hi", "ratio_dB"]
    with table.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows({key: r[key] for key in keys} for r in rows + mist)
    print(f"table -> {table}")

    import matplotlib
    matplotlib.use("Agg")
    import figstyle
    figstyle.use()
    plot_overlay_pair(rows, f, args.out_dir / f"psd_windtunnel_overlay_pair{out}.png")
    plot_peaks(rows, mist, k, se, args.out_dir / f"peak_vs_windtunnel_speed{out}.png",
               cfg["name"])
    if args.set != "unrotated":
        return 0
    # The unrotated set also has the single overlays, the appendix grid and
    # the averaging comparisons.
    plot_overlay(rows, f, args.out_dir / "psd_windtunnel_overlay.png",
                 f"{args.label} wind-tunnel PSD")
    plot_overlay(rows, f, args.out_dir / "psd_windtunnel_overlay_floor.png",
                 f"{args.label} wind-tunnel PSD relative to its own noise floor",
                 y_of=lambda r: 10 * np.log10(r["psd"] / r["floor"]),
                 ylabel="PSD / local noise floor (dB)")
    grid_mist = [{**r, "paths": captures(args.data_dir, date, r["path"].stem[16:])}
                 for r in mist if r["path"].stem[16:] in GRID_MIST]
    measure_all(grid_mist, args.nperseg, f)
    plot_case_grid(rows, grid_mist, f, args.out_dir / "psd_windtunnel_cases.png")

    for date, tag, fmax in AVERAGING:
        avg = [{"path": p} for p in captures(args.data_dir, date, tag)]
        if len(avg) < 2:
            continue
        measure_all(avg, args.nperseg, f)
        panels = plot_averaging(avg, f, args.out_dir / f"psd_{tag}_averaging.png",
                                fmax)
        print(f"  {tag} averaging:")
        for p in panels:
            print(f"    {p['captures']} capture(s), {p['records']:5d} records: "
                  f"peak {p['ratio_dB']:.3f} dB at {p['centroid']/1e6:.3f} MHz, "
                  f"floor sd {p['noise_sd_dB']:.3f} dB "
                  f"-> {p['sigmas']:.1f} sd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
