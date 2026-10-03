"""psd_vs_current.py - band-averaged PSD against laser drive current.

Answers one question: how does the 5-20 MHz noise floor grow with laser drive
current? One detector configuration per figure, chosen with --detector.

    python psd_vs_current.py                      # balanced (A2)
    python psd_vs_current.py --detector single    # single-ended (A3)
    python psd_vs_current.py --band 5e6 20e6 --out psd_vs_current.png

The drive current is taken from the capture *tag*, not from
config.metadata.laser_current_mA: several runs were saved with the previous
run's metadata still in place, so the tag is the only reliable record.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from prepare_welch import load_run
from Welch import band_average, welch_psd

# dataviz reference palette, light mode, categorical slots 1-3 (validated all-pairs).
C_BALANCED = "#2a78d6"   # slot 1 blue
C_SINGLE = "#eb6834"     # slot 2 orange
C_SINGLE2 = "#1baf7a"    # slot 3 aqua
C_INK = "#0b0b0b"
C_MUTED = "#52514e"
C_GRID = "#d8d7d3"

TAG_CURRENT = re.compile(r"_(\d+)_?mA", re.IGNORECASE)

# Corrections applied after acquisition. The capture tagged A2_47mA is the one
# kept for the 48 mA point; the separate A2_48mA capture is dropped rather than
# plotted as a second point at nearly the same current. Both live here, and
# every figure imports resolve_tag, so no figure can disagree with another.
EXCLUDE_TAGS = {"A2_48mA"}
RETAG = {"A2_47mA": "A2_48mA"}


def resolve_tag(tag: str) -> str | None:
    """The tag a capture should be plotted under, or None to drop the capture."""
    if tag in EXCLUDE_TAGS:
        return None
    return RETAG.get(tag, tag)


def current_from_tag(tag: str) -> int | None:
    """Drive current in mA, parsed from the capture tag."""
    m = TAG_CURRENT.search(tag)
    return int(m.group(1)) if m else None


def collect(data_dir: Path, band: tuple[float, float], nperseg: int) -> dict:
    """Band-average every capture, grouped by detector configuration."""
    groups: dict[str, list] = {}
    floor = None
    for p in sorted(data_dir.glob("*.npz")):
        if p.stem.endswith(("_volts", "_psd")):
            continue
        run = load_run(p)
        tag = resolve_tag(run.meta.get("tag", p.stem))
        if tag is None:
            continue
        f, _, per = welch_psd(run.v, run.fs, nperseg=nperseg)
        psd, sem = band_average(f, per, *band)

        if tag.startswith("A1"):                    # laser off: electronic floor
            floor = (psd, sem)
        elif tag.startswith("A2"):
            groups.setdefault("balanced", []).append((current_from_tag(tag), psd, sem))
        elif tag.startswith("A3"):
            key = run.meta["config"]["metadata"]["photodiodes"]   # single_pd1 / single_pd2
            groups.setdefault(key, []).append((current_from_tag(tag), psd, sem))
        # B-series is a separate experiment; not part of this comparison.

    for k in groups:
        groups[k] = np.array(sorted(groups[k]), dtype=float)
    return {"groups": groups, "floor": floor}


def plot(data: dict, band: tuple[float, float], out: Path, detector: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g, floor = data["groups"], data["floor"]
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    if floor is not None:
        ax.axhline(floor[0], color=C_MUTED, lw=1.0, ls=(0, (5, 4)), zorder=1)
        ax.text(1.5, floor[0], "electronic floor (laser off)", va="bottom",
                ha="left", fontsize=8.5, color=C_MUTED)

    if detector == "balanced" and "balanced" in g:
        # The duplicate 48 mA capture is dropped in collect(), via RETAG and
        # EXCLUDE_TAGS, so that every figure drops the same one.
        i, psd, sem = g["balanced"].T
        ax.errorbar(i, psd, yerr=sem, color=C_BALANCED, marker="o", ms=7, lw=1.8,
                    mec="#fcfcfb", mew=1.2, capsize=0, zorder=3)

    if detector == "single":
        if "single_pd1" in g:
            i, psd, sem = g["single_pd1"].T
            ax.errorbar(i, psd, yerr=sem, color=C_SINGLE, marker="s", ms=7, lw=1.8,
                        mec="#fcfcfb", mew=1.2, capsize=0, label="PD1", zorder=3)
        if "single_pd2" in g:                   # one-point check on the other diode
            i, psd, sem = g["single_pd2"].T
            ax.errorbar(i, psd, yerr=sem, color=C_SINGLE2, marker="D", ms=7, lw=0,
                        mec="#fcfcfb", mew=1.2, label="PD2", zorder=4)

    ax.set_xlabel("laser drive current (mA)", color=C_INK)
    ax.set_ylabel(f"PSD averaged {band[0]/1e6:g}-{band[1]/1e6:g} MHz  (V$^2$/Hz)", color=C_INK)
    ax.set_xlim(0, 52)
    ax.set_ylim(0, None)
    ax.grid(True, color=C_GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=9)

    sec = ax.secondary_yaxis("right", functions=(lambda v: np.sqrt(np.maximum(v, 0)) * 1e9,
                                                 lambda n: (n / 1e9) ** 2))
    sec.set_ylabel(r"voltage noise (nV/$\sqrt{\mathrm{Hz}}$)", color=C_MUTED)
    sec.tick_params(colors=C_MUTED, labelsize=9)

    if ax.get_legend_handles_labels()[1]:
        ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=C_INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"plot -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--band", nargs=2, type=float, default=[5e6, 20e6])
    ap.add_argument("--nperseg", type=int, default=2500)
    ap.add_argument("--detector", choices=("balanced", "single"), default="balanced")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.out is None:
        suffix = "" if args.detector == "balanced" else f"_{args.detector}"
        args.out = Path(f"psd_vs_current{suffix}.png")

    data = collect(args.data_dir, tuple(args.band), args.nperseg)
    floor = data["floor"]

    print(f"band {args.band[0]/1e6:g}-{args.band[1]/1e6:g} MHz, nperseg={args.nperseg}")
    if floor:
        print(f"electronic floor (laser off): {floor[0]:.4e} V^2/Hz")
    for key, arr in data["groups"].items():
        print(f"\n{key}")
        for i, psd, sem in arr:
            excess = psd - floor[0] if floor else float("nan")
            print(f"  {i:5.0f} mA  PSD {psd:.4e} +/- {sem:.1e}   "
                  f"excess {excess:.4e}   {np.sqrt(psd)*1e9:6.1f} nV/rtHz")

    plot(data, tuple(args.band), args.out, args.detector)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
