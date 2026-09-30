"""psd_spectra.py - overlaid Welch PSD spectra for one measurement series.

Where psd_vs_current.py collapses each capture to a single band-averaged number,
this keeps the whole spectrum: one curve per capture, overlaid, so the shape of
the noise (roll-off, spurs, where the excess over the electronic floor lives)
is visible rather than averaged away.

    python psd_spectra.py                       # A3, single-ended
    python psd_spectra.py --series A2           # balanced
    python psd_spectra.py --band 5e6 20e6 --out psd_spectra_A3.png

The laser-off capture (A1) is drawn as the electronic floor on every figure.
Drive current is parsed from the capture tag, for the reason given in
psd_vs_current.py: some runs carry the previous run's metadata.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from prepare_welch import load_run
from psd_vs_current import current_from_tag
from Welch import welch_psd

# dataviz reference palette, light mode. Drive current is an ordered quantity,
# so the sweep uses the blue ordinal ramp (steps 250-650, light -> dark); the
# second photodiode is a different entity, so it takes categorical slot 2.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#b05ccc"]   # categorical slots 1-4
C_SLOTS = CAT[1:]                                # slots 2-4, for extra variants
DASHES = [(0, (4, 2)), (0, (1, 1.6)), (0, (6, 2, 1, 2))]
C_INK = "#0b0b0b"
C_MUTED = "#52514e"
C_GRID = "#d8d7d3"
C_SURFACE = "#fcfcfb"
C_BAND = "#eceae6"


VARIANT_CURRENT = re.compile(r"_?\d+_?mA", re.IGNORECASE)


def variant_of(tag: str) -> str:
    """Tag with the drive current stripped: the measurement variant it belongs to."""
    return VARIANT_CURRENT.sub("", tag).strip("_") or tag


def ramp_colours(n: int) -> list[str]:
    """n evenly spaced steps of the ordinal ramp, darkening with the index."""
    if n >= len(RAMP):
        return [RAMP[min(i * (len(RAMP) - 1) // max(n - 1, 1), len(RAMP) - 1)]
                for i in range(n)]
    idx = np.linspace(0, len(RAMP) - 1, n).round().astype(int)
    return [RAMP[i] for i in idx]


def collect(data_dir: Path, series: str, nperseg: int) -> dict:
    """Full PSD of every capture in the series, plus the laser-off floor."""
    curves, floor = [], None
    for p in sorted(data_dir.glob("*.npz")):
        if p.stem.endswith(("_volts", "_psd")):
            continue
        run = load_run(p)
        tag = run.meta.get("tag", p.stem)
        if not (tag.startswith(series) or tag.startswith("A1")):
            continue
        f, psd, _ = welch_psd(run.v, run.fs, nperseg=nperseg)
        if tag.startswith("A1"):
            floor = (f, psd)
            continue
        curves.append({"f": f, "psd": psd, "mA": current_from_tag(tag), "tag": tag,
                       "variant": variant_of(tag), "date": p.stem[:8],
                       "time": f"{p.stem[9:11]}:{p.stem[11:13]}:{p.stem[13:15]}"})
    curves.sort(key=lambda c: (c["variant"], c["mA"] if c["mA"] is not None else -1,
                               c["date"], c["time"]))
    return {"curves": curves, "floor": floor}


def by_variant(curves: list[dict]) -> list[tuple[str, list[dict]]]:
    """Variants, the one with the widest current sweep first."""
    groups: dict[str, list[dict]] = {}
    for c in curves:
        groups.setdefault(c["variant"], []).append(c)
    return sorted(groups.items(),
                  key=lambda kv: -len({c["mA"] for c in kv[1]}))


def repeats_only(group: list[dict]) -> bool:
    """True when a variant is several captures at one drive current."""
    return len(group) > 1 and len({c["mA"] for c in group}) == 1


def plot(data: dict, band: tuple[float, float], out: Path, series: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves, floor = data["curves"], data["floor"]
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)

    ax.axvspan(*band, color=C_BAND, lw=0, zorder=0)
    ax.text(np.sqrt(band[0] * band[1]), 0.965, f"{band[0]/1e6:g}-{band[1]/1e6:g} MHz band",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=8.5, color=C_MUTED)

    # A current sweep is an ordered quantity, so the leading variant takes the
    # blue ordinal ramp. Repeats of one current are not ordered - they are
    # nominally the same measurement - so they take categorical hues instead.
    # Any further variant is its own entity: one hue and dash for the whole set.
    for n, (variant, group) in enumerate(by_variant(curves)):
        currents = sorted({c["mA"] for c in group})
        ramp = dict(zip(currents, ramp_colours(len(currents))))
        seen: set[float] = set()
        for j, c in enumerate(group):
            m = c["f"] > 0
            if n == 0 and repeats_only(group):
                ax.loglog(c["f"][m], np.sqrt(c["psd"][m]) * 1e9, lw=1.3,
                          color=CAT[j % len(CAT)], zorder=3,
                          label=f"{c['time']} ({c['mA']:.0f} mA)")
                continue
            colour = ramp[c["mA"]] if n == 0 else C_SLOTS[(n - 1) % len(C_SLOTS)]
            style = {} if n == 0 else {"ls": DASHES[(n - 1) % len(DASHES)]}
            reps = sum(1 for o in group if o["mA"] == c["mA"])
            label = None
            if c["mA"] not in seen:                 # one legend entry per current
                seen.add(c["mA"])
                label = f"{c['mA']:.0f} mA"
                if n:
                    label += f" - {variant}"
                if reps > 1:
                    label += f" (x{reps})"
            ax.loglog(c["f"][m], np.sqrt(c["psd"][m]) * 1e9, color=colour, lw=1.3,
                      label=label, zorder=3 + n, **style)

    if floor is not None:
        f, psd = floor
        m = f > 0
        ax.loglog(f[m], np.sqrt(psd[m]) * 1e9, color=C_MUTED, lw=1.1,
                  ls=(0, (5, 4)), label="laser off (electronic floor)", zorder=2)

    ax.set_xlabel("frequency (Hz)", color=C_INK)
    ax.set_ylabel(r"voltage noise (nV/$\sqrt{\mathrm{Hz}}$)", color=C_INK)
    ax.grid(True, which="major", color=C_GRID, lw=0.7, alpha=0.9)
    ax.grid(True, which="minor", color=C_GRID, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=9, which="both")

    lead = by_variant(curves)[0][1]
    legend_title = (f"{series} repeats, {lead[0]['mA']:.0f} mA" if repeats_only(lead)
                    else f"{series} laser drive current")

    # Curves fill the axes, so the legend sits outside rather than over the data.
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=C_INK,
                    bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                    title=legend_title, alignment="left")
    leg.get_title().set_fontsize(8.5)
    leg.get_title().set_color(C_MUTED)

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"plot -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--series", default="A3")
    ap.add_argument("--band", nargs=2, type=float, default=[5e6, 20e6])
    ap.add_argument("--nperseg", type=int, default=2500)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.out is None:
        args.out = Path(f"psd_spectra_{args.series}.png")

    data = collect(args.data_dir, args.series, args.nperseg)
    if not data["curves"]:
        print(f"no {args.series} captures in {args.data_dir}")
        return 1

    print(f"{args.series}: {len(data['curves'])} captures, nperseg={args.nperseg}")
    for c in data["curves"]:
        sel = (c["f"] >= args.band[0]) & (c["f"] <= args.band[1])
        print(f"  {c['date']} {c['tag']:24s} {c['mA'] or 0:3.0f} mA  "
              f"{np.sqrt(c['psd'][sel].mean())*1e9:6.1f} nV/rtHz in band")

    plot(data, tuple(args.band), args.out, args.series)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
