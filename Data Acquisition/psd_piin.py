"""psd_piin.py - PIIN measurement (B1) against the balanced sweep point (A2).

The B1 captures are repeats of one condition, so each acquisition session is
averaged into a single curve; the run-to-run spread is kept as a shaded min-max
envelope rather than hidden, because it is large. Sessions are plotted
separately - a later batch is a fresh alignment, not more of the same run, so
pooling them would hide any drift between sittings. The A2 capture at the same
drive current is the balanced reference, and A1 (laser off) is the electronic
floor.

    python psd_piin.py
    python psd_piin.py --ref A2_47mA --out piin_vs_47mA.png
    python psd_piin.py --session-gap 30      # minutes between sessions

Averaging is done on the PSDs in linear V^2/Hz, never in dB, matching
Welch.band_average.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from prepare_welch import load_run
from Welch import welch_psd

# dataviz reference palette, light mode. The reference keeps categorical slot 2;
# B1 sessions take the remaining slots in acquisition order.
C_SESSION = ["#2a78d6", "#1baf7a", "#b05ccc"]    # slots 1, 3, 4
C_REF = "#eb6834"                                # slot 2
C_INK = "#0b0b0b"
C_MUTED = "#52514e"
C_GRID = "#d8d7d3"
C_SURFACE = "#fcfcfb"
C_BAND = "#eceae6"


def capture_time(stem: str) -> datetime:
    """Acquisition time from the filename stem, YYYYmmdd_HHMMSS_tag."""
    return datetime.strptime(stem[:15], "%Y%m%d_%H%M%S")


def split_sessions(reps: list[tuple[datetime, np.ndarray]], gap_min: float) -> list[dict]:
    """Group repeats into sittings: a gap longer than gap_min starts a new one."""
    sessions: list[dict] = []
    for when, psd in sorted(reps):
        if sessions and (when - sessions[-1]["last"]).total_seconds() <= gap_min * 60:
            sessions[-1]["psds"].append(psd)
        else:
            sessions.append({"start": when, "psds": [psd]})
        sessions[-1]["last"] = when
    for s in sessions:
        s["psds"] = np.array(s["psds"])
        s["label"] = s["start"].strftime("%d %b %H:%M")
    return sessions


def collect(data_dir: Path, series: str, ref_tag: str, nperseg: int,
            gap_min: float) -> dict:
    """PSDs of the B1 repeats by session, the reference capture, and the floor."""
    reps, ref, floor, f_axis = [], None, None, None
    for p in sorted(data_dir.glob("*.npz")):
        if p.stem.endswith(("_volts", "_psd")):
            continue
        run = load_run(p)
        tag = run.meta.get("tag", p.stem)
        if not (tag.startswith(series) or tag == ref_tag or tag.startswith("A1")):
            continue
        f, psd, _ = welch_psd(run.v, run.fs, nperseg=nperseg)
        if f_axis is None:
            f_axis = f
        elif not np.array_equal(f, f_axis):
            raise ValueError(f"{p.name}: frequency axis differs; cannot overlay")
        if tag.startswith(series):
            reps.append((capture_time(p.stem), psd))
        elif tag == ref_tag:
            ref = psd
        else:
            floor = psd
    return {"f": f_axis, "sessions": split_sessions(reps, gap_min),
            "reps": np.array([psd for _, psd in reps]), "ref": ref, "floor": floor}


def plot(d: dict, band: tuple[float, float], out: Path, series: str,
         ref_tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = d["f"]
    m = f > 0
    nv = lambda psd: np.sqrt(psd[m]) * 1e9        # V^2/Hz -> nV/rtHz

    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)

    ax.axvspan(*band, color=C_BAND, lw=0, zorder=0)
    ax.text(np.sqrt(band[0] * band[1]), 0.965,
            f"{band[0]/1e6:g}-{band[1]/1e6:g} MHz band", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8.5, color=C_MUTED)

    if d["floor"] is not None:
        ax.loglog(f[m], nv(d["floor"]), color=C_MUTED, lw=1.1, ls=(0, (5, 4)),
                  label="laser off (electronic floor)", zorder=2)
    if d["ref"] is not None:
        ax.loglog(f[m], nv(d["ref"]), color=C_REF, lw=1.5,
                  label=f"balanced, {ref_tag}", zorder=3)

    for n, sess in enumerate(d["sessions"]):
        colour = C_SESSION[n % len(C_SESSION)]
        psds = sess["psds"]
        if len(psds) > 1:
            ax.fill_between(f[m], nv(psds.min(axis=0)), nv(psds.max(axis=0)),
                            color=colour, alpha=0.15, lw=0, zorder=3 + n)
        ax.loglog(f[m], nv(psds.mean(axis=0)), color=colour, lw=1.8, zorder=5 + n,
                  label=f"{series} {sess['label']}, mean of {len(psds)}")
    ax.fill_between([], [], [], color=C_SESSION[0], alpha=0.15, lw=0,
                    label="session spread (min-max)")

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

    # Curves span the axes top to bottom, so the legend sits outside the data.
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=C_INK,
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"plot -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--series", default="B1")
    ap.add_argument("--ref", default="A2_48mA", help="tag of the balanced reference")
    ap.add_argument("--band", nargs=2, type=float, default=[5e6, 20e6])
    ap.add_argument("--nperseg", type=int, default=2500)
    ap.add_argument("--session-gap", type=float, default=30,
                    help="minutes between captures that starts a new session")
    ap.add_argument("--out", type=Path, default=Path("psd_piin.png"))
    args = ap.parse_args()

    d = collect(args.data_dir, args.series, args.ref, args.nperseg, args.session_gap)
    if not len(d["reps"]):
        print(f"no {args.series} captures in {args.data_dir}")
        return 1
    if d["ref"] is None:
        print(f"note: reference tag {args.ref} not found; plotting {args.series} alone")

    f, sel = d["f"], (d["f"] >= args.band[0]) & (d["f"] <= args.band[1])
    mean = d["reps"].mean(axis=0)
    print(f"band {args.band[0]/1e6:g}-{args.band[1]/1e6:g} MHz, nperseg={args.nperseg}")
    for name, psd in (("floor (A1)", d["floor"]), (f"ref ({args.ref})", d["ref"]),
                      (f"{args.series} mean", mean)):
        if psd is not None:
            b = psd[sel].mean()
            print(f"  {name:20s} {b:.4e} V^2/Hz   {np.sqrt(b)*1e9:8.1f} nV/rtHz")
    for sess in d["sessions"]:
        per_rep = sess["psds"][:, sel].mean(axis=1)
        b = sess["psds"].mean(axis=0)[sel].mean()
        print(f"  session {sess['label']} (n={len(per_rep)}): "
              f"mean {np.sqrt(b)*1e9:.0f}, runs "
              + "/".join(f"{np.sqrt(v)*1e9:.0f}" for v in per_rep) + " nV/rtHz")
    if d["ref"] is not None:
        print(f"  {args.series} mean / reference: "
              f"{mean[sel].mean()/d['ref'][sel].mean():.1f}x in power, "
              f"{np.sqrt(mean[sel].mean()/d['ref'][sel].mean()):.1f}x in amplitude")

    plot(d, tuple(args.band), args.out, args.series, args.ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
