"""psd_piin.py - PIIN measurement (B1) against the balanced sweep point (A2).

The B1 captures are repeats of one condition, so they are pooled into a single
mean curve. The run-to-run spread is kept as a shaded min-max envelope rather
than hidden, because it is large. The A2 capture at the same drive current is
the balanced reference, and A1 (laser off) is the electronic floor.

Repeats were taken in more than one sitting. The sittings agreed to well within
the run-to-run spread, so they are averaged together; the per-session band
figures are still printed, so drift between sittings stays visible even though
the plot shows one curve.

    python psd_piin.py
    python psd_piin.py --ref A2_47mA --out piin_vs_47mA.png
    python psd_piin.py --session-gap 30      # minutes; affects the printout only
    python psd_piin.py --label B1            # show the raw tag prefix instead

Several series can be overlaid: each --series prefix is pooled into its own
mean curve, named by the matching --label entry.

    python psd_piin.py --series PIIN_ref PIIN_2m PIIN_3m PIIN_4m \
        --label "no matching" "2m patch cable" "3m patch cable" "4m patch cable" \
        --ref-label "A2 shot noise (balanced, 48 mA)" \
        --out figures/psd_piin_lengths.png

The captures are tagged B1, but the figure calls the series "B": the tag prefix
(--series) selects files and the label (--label) names them, so the display name
can change without widening the match to B2.

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

# dataviz reference palette, light mode, categorical slots in fixed order. The
# series take the first slots and the balanced reference takes the next one, so
# a single series keeps slot 1 (blue) against slot 2 (orange) as before.
C_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
C_INK = "#0b0b0b"
C_MUTED = "#52514e"
C_GRID = "#d8d7d3"
C_SURFACE = "#fcfcfb"
C_BAND = "#eceae6"


def capture_time(stem: str) -> datetime:
    """Acquisition time from the filename stem, YYYYmmdd_HHMMSS_tag."""
    return datetime.strptime(stem[:15], "%Y%m%d_%H%M%S")


def split_sessions(reps: list[tuple[datetime, np.ndarray]], gap_min: float) -> list[dict]:
    """Group repeats into sittings: a gap longer than gap_min starts a new one.

    Reporting only - the plot pools every repeat into one curve.
    """
    sessions: list[dict] = []
    for when, psd in sorted(reps, key=lambda r: r[0]):
        if sessions and (when - sessions[-1]["last"]).total_seconds() <= gap_min * 60:
            sessions[-1]["psds"].append(psd)
        else:
            sessions.append({"start": when, "psds": [psd]})
        sessions[-1]["last"] = when
    for s in sessions:
        s["psds"] = np.array(s["psds"])
        s["label"] = s["start"].strftime("%d %b %H:%M")
    return sessions


def collect(data_dir: Path, series: list[str], ref_tag: str, nperseg: int,
            gap_min: float) -> dict:
    """PSDs of every repeat of each series, the reference capture, and the floor."""
    reps = {s: [] for s in series}
    ref, floor, f_axis = None, None, None
    wanted = lambda tag: (tag.startswith(tuple(series)) or tag == ref_tag
                          or tag.startswith("A1"))
    for p in sorted(data_dir.glob("*.npz")):
        if p.stem.endswith(("_volts", "_psd")):
            continue
        # Screen on the filename (YYYYmmdd_HHMMSS_tag) before loading: skips
        # derived files such as *_stack.npz, which hold no raw records.
        try:
            capture_time(p.stem)
        except ValueError:
            continue
        if not wanted(p.stem[16:]):
            continue
        run = load_run(p)
        tag = run.meta.get("tag", p.stem)
        match = next((s for s in series if tag.startswith(s)), None)
        if not (match or tag == ref_tag or tag.startswith("A1")):
            continue
        f, psd, _ = welch_psd(run.v, run.fs, nperseg=nperseg)
        if f_axis is None:
            f_axis = f
        elif not np.array_equal(f, f_axis):
            raise ValueError(f"{p.name}: frequency axis differs; cannot overlay")
        if match:
            reps[match].append((capture_time(p.stem), psd))
        elif tag == ref_tag:
            ref = psd
        else:
            floor = psd
    return {"f": f_axis,
            "sessions": {s: split_sessions(r, gap_min) for s, r in reps.items()},
            "reps": {s: np.array([psd for _, psd in r]) for s, r in reps.items()},
            "ref": ref, "floor": floor}


def plot(d: dict, band: tuple[float, float], out: Path, labels: dict[str, str],
         ref_label: str, spread: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    import figstyle
    figstyle.use()

    f = d["f"]
    m = f > 0
    pw = lambda psd: psd[m]                       # plotted as stored, V^2/Hz
    single = len(d["reps"]) == 1

    fig, ax = plt.subplots(figsize=(figstyle.TEXT_WIDTH, 4.0))
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)

    ax.axvspan(*band, color=C_BAND, lw=0, zorder=0)
    ax.text(np.sqrt(band[0] * band[1]), 0.98,
            f"{band[0]/1e6:g}-{band[1]/1e6:g} MHz band", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=figstyle.FS_NOTE, color=C_MUTED)

    if d["floor"] is not None:
        ax.loglog(f[m], pw(d["floor"]), color=C_MUTED, lw=1.1, ls=(0, (5, 4)),
                  label="electronic floor (laser off)", zorder=2)
    if d["ref"] is not None:
        ax.loglog(f[m], pw(d["ref"]), color=C_SLOTS[len(d["reps"])], lw=1.5,
                  label=ref_label, zorder=3)

    for i, (s, reps) in enumerate(d["reps"].items()):
        c = C_SLOTS[i]
        if spread and len(reps) > 1:
            ax.fill_between(f[m], pw(reps.min(axis=0)), pw(reps.max(axis=0)),
                            color=c, alpha=0.15 if single else 0.10, lw=0, zorder=4,
                            label=f"run-to-run spread (min-max, n={len(reps)})"
                                  if single else None)
        name = (f"Experiment {labels[s]}: Total system noise, mean of {len(reps)}" if single
                else f"total noise, {labels[s]}")
        ax.loglog(f[m], pw(reps.mean(axis=0)), color=c, lw=1.8, zorder=5, label=name)

    # Ranged to what is drawn, envelopes included, with a small log margin.
    drawn = [c for c in (d["floor"], d["ref"]) if c is not None]
    for reps in d["reps"].values():
        drawn += [reps.min(axis=0), reps.max(axis=0)] if spread else [reps.mean(axis=0)]
    ax.set_ylim(min(pw(c).min() for c in drawn) / 1.15,
                max(pw(c).max() for c in drawn) * 1.15)

    ax.set_xlabel("frequency (MHz)", color=C_INK)
    ax.set_ylabel(r"PSD (V$^2$/Hz)", color=C_INK)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e6:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color=C_GRID, lw=0.7, alpha=0.9)
    ax.grid(True, which="minor", color=C_GRID, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, which="both")

    # Curves span the axes top to bottom, so the legend sits above the data,
    # in two columns to keep the plot the full width.
    ax.legend(frameon=False, loc="lower left", labelcolor=C_INK, ncol=2,
              bbox_to_anchor=(0, 1.0), borderaxespad=0.3, columnspacing=1.2,
              fontsize=figstyle.FS_NOTE)
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"plot -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--series", nargs="+", default=["B1"],
                    help="tag prefix(es) that select the captures; one mean "
                         "curve per prefix")
    ap.add_argument("--label", nargs="+", default=None,
                    help="name for each series in the figure and printout; kept "
                         "separate from --series so selection stays exact "
                         "(default: B for the default series, else the prefix)")
    ap.add_argument("--ref", default="A2_48mA", help="tag of the balanced reference")
    ap.add_argument("--ref-label", default=None,
                    help="legend entry for the reference (default: 'balanced, REF')")
    ap.add_argument("--band", nargs=2, type=float, default=[5e6, 20e6])
    ap.add_argument("--nperseg", type=int, default=2500)
    ap.add_argument("--session-gap", type=float, default=30,
                    help="minutes between captures that starts a new session "
                         "(printed breakdown only; the plot pools all repeats)")
    ap.add_argument("--no-spread", action="store_true",
                    help="omit the min-max run-to-run envelopes")
    ap.add_argument("--out", type=Path, default=Path("figures/psd_piin.png"))
    ap.add_argument("--txt", type=Path, default=None,
                    help="results file (default: the --out name with .txt, in "
                         "the working directory)")
    args = ap.parse_args()

    if args.label is None:
        args.label = ["B"] if args.series == ["B1"] else list(args.series)
    if len(args.label) != len(args.series):
        ap.error("--label needs one entry per --series prefix")
    labels = dict(zip(args.series, args.label))
    ref_label = args.ref_label or f"balanced, {args.ref}"

    d = collect(args.data_dir, args.series, args.ref, args.nperseg, args.session_gap)
    for s in args.series:
        if not len(d["reps"][s]):
            print(f"no {s} captures in {args.data_dir}")
            return 1
    if d["ref"] is None:
        print(f"note: reference tag {args.ref} not found; plotting without it")

    sel = (d["f"] >= args.band[0]) & (d["f"] <= args.band[1])
    lines = [
        "Total system noise (PIIN) vs balanced reference - psd_piin.py",
        f"generated: {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}",
        "series:    " + ", ".join(f"{labels[s]} ({s}*, n={len(d['reps'][s])})"
                                  for s in args.series),
        f"band {args.band[0]/1e6:g}-{args.band[1]/1e6:g} MHz, nperseg={args.nperseg}; "
        "band averages taken on linear PSD",
        "",
    ]
    for name, psd in (("floor (A1)", d["floor"]), (f"ref ({args.ref})", d["ref"])):
        if psd is not None:
            b = psd[sel].mean()
            lines.append(f"  {name:28s} {b:.4e} V^2/Hz   {np.sqrt(b)*1e9:8.1f} nV/rtHz")

    for s in args.series:
        label, reps = labels[s], d["reps"][s]
        mean = reps.mean(axis=0)
        b = mean[sel].mean()
        lines.append(f"  {f'{label} mean of {len(reps)}':28s} {b:.4e} V^2/Hz   "
                     f"{np.sqrt(b)*1e9:8.1f} nV/rtHz")
        per_rep = reps[:, sel].mean(axis=1)
        lines.append(f"    run-to-run: min {np.sqrt(per_rep.min())*1e9:.0f}, "
                     f"max {np.sqrt(per_rep.max())*1e9:.0f} nV/rtHz; "
                     f"PSD spread +/-{100*per_rep.std(ddof=1)/per_rep.mean():.1f}% (1 SD)")
        # Pooled, but report the sittings so drift between them stays visible.
        for sess in d["sessions"][s]:
            sb = sess["psds"][:, sel].mean(axis=1)
            lines.append(f"    session {sess['label']} (n={len(sb)}): "
                         f"mean {np.sqrt(sb.mean())*1e9:.0f} nV/rtHz, runs "
                         + "/".join(f"{np.sqrt(v)*1e9:.0f}" for v in sb))
        if d["ref"] is not None:
            ratio = b / d["ref"][sel].mean()
            lines.append(f"    {label} mean / reference: {ratio:.1f}x in power, "
                         f"{np.sqrt(ratio):.1f}x in amplitude "
                         f"({10*np.log10(ratio):.1f} dB)")

    report = "\n".join(lines) + "\n"
    print(report, end="")
    txt = args.txt or Path(args.out.stem + ".txt")
    txt.write_text(report, encoding="utf-8")
    print(f"results -> {txt}")

    plot(d, tuple(args.band), args.out, labels, ref_label, not args.no_spread)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
