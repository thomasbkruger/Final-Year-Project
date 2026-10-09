"""psd_rpm_compare.py - one RPM, old rotating-target set against the new one.

Overlays the PSD of the same target speed from two acquisition sets - by
default the 2 October C1 set and the 9 October 4 m fibre set - each with its
own stationary reference dashed in the same colour, so a change in the Doppler
peak can be read against the change in the background under it. Drawn as in
psd_rpm.py: Welch PSD in V^2/Hz on linear axes, ranged to the Doppler content.

    python psd_rpm_compare.py                  # every RPM both sets have
    python psd_rpm_compare.py --rpm 2000
    python psd_rpm_compare.py --old-capture 2

One figure per RPM, figures/psd_rpm_old_vs_new_<rpm>RPM.png. Only RPMs present
in both sets can be compared; any others asked for are reported and skipped.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from psd_rpm import C_INK, C_SURFACE, CAT, amp, collect, style_axes

# Each set keeps one hue for its capture and its reference: slot 2 (orange)
# for the old set, slot 1 (blue) for the new.
C_OLD, C_NEW = CAT[1], CAT[0]


def load_set(data_dir: Path, series: str, ref: str, date: str, capture: int,
             nperseg: int) -> dict:
    d = collect(data_dir, series, ref, date, nperseg)
    picked = {rpm: caps[capture - 1] for rpm, caps in d["rpm"].items()
              if len(caps) >= capture}
    ref_psd = np.mean([e["psd"] for e in d["ref"]], axis=0) if d["ref"] else None
    return {"f": d["f"].get("psd"), "rpm": picked, "ref": ref_psd, "n_ref": len(d["ref"])}


def plot(rpm: int, old: dict, new: dict, names: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = new["f"]
    m = f > 0
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)

    for s, colour, name in ((old, C_OLD, names["old"]), (new, C_NEW, names["new"])):
        if s["ref"] is not None:
            ax.plot(f[m], amp(s["ref"], m, "psd"), color=colour, lw=1.1,
                    ls=(0, (5, 4)), alpha=0.8, zorder=2,
                    label=f"{name}, stationary ({names[id(s)]})")
        ax.plot(f[m], amp(s["rpm"][rpm]["psd"], m, "psd"), color=colour, lw=1.6,
                zorder=3, label=f"{name}, {rpm} RPM")

    # Range to the Doppler content: 1.5x the higher of the two peaks.
    sel = f >= 20e3
    peak = max(f[sel][np.argmax(s["rpm"][rpm]["psd"][sel])] for s in (old, new))
    ax.set_xlim(0, max(1.5 * peak, 1e6))
    ax.set_ylim(0, None)
    style_axes(ax, "psd", f"{rpm} RPM, old vs new")
    # The peak can sit anywhere across the range, so the legend goes outside.
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=C_INK,
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"plot -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--rpm", type=int, nargs="*",
                    help="RPMs to compare; default is every RPM both sets have")
    ap.add_argument("--old-series", default="C1")
    ap.add_argument("--old-ref", default="C1_ref")
    ap.add_argument("--old-date", default="20261002")
    ap.add_argument("--old-name", default="2 Oct")
    ap.add_argument("--old-capture", type=int, default=1)
    ap.add_argument("--new-series", default="")
    ap.add_argument("--new-ref", default="PIIN_4m")
    ap.add_argument("--new-date", default="20261009")
    ap.add_argument("--new-name", default="9 Oct, 4 m fibre")
    ap.add_argument("--nperseg", type=int, default=12500)
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()

    old = load_set(args.data_dir, args.old_series, args.old_ref, args.old_date,
                   args.old_capture, args.nperseg)
    new = load_set(args.data_dir, args.new_series, args.new_ref, args.new_date,
                   1, args.nperseg)
    if old["f"] is None or new["f"] is None:
        print("one of the sets has no captures; check --*-series/--*-date")
        return 1
    if not np.array_equal(old["f"], new["f"]):
        print("the two sets have different frequency axes; cannot overlay")
        return 1

    shared = sorted(set(old["rpm"]) & set(new["rpm"]))
    wanted = args.rpm or shared
    missing = [r for r in wanted if r not in shared]
    for r in missing:
        print(f"{r} RPM: not in both sets "
              f"(old has {sorted(old['rpm'])}, new has {sorted(new['rpm'])}); skipped")
    todo = [r for r in wanted if r in shared]
    if not todo:
        return 1

    names = {"old": args.old_name, "new": args.new_name,
             id(old): f"{args.old_ref}, mean of {old['n_ref']}",
             id(new): f"{args.new_ref}, mean of {new['n_ref']}"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for rpm in todo:
        plot(rpm, old, new, names, args.out_dir / f"psd_rpm_old_vs_new_{rpm}RPM.png")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
