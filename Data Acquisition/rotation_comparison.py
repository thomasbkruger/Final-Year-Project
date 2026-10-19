"""rotation_comparison.py - Doppler speed against setpoint, unrotated vs rotated rig.

The two wind-tunnel sets of wind_tunnel_doppler.py on one axis, captures
picked exactly as there (wind_tunnel_doppler.SETS): tunnel speed from the
Doppler peak, f_D / 1.057 MHz per m/s, against the setpoint, with the
prediction (speed = setpoint) and each set's least-squares line through the
origin, dashed in that set's colour.

    python rotation_comparison.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from peak_vs_rpm import fit_origin
from psd_rpm import C_GRID, C_INK, C_MUTED, C_SURFACE, CAT
from wind_tunnel_doppler import HZ_PER_MPS, SETS, measure_all, select

# Per set: legend name, colour and marker. Slots follow the categorical order
# the other wind-tunnel figures use - CAT[0] the original rig, CAT[1] the
# prediction, CAT[2] mist - so the rotated rig takes the next one. A different
# marker shape keeps the two rigs apart without relying on colour alone.
LOOK = {"unrotated": ("unrotated", CAT[0], "o"),
        "rotated": ("rotated", CAT[3], "s")}


def load(data_dir: Path, name: str, nperseg: int) -> dict:
    """Peaks of one set, its mist runs, and its fit through the origin."""
    cfg = SETS[name]
    pick = {"suffix": cfg["suffix"], "vdiv": cfg["vdiv"], "average": cfg["average"]}
    rows = select(data_dir, cfg["date"], extra=cfg["extra"], **pick)
    mist = select(data_dir, cfg["date"], mist=True, **pick) if cfg["mist"] else []
    f = measure_all(rows, nperseg)
    measure_all(mist, nperseg, f)
    fit = [r for r in rows if not r["approx"]]
    k, se, _ = fit_origin(np.array([r["speed"] for r in fit]),
                          np.array([r["centroid"] for r in fit]))
    return {"name": name, "cfg": cfg, "rows": rows, "mist": mist, "k": k, "se": se}


def style(ax) -> None:
    ax.set_facecolor(C_SURFACE)
    ax.grid(True, color=C_GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED)


def plot(sets: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import figstyle
    figstyle.use()

    fig, ax = plt.subplots(figsize=(figstyle.TEXT_WIDTH, 4.4))
    fig.patch.set_facecolor(C_SURFACE)
    every = [r for s in sets for r in s["rows"] + s["mist"]]
    umax = max(r["speed"] for r in every) * 1.05
    xs = np.array([0, umax])

    ax.plot(xs, xs, color=CAT[1], lw=2, zorder=2,
            label="predicted (1.057 MHz per m/s)")
    for s in sets:
        name, colour, marker = LOOK[s["name"]]
        gain = s["k"] / HZ_PER_MPS                # measured / predicted slope
        ax.plot(xs, gain * xs, color=colour, lw=1.2, ls=(0, (5, 4)), zorder=2,
                label=f"fit, {name}: +{100 * (gain - 1):.1f} %")
        # Mist under the fog markers, as in the single-set plot.
        for group, mk, c, z, label in (
                (s["rows"], marker, colour, 4, f"{name}, fog"),
                (s["mist"], "D", CAT[2], 3, f"{name}, mist")):
            if not group:
                continue
            u = np.array([r["speed"] for r in group])
            meas = np.array([r["centroid"] for r in group]) / HZ_PER_MPS
            ax.plot(u, meas, mk, ls="none", ms=6.5, color=c, mec=C_SURFACE,
                    mew=1.3, zorder=z, label=label)

    style(ax)
    ax.set_xlim(0, umax)
    ax.set_ylim(0, None)
    ax.set_xlabel("wind-tunnel speed setpoint (m/s)", color=C_INK)
    ax.set_ylabel("speed from Doppler peak (m/s)", color=C_INK)
    # The points run up the diagonal, which leaves the top left empty; labels
    # kept short so the legend stays clear of the fit lines.
    ax.legend(frameon=False, loc="upper left", labelcolor=C_INK,
              fontsize=figstyle.FS_NOTE, handlelength=1.8,
              title="wedge and collimator", title_fontsize=figstyle.FS_NOTE,
              alignment="left").get_title().set_color(C_MUTED)
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"plot -> {out}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--nperseg", type=int, default=2500)
    ap.add_argument("--out", type=Path,
                    default=Path("figures/peak_vs_windtunnel_speed_compare.png"))
    args = ap.parse_args()

    sets = [load(args.data_dir, name, args.nperseg) for name in ("unrotated", "rotated")]
    for s in sets:
        errs = [100 * (r["centroid"] / HZ_PER_MPS / r["speed"] - 1) for r in s["rows"]]
        print(f"{s['name']:>9}: fit {s['k']/1e6:.4f} +/- {s['se']/1e6:.4f} MHz per m/s "
              f"({100 * (s['k']/HZ_PER_MPS - 1):+.1f} %), per-point "
              f"{min(errs):+.1f} to {max(errs):+.1f} %")
    plot(sets, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
