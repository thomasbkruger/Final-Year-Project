"""run_welch.py - run the Welch.py analysis on a captured run, from the command line.

Welch.py is a library plus a synthetic self-test; on its own it cannot read a
capture. This joins the three pieces: prepare_welch loads the raw file and
converts it to volts, Welch.py computes the PSD, this prints and plots it.

    python run_welch.py data.npz
    python run_welch.py --all
    python run_welch.py data/....npz --plot --save
    python run_welch.py data/....npz --nperseg 5000 --band 1e6 10e6
    python run_welch.py data/....npz --gain 39e3          # override the TIA gain

The transimpedance gain defaults to the value implied by the capture metadata:
19.5 kV/A when terminator_50ohm is true, 39 kV/A otherwise. Getting this wrong
scales the input-referred current noise by two, so it is reported explicitly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from prepare_welch import load_run
from psd_vs_current import resolve_tag
from Welch import (Q, R_PD, band_average, to_input_current,
                   welch_psd, welch_stats)

GAIN_50OHM = 19.5e3     # PD100B transimpedance into 50 ohm (V/A)
GAIN_HIGHZ = 39.0e3     # PD100B transimpedance into high impedance (V/A)


def gain_from_meta(meta: dict) -> tuple[float, str]:
    """Pick the TIA gain implied by the capture metadata."""
    try:
        terminated = bool(meta["config"]["metadata"]["terminator_50ohm"])
    except (KeyError, TypeError):
        return GAIN_50OHM, "50 ohm assumed (metadata missing)"
    if terminated:
        return GAIN_50OHM, "50 ohm terminator per metadata"
    return GAIN_HIGHZ, "high-Z per metadata (terminator_50ohm false)"


def analyse(path: Path, args: argparse.Namespace) -> dict:
    """Load one capture, compute its PSD, print the numbers, return the results."""
    run = load_run(path, drop_clipped=args.drop_clipped)
    gain, gain_why = (args.gain, "set on the command line") if args.gain else gain_from_meta(run.meta)

    f, psd, per = welch_psd(run.v, run.fs, nperseg=args.nperseg,
                            overlap=args.overlap, window=args.window,
                            hres_droop_correct=args.droop)

    f_lo, f_hi = args.band
    if f_hi > run.fs / 2:
        print(f"  note: band top {f_hi/1e6:.3g} MHz exceeds Nyquist "
              f"{run.fs/2e6:.3g} MHz; clipping to Nyquist")
        f_hi = run.fs / 2
    band, sem = band_average(f, per, f_lo, f_hi)
    st = welch_stats(run.n_samples, args.nperseg, args.overlap, run.n_records)

    i_psd = to_input_current(band, gain)
    i_rt = np.sqrt(i_psd)
    p_lo_equiv = i_psd / (2 * Q * R_PD)

    print(f"\n{path.name}")
    print(f"  tag / note      : {run.meta.get('tag', '?')} | {run.meta.get('note', '') or '-'}")
    print(f"  records         : {run.n_records} x {run.n_samples} pts, "
          f"{run.fs/1e6:.4g} MSa/s, {run.duration_s*1e3:.4g} ms each")
    print(f"  resolution      : {f[1]-f[0]:.4g} Hz over {len(f)} bins "
          f"(0 to {f[-1]/1e6:.4g} MHz)")
    print(f"  averages        : {st['total_averages']} "
          f"({st['segments_per_record']}/record) -> per-bin +/-{st['per_bin_dB']:.3f} dB")
    if st["segments_per_record"] < 4:
        print(f"  WARNING: only {st['segments_per_record']} segments per record. "
              "Lower --nperseg or capture a longer record.")
    print(f"  gain            : {gain/1e3:.4g} kV/A ({gain_why})")
    print(f"  band {f_lo/1e6:.4g}-{f_hi/1e6:.4g} MHz")
    print(f"    PSD           : {band:.4e} +/- {sem:.2e} V^2/Hz")
    print(f"    voltage noise : {np.sqrt(band)*1e9:.4g} nV/rtHz")
    print(f"    input current : {i_rt*1e12:.4g} pA/rtHz")
    print(f"    shot-limited at LO power: {p_lo_equiv*1e6:.4g} uW total on both diodes")

    if args.save:
        out = path.with_name(path.stem + "_psd.npz")
        np.savez_compressed(out, f=f, psd=psd, psd_per_record=per, fs=run.fs,
                            gain=gain, band=np.array([f_lo, f_hi]),
                            band_psd=band, band_sem=sem)
        print(f"  -> {out}")

    return {"f": f, "psd": psd,
            "label": resolve_tag(run.meta.get("tag", path.stem)) or path.stem,
            "band": (f_lo, f_hi), "gain": gain}


def plot(results: list[dict], out_path: Path | None, gain: float) -> None:
    """Plot the noise floor in nV/rtHz and pA/rtHz against frequency."""
    import matplotlib
    if out_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r in results:
        f, psd = r["f"], r["psd"]
        m = f > 0
        ax.loglog(f[m], psd[m], lw=1.0, label=r["label"])
    lo, hi = results[0]["band"]
    ax.axvspan(lo, hi, color="0.85", zorder=0, label=f"band {lo/1e6:g}-{hi/1e6:g} MHz")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(r"PSD (V$^2$/Hz)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # Same quantity referred to the photodiode input: divide by the TIA gain
    # squared, in pA^2/Hz so the numbers stay readable.
    sec = ax.secondary_yaxis("right", functions=(lambda v: v / gain**2 * 1e24,
                                                 lambda i: i * gain**2 / 1e24))
    sec.set_ylabel(r"input current PSD (pA$^2$/Hz)")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"\nplot -> {out_path}")
    else:
        plt.show()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help="capture .npz files")
    ap.add_argument("--all", action="store_true", help="every capture in --data-dir")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--nperseg", type=int, default=2500,
                    help="Welch segment length; resolution = fs/nperseg (default 2500)")
    ap.add_argument("--overlap", type=float, default=0.5, help="segment overlap (default 0.5)")
    ap.add_argument("--window", default="hann")
    ap.add_argument("--band", nargs=2, type=float, default=[5e6, 20e6],
                    metavar=("F_LO", "F_HI"), help="band to average (default 5e6 20e6)")
    ap.add_argument("--gain", type=float, help="TIA gain V/A; default from capture metadata")
    ap.add_argument("--droop", action="store_true",
                    help="apply the HRESolution sinc^2 droop correction")
    ap.add_argument("--drop-clipped", action="store_true")
    ap.add_argument("--plot", action="store_true", help="plot the floor")
    ap.add_argument("--plot-file", type=Path, help="write the plot here instead of showing it")
    ap.add_argument("--save", action="store_true", help="write <name>_psd.npz per file")
    args = ap.parse_args()

    files = list(args.files)
    if args.all:
        files += sorted(p for p in args.data_dir.glob("*.npz")
                        if not p.stem.endswith(("_volts", "_psd")))
    if not files:
        ap.error("give one or more capture .npz files, or --all")
    if args.nperseg < 8:
        ap.error("--nperseg must be at least 8")
    if not 0.0 <= args.overlap < 1.0:
        ap.error("--overlap must be in [0, 1)")

    results, failed = [], 0
    for src in files:
        try:
            if resolve_tag(load_run(src).meta.get("tag", src.stem)) is None:
                print(f"skip {src.name}: superseded (see psd_vs_current.EXCLUDE_TAGS)")
                continue
            results.append(analyse(src, args))
        except (OSError, ValueError, KeyError) as e:
            print(f"SKIP {src.name}: {e}", file=sys.stderr)
            failed += 1

    if results and (args.plot or args.plot_file):
        plot(results, args.plot_file, results[0]["gain"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
