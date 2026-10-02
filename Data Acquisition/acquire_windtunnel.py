"""acquire_windtunnel.py - wind-tunnel capture: Acquisition.py at 1000 records, then the PSD.

The capture is Acquisition.py itself, called unchanged - same scope setup, same
file format, same daq_config.yaml - except that the record count defaults to
1000 instead of capture.n_records. Once the file is saved, the Welch PSD is
computed and plotted automatically:

    python acquire_windtunnel.py --tag WT_noflow
    python acquire_windtunnel.py --tag WT_10mps --ref WT_noflow --note "fan 40 %"
    python acquire_windtunnel.py --tag test --dry-run          # nothing saved, no PSD
    python acquire_windtunnel.py --psd data/20261005_101500_WT_10mps.npz --ref WT_noflow

Every Acquisition.py option (--tag, --n, --note, --config, --no-configure,
--dry-run, --list) passes straight through; --n still overrides the 1000.

PSD methodology, as in psd_rpm.py: Welch, Hann window, 50 % overlap, each
record transformed on its own and the records' PSDs averaged in linear V^2/Hz;
nperseg 12500 (10 kHz bins at 125 MSa/s). Plotted as log-log voltage noise in
nV/rtHz. Writes, next to the capture:

    <stem>_psd.npz   f, psd, fs, nperseg, n_records (the per-record PSDs are
                     not kept: at 1000 records they outweigh the raw capture,
                     and can be recomputed from it)
    <stem>_psd.png

--ref TAG overlays the most recent earlier capture with that tag (a no-flow or
laser-off run, say) as a dashed baseline, processed the same way.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

import Acquisition
from prepare_welch import load_run
from psd_rpm import C_MUTED, C_SURFACE, CAT, style_axes
from Welch import welch_psd, welch_stats

N_RECORDS = 1000
NPERSEG = 12500
CHUNK = 100     # records per Welch call; all 1000 at once would need ~1 GB of RAM


def captures(data_dir: Path) -> set[Path]:
    return {p for p in data_dir.glob("*.npz") if not p.stem.endswith(("_volts", "_psd"))}


def psd_of(path: Path, nperseg: int) -> dict:
    """Welch PSD of one capture, computed CHUNK records at a time.

    Each record's PSD is independent of the others, so chunking gives the same
    per-record PSDs, and the same mean, as one call over every record.
    """
    run = load_run(path)
    per = []
    for i in range(0, len(run.v), CHUNK):
        f, _, p = welch_psd(run.v[i:i + CHUNK], run.fs, nperseg=nperseg)
        per.append(p)
    per = np.concatenate(per)
    return {"f": f, "psd": per.mean(axis=0), "per": per, "fs": run.fs,
            "tag": run.meta.get("tag", path.stem[16:]), "n": len(run.v),
            "points": run.v.shape[-1], "clipped": int(run.clipped.sum())}


def latest_with_tag(data_dir: Path, tag: str, exclude: Path) -> Path | None:
    """Most recent capture whose filename tag is exactly `tag`."""
    hits = sorted(p for p in captures(data_dir)
                  if p.stem[16:] == tag and p.resolve() != exclude.resolve())
    return hits[-1] if hits else None


def plot(d: dict, ref: dict | None, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = d["f"]
    m = f > 0
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)

    if ref is not None:
        ax.loglog(f[m], np.sqrt(ref["psd"][m]) * 1e9, color=C_MUTED, lw=1.1,
                  ls=(0, (5, 4)), zorder=2, label=ref["tag"])
    ax.loglog(f[m], np.sqrt(d["psd"][m]) * 1e9, color=CAT[0], lw=1.6, zorder=3,
              label=d["tag"])

    style_axes(ax, "psd", f"{d['tag']}, PSD")
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor="#0b0b0b")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def process(path: Path, ref_tag: str | None, nperseg: int) -> int:
    """PSD, saved arrays and plot for one capture."""
    Acquisition.log(f"PSD: {path.name}, nperseg={nperseg}")
    d = psd_of(path, nperseg)
    st = welch_stats(d["points"], nperseg, 0.5, d["n"])
    Acquisition.log(f"  {d['n']} records x {st['segments_per_record']} segments = "
                    f"{st['total_averages']} averages, df = {d['fs']/nperseg/1e3:g} kHz, "
                    f"per-bin scatter ~{st['per_bin_dB']:.2f} dB")
    if d["clipped"]:
        Acquisition.log(f"  WARNING: {d['clipped']} clipped records are included in the PSD")

    ref = None
    if ref_tag:
        ref_path = latest_with_tag(path.parent, ref_tag, exclude=path)
        if ref_path is None:
            Acquisition.log(f"  note: no capture tagged {ref_tag} in {path.parent}; "
                            "plotting without a reference")
        else:
            ref = psd_of(ref_path, nperseg)
            if not np.array_equal(ref["f"], d["f"]):
                Acquisition.log(f"  note: {ref_path.name} has a different sample rate; "
                                "not overlaid")
                ref = None
            else:
                Acquisition.log(f"  reference: {ref_path.name} ({ref['n']} records)")

    out_npz = path.with_name(path.stem + "_psd.npz")
    np.savez_compressed(out_npz, f=d["f"], psd=d["psd"], fs=d["fs"],
                        nperseg=nperseg, n_records=d["n"])
    out_png = path.with_name(path.stem + "_psd.png")
    plot(d, ref, out_png)
    total = np.sqrt(np.trapezoid(d["psd"], d["f"])) * 1e3
    Acquisition.log(f"  total {total:.2f} mV rms")
    Acquisition.log(f"  saved {out_npz.name}, {out_png.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,     # else --n (for Acquisition.py) would match --nperseg
        epilog="All other options are passed to Acquisition.py "
               "(see python Acquisition.py --help).")
    ap.add_argument("--ref", help="tag of a capture to overlay as a dashed baseline")
    ap.add_argument("--nperseg", type=int, default=NPERSEG,
                    help=f"Welch segment length (default {NPERSEG}: 10 kHz bins at 125 MSa/s)")
    ap.add_argument("--no-psd", action="store_true", help="capture only, skip the PSD")
    ap.add_argument("--psd", type=Path, metavar="NPZ",
                    help="skip the capture; just compute and plot the PSD of this file")
    args, acq_argv = ap.parse_known_args()

    if args.psd:
        return process(args.psd, args.ref, args.nperseg)

    # Acquisition.py's own parser reads sys.argv; give it the pass-through
    # options, with --n defaulting to 1000 rather than the config's n_records.
    if not any(a == "--n" or a.startswith("--n=") for a in acq_argv):
        acq_argv += ["--n", str(N_RECORDS)]
    cfg_ap = argparse.ArgumentParser(add_help=False)    # only to find output_dir
    cfg_ap.add_argument("--config", default="daq_config.yaml")
    config = Path(cfg_ap.parse_known_args(acq_argv)[0].config)
    data_dir = Path(yaml.safe_load(config.read_text())["output_dir"])
    before = captures(data_dir) if data_dir.is_dir() else set()

    sys.argv = [Acquisition.__file__] + acq_argv
    code = Acquisition.main()

    new = sorted(captures(data_dir) - before) if data_dir.is_dir() else []
    if not new:                         # --list, --dry-run, or nothing captured
        return code
    if args.no_psd:
        return code
    if code == 4:
        Acquisition.log("Capture was interrupted; computing the PSD of the records saved")
    try:
        process(new[-1], args.ref, args.nperseg)
    except Exception as e:  # noqa: BLE001 - the capture is saved; don't lose that
        Acquisition.log(f"PSD failed ({type(e).__name__}: {e}). The capture is saved; "
                        f"rerun with: python acquire_windtunnel.py --psd {new[-1]}")
        return code or 5
    return code


if __name__ == "__main__":
    sys.exit(main())
