"""prepare_welch.py - turn raw Acquisition.py captures into Welch-ready voltage records.

Acquisition.py stores lossless ADC codes plus the scope preamble. Welch needs
volts and a sample rate. This module does that conversion, and nothing else: no
windowing, no PSD, no averaging.

Use it either way.

1. As an import, which avoids writing a second copy of the data to disk:

       from prepare_welch import load_run
       run = load_run("data/20260925_120719_A1_laser_off.npz")
       f, pxx = scipy.signal.welch(run.v, fs=run.fs, nperseg=4096, axis=-1)
       pxx = pxx.mean(axis=0)          # ensemble average over records

2. As a command line tool, which writes a converted .npz (or CSV):

       python prepare_welch.py data/20260925_120719_A1_laser_off.npz
       python prepare_welch.py --all --drop-clipped
       python prepare_welch.py data/....npz --csv

Output .npz layout (the "format required by Welch.py"):

    v            float64 (n_records, n_samples)  volts
    fs           float64 scalar                  sample rate, Hz
    dt           float64 scalar                  sample interval, s
    t0_unix      float64 (n_records,)            wall clock time of each record
    clipped      bool    (n_records,)            record touched the ADC rail
    meta_json    str                             metadata carried from the capture
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Preamble column indices, per the Keysight X-Series Programmer's Guide.
# [0]=format [1]=type [2]=points [3]=count [4]=xincrement [5]=xorigin
# [6]=xreference [7]=yincrement [8]=yorigin [9]=yreference
XINC, YINC, YORIG, YREF = 4, 7, 8, 9


@dataclass
class Run:
    """One acquisition run, converted to volts."""

    v: np.ndarray           # (n_records, n_samples) float64, volts
    fs: float               # sample rate, Hz
    dt: float               # sample interval, s
    t0_unix: np.ndarray     # (n_records,) float64
    clipped: np.ndarray     # (n_records,) bool
    meta: dict              # metadata from the capture
    path: Path

    @property
    def n_records(self) -> int:
        return self.v.shape[0]

    @property
    def n_samples(self) -> int:
        return self.v.shape[1]

    @property
    def duration_s(self) -> float:
        return self.n_samples * self.dt

    def describe(self) -> str:
        rms = self.v.std(axis=-1) * 1e3
        return (f"{self.path.name}\n"
                f"  tag         : {self.meta.get('tag', '?')}\n"
                f"  note        : {self.meta.get('note', '') or '-'}\n"
                f"  records     : {self.n_records} x {self.n_samples} samples\n"
                f"  sample rate : {self.fs / 1e6:.4f} MSa/s  (dt = {self.dt:.4g} s)\n"
                f"  record      : {self.duration_s * 1e3:.4f} ms  "
                f"-> Welch bin floor {1 / self.duration_s:.1f} Hz\n"
                f"  Nyquist     : {self.fs / 2e6:.4f} MHz\n"
                f"  rms         : {rms.mean():.4f} mV mean "
                f"(min {rms.min():.4f}, max {rms.max():.4f})\n"
                f"  clipped     : {int(self.clipped.sum())} records")


def load_run(path: str | Path, drop_clipped: bool = False) -> Run:
    """Load a capture and convert ADC codes to volts.

    Applies v = (raw - yreference) * yincrement + yorigin per record, using that
    record's own preamble, so a mid-run scale change cannot corrupt the result.
    """
    path = Path(path)
    with np.load(path) as d:
        raw = d["raw"].astype(np.float64)
        pre = d["preamble"].astype(np.float64)
        t0 = d["t0_unix"].copy()
        clipped = d["clipped"].copy()
        meta = json.loads(str(d["meta_json"]))

    v = (raw - pre[:, YREF, None]) * pre[:, YINC, None] + pre[:, YORIG, None]

    dts = pre[:, XINC]
    # atol must be 0: the default 1e-8 is larger than a sample interval
    # (8 ns here), which would silently accept any rate change.
    if not np.allclose(dts, dts[0], rtol=1e-6, atol=0.0):
        raise ValueError(
            f"{path.name}: sample interval changed mid-run "
            f"(min {dts.min():.6g} s, max {dts.max():.6g} s). "
            "Welch needs one sample rate; split this file by rate before continuing."
        )
    dt = float(dts[0])
    if dt <= 0:
        raise ValueError(f"{path.name}: non-positive sample interval {dt}")

    if drop_clipped and clipped.any():
        keep = ~clipped
        if not keep.any():
            raise ValueError(f"{path.name}: every record is clipped, nothing left after --drop-clipped")
        v, t0, clipped = v[keep], t0[keep], clipped[keep]

    return Run(v=v, fs=1.0 / dt, dt=dt, t0_unix=t0, clipped=clipped, meta=meta, path=path)


def write_npz(run: Run, out_path: Path) -> None:
    """Write the Welch-ready arrays."""
    np.savez_compressed(
        out_path,
        v=run.v,
        fs=np.float64(run.fs),
        dt=np.float64(run.dt),
        t0_unix=run.t0_unix,
        clipped=run.clipped,
        meta_json=json.dumps(run.meta),
    )


def write_csv(run: Run, out_path: Path, record: int = 0) -> None:
    """Write one record as time_s, volts - the fallback convention in the README."""
    t = np.arange(run.n_samples) * run.dt
    np.savetxt(out_path, np.column_stack([t, run.v[record]]),
               delimiter=",", header="time_s,volts", comments="", fmt="%.9g")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help="capture .npz files to convert")
    ap.add_argument("--all", action="store_true",
                    help="convert every capture in the data directory")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path,
                    help="where to write (default: alongside the input)")
    ap.add_argument("--suffix", default="_volts", help="added to the output stem")
    ap.add_argument("--drop-clipped", action="store_true",
                    help="exclude records flagged as clipped")
    ap.add_argument("--csv", action="store_true",
                    help="also write record 0 as time_s,volts CSV")
    ap.add_argument("--describe", action="store_true",
                    help="print what each file contains and write nothing")
    args = ap.parse_args()

    files = list(args.files)
    if args.all:
        files += sorted(p for p in args.data_dir.glob("*.npz")
                        if not p.stem.endswith(args.suffix))
    if not files:
        ap.error("give one or more .npz files, or --all")

    failed = 0
    for src in files:
        try:
            run = load_run(src, drop_clipped=args.drop_clipped)
        except (OSError, ValueError, KeyError) as e:
            print(f"SKIP {src.name}: {e}", file=sys.stderr)
            failed += 1
            continue

        print(run.describe())
        if args.describe:
            print()
            continue

        out_dir = args.out_dir or src.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{src.stem}{args.suffix}.npz"
        write_npz(run, dest)
        print(f"  -> {dest}")
        if args.csv:
            csv_path = out_dir / f"{src.stem}{args.suffix}_rec0.csv"
            write_csv(run, csv_path)
            print(f"  -> {csv_path}")
        print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
