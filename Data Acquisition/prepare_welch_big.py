"""prepare_welch_big.py - prepare_welch.py for captures too large to hold in memory.

prepare_welch.load_run converts the whole capture to volts in one go:

    raw = d["raw"].astype(np.float64)

At 100 records that is 20 MB and nobody notices. At the 10 000 records
Acquisition_fast.py can now take, it is 10000 x 25000 x 8 = 2 GB in a single
allocation, on top of the 500 MB of int16 it was converted from. This module
does the same conversion a block of records at a time, so the float64 copy is
only ever one block wide.

The records are also streamed out of the .npz rather than read whole. A .npz is
a zip, and zipfile decompresses a member incrementally, so iter_volts never
holds more than one block of raw samples either - compressed or not. Peak
memory for a 10 000-record capture goes from roughly 2.5 GB to about 40 MB.

    from prepare_welch_big import open_run, iter_volts
    info = open_run("data/20261006_120000_WT_10mps.npz")
    for i0, v in iter_volts(info, chunk=100):
        ...                                  # v is (<=100, n_samples) float64

load_run is kept, with prepare_welch's exact signature and behaviour, so this
module is a drop-in superset for small files. It carries the same memory cost
as the original by definition - use open_run/iter_volts for the large ones.

The command line is unchanged from prepare_welch.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.lib import format as npy_format

# Preamble column indices, per the Keysight X-Series Programmer's Guide.
# [0]=format [1]=type [2]=points [3]=count [4]=xincrement [5]=xorigin
# [6]=xreference [7]=yincrement [8]=yorigin [9]=yreference
XINC, YINC, YORIG, YREF = 4, 7, 8, 9

DEFAULT_CHUNK = 100


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


@dataclass
class RunInfo:
    """Everything about a capture except the samples themselves."""

    path: Path
    fs: float
    dt: float
    t0_unix: np.ndarray
    clipped: np.ndarray
    meta: dict
    preamble: np.ndarray    # (n_records, 10)
    n_records: int
    n_samples: int
    keep: np.ndarray        # (n_records,) bool, records iter_volts will yield

    @property
    def n_kept(self) -> int:
        return int(self.keep.sum())

    @property
    def duration_s(self) -> float:
        return self.n_samples * self.dt

    def describe(self) -> str:
        return (f"{self.path.name}\n"
                f"  tag         : {self.meta.get('tag', '?')}\n"
                f"  note        : {self.meta.get('note', '') or '-'}\n"
                f"  records     : {self.n_kept} x {self.n_samples} samples"
                + ("" if self.n_kept == self.n_records
                   else f" ({self.n_records - self.n_kept} dropped)") + "\n"
                f"  sample rate : {self.fs / 1e6:.4f} MSa/s  (dt = {self.dt:.4g} s)\n"
                f"  record      : {self.duration_s * 1e3:.4f} ms  "
                f"-> Welch bin floor {1 / self.duration_s:.1f} Hz\n"
                f"  Nyquist     : {self.fs / 2e6:.4f} MHz\n"
                f"  raw size    : {self.n_records * self.n_samples * 2 / 1e6:.0f} MB "
                f"as int16\n"
                f"  clipped     : {int(self.clipped.sum())} records")


def _check_dt(path: Path, pre: np.ndarray) -> float:
    """The one sample interval the run was taken at, or raise."""
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
    return dt


def open_run(path: str | Path, drop_clipped: bool = False) -> RunInfo:
    """Read a capture's metadata without reading a single sample."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as d:
        pre = d["preamble"].astype(np.float64)
        t0 = d["t0_unix"].copy()
        clipped = d["clipped"].copy()
        meta = json.loads(str(d["meta_json"]))
        # .shape on an NpzFile member still reads it; take the length from the
        # header instead, which is a few hundred bytes.
        n_records, n_samples = _raw_shape(path)

    dt = _check_dt(path, pre)
    keep = ~clipped if drop_clipped else np.ones(n_records, dtype=bool)
    if drop_clipped and not keep.any():
        raise ValueError(f"{path.name}: every record is clipped, "
                         "nothing left after --drop-clipped")
    return RunInfo(path=path, fs=1.0 / dt, dt=dt, t0_unix=t0, clipped=clipped,
                   meta=meta, preamble=pre, n_records=n_records,
                   n_samples=n_samples, keep=keep)


def _raw_member(zf: zipfile.ZipFile) -> str:
    for name in ("raw.npy", "raw"):
        if name in zf.namelist():
            return name
    raise KeyError(f"no 'raw' array in {zf.filename}")


def _raw_shape(path: Path) -> tuple[int, int]:
    """Shape of the raw array, read from its .npy header alone."""
    with zipfile.ZipFile(path) as zf, zf.open(_raw_member(zf)) as fh:
        shape, fortran, _ = npy_format._read_array_header(
            fh, npy_format.read_magic(fh))
    if fortran:
        raise ValueError(f"{path.name}: raw array is Fortran-ordered; "
                         "stream reading assumes C order")
    if len(shape) != 2:
        raise ValueError(f"{path.name}: raw array has shape {shape}, expected 2-D")
    return int(shape[0]), int(shape[1])


def iter_volts(info: RunInfo, chunk: int = DEFAULT_CHUNK
               ) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (first record index, volts block) a chunk of records at a time.

    Each block is float64 volts, converted with that record's own preamble, so
    a mid-run scale change cannot corrupt the result - the same guarantee
    prepare_welch.load_run gives. Records excluded by drop_clipped are skipped,
    and the index yielded is the index in the original file.
    """
    if chunk < 1:
        raise ValueError("chunk must be at least 1")
    yinc = info.preamble[:, YINC]
    yorig = info.preamble[:, YORIG]
    yref = info.preamble[:, YREF]
    row_bytes = info.n_samples * 2            # int16

    with zipfile.ZipFile(info.path) as zf, zf.open(_raw_member(zf)) as fh:
        npy_format._read_array_header(fh, npy_format.read_magic(fh))
        done = 0
        while done < info.n_records:
            m = min(chunk, info.n_records - done)
            buf = fh.read(row_bytes * m)
            if len(buf) != row_bytes * m:
                raise ValueError(f"{info.path.name}: raw data ended early at "
                                 f"record {done} ({len(buf)} of "
                                 f"{row_bytes * m} bytes)")
            block = np.frombuffer(buf, dtype="<i2").reshape(m, info.n_samples)
            sl = slice(done, done + m)
            take = info.keep[sl]
            if take.any():
                v = block[take].astype(np.float64)
                v -= yref[sl][take, None]
                v *= yinc[sl][take, None]
                v += yorig[sl][take, None]
                yield done, v
            done += m


def load_run(path: str | Path, drop_clipped: bool = False) -> Run:
    """Load a capture and convert ADC codes to volts, as prepare_welch does.

    Present so this module is a drop-in superset. It builds the whole float64
    array, so it has the original's memory cost: use open_run and iter_volts
    for captures of more than a few hundred records.
    """
    info = open_run(path, drop_clipped=drop_clipped)
    v = np.empty((info.n_kept, info.n_samples), dtype=np.float64)
    at = 0
    for _, block in iter_volts(info, chunk=DEFAULT_CHUNK):
        v[at:at + len(block)] = block
        at += len(block)
    keep = info.keep
    return Run(v=v, fs=info.fs, dt=info.dt, t0_unix=info.t0_unix[keep],
               clipped=info.clipped[keep], meta=info.meta, path=info.path)


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
                    help="print what each file contains and write nothing "
                         "(reads no samples, so it is instant on a large capture)")
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
            if args.describe:                 # no samples read at all
                print(open_run(src, drop_clipped=args.drop_clipped).describe())
                print()
                continue
            run = load_run(src, drop_clipped=args.drop_clipped)
        except (OSError, ValueError, KeyError) as e:
            print(f"SKIP {src.name}: {e}", file=sys.stderr)
            failed += 1
            continue

        print(run.describe())
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
