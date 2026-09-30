"""Acquisition.py - capture raw waveform records from a Keysight EDUX1002G over USB.

Usage:
    python Acquisition.py --tag A1_laser_off
    python Acquisition.py --tag A1_leg_A_10mA --n 50 --note "coupler leg A on combiner"
    python Acquisition.py --tag test --dry-run

Each run writes one .npz (raw int16 samples + preamble + metadata) and a
matching .json sidecar you can open in any text editor. No processing is done
here. To convert to volts later:

    v = (raw - yreference) * yincrement + yorigin

where yincrement/yorigin/yreference are preamble columns 7, 8, 9.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def _enable_keysight_dlls() -> None:
    """Put the Keysight IO Libraries folders on the DLL search path (Windows only).

    Keysight's VISA implementation (ktvisa32.dll) lives in System32 but its
    dependencies live under "IO Libraries Suite". Python 3.8+ no longer searches
    PATH for dependent DLLs, so without this pyvisa fails with
    VI_ERROR_LIBRARY_NFOUND even though Connection Expert works fine.
    """
    if os.name != "nt":
        return
    root = Path(r"C:\Program Files\Keysight\IO Libraries Suite")
    for sub in ("bin", "lib_x64", ""):
        d = root / sub if sub else root
        if d.is_dir():
            try:
                os.add_dll_directory(str(d))
            except OSError:
                pass


_enable_keysight_dlls()

import pyvisa  # noqa: E402  (must follow the DLL search-path fix above)
import yaml  # noqa: E402

EXPECTED_MODEL = ("EDU-X 1002G", "EDUX1002G")

# ---------------------------------------------------------------------------
# InfiniiVision command set - see Keysight X-Series Programmer's Guide
# ---------------------------------------------------------------------------
SCOPE_SETUP = [  # acquisition settings; skipped with --no-configure
    ":CHANnel1:COUPling {coupling}",
    ":CHANnel1:SCALe {volts_per_div}",
    ":CHANnel1:OFFSet {offset_v}",
    ":CHANnel2:DISPlay OFF",            # frees full memory for channel 1
    ":TIMebase:SCALe {seconds_per_div}",
    ":TIMebase:REFerence CENTer",
    ":ACQuire:TYPE {acquire_type}",     # HRESolution; never AVERage
    ":ACQuire:MODE RTIMe",
    ":TRIGger:SWEep AUTO",              # free-running; noise has nothing to trigger on
]
TRANSFER_SETUP = [  # how the waveform is sent to the PC; always sent
    ":WAVeform:SOURce CHANnel1",
    ":WAVeform:FORMat WORD",            # 16-bit; BYTE would throw away High-Res bits
    ":WAVeform:BYTeorder LSBFirst",
    ":WAVeform:UNSigned 0",
    ":WAVeform:POINts:MODE RAW",
    ":WAVeform:POINts {points}",
]
READBACK = {
    "sample_rate": ":ACQuire:SRATe?",
    "acq_points": ":ACQuire:POINts?",
    "wav_points": ":WAVeform:POINts?",
    "volts_per_div": ":CHANnel1:SCALe?",
    "acquire_type": ":ACQuire:TYPE?",
    "input_impedance": ":CHANnel1:IMPedance?",   # ONEM on this model; it has no 50 ohm input,
                                                 # so any termination is an external feed-through
}
CMD_DIGITIZE = ":DIGitize CHANnel1"
CMD_PREAMBLE = ":WAVeform:PREamble?"
CMD_DATA = ":WAVeform:DATA?"
CMD_RUN = ":RUN"


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S}  {msg}", flush=True)


def list_instruments(backend: str = "") -> None:
    """Print every VISA address currently visible, with its *IDN? response."""
    rm = pyvisa.ResourceManager(backend or "")
    found = rm.list_resources()
    if not found:
        print("No instruments found. Check the scope is on and plugged into the REAR USB-B port.")
        return
    print("Paste the address below into visa_address in daq_config.yaml:\n")
    for addr in found:
        try:
            inst = rm.open_resource(addr)
            inst.timeout = 5000
            idn = inst.query("*IDN?").strip()
            inst.close()
        except Exception as e:  # noqa: BLE001
            idn = f"<no response: {type(e).__name__}>"
        print(f"  {addr}\n      {idn}")


def connect(cfg: dict) -> pyvisa.resources.MessageBasedResource:
    """Open the scope, set transfer limits, and check the identity string."""
    try:
        rm = pyvisa.ResourceManager(cfg.get("visa_backend") or "")
    except pyvisa.errors.VisaIOError as e:
        raise SystemExit(
            f"Could not load a VISA library ({e}).\n"
            "Install Keysight IO Libraries Suite, or set visa_backend: \"@py\" in the config "
            "after 'pip install pyvisa-py pyusb'."
        ) from e
    scope = rm.open_resource(cfg["visa_address"])
    scope.timeout = 20_000          # ms; a 100 kpt WORD transfer takes a few seconds
    scope.chunk_size = 1 << 20
    idn = scope.query("*IDN?").strip()
    log(f"Connected: {idn}")
    if not any(m in idn for m in EXPECTED_MODEL):
        log("WARNING: this does not look like an EDUX1002G; continuing anyway")
    return scope


def configure(scope, sc: dict, full: bool) -> dict:
    """Send the setup commands (transfer settings always; scope settings if full) and read back."""
    cmds = (SCOPE_SETUP if full else []) + TRANSFER_SETUP
    for c in cmds:
        scope.write(c.format(**sc))
    rb = {k: scope.query(q).strip() for k, q in READBACK.items()}
    log("Scope reports: " + ", ".join(f"{k}={v}" for k, v in rb.items()))

    exp_sr = float(sc["expected_sample_rate"])
    got_sr = float(rb["sample_rate"])
    if abs(got_sr - exp_sr) > 0.01 * exp_sr:
        log(f"WARNING: sample rate {got_sr:g} Sa/s, expected {exp_sr:g} "
            "(check channel 2 is off / seconds_per_div / expected_sample_rate in config)")
    # Note: :WAVeform:POINts? read here reports the shallow display record, not what
    # :DIGitize will deliver. The real length is checked against the first record below.
    return rb


def capture_one(scope) -> tuple[float, list[float], np.ndarray]:
    """One single-shot acquisition. Returns (unix time, preamble[10], raw int16 array)."""
    t0 = time.time()
    scope.write(CMD_DIGITIZE)                       # blocks the scope until done
    pre = [float(x) for x in scope.query(CMD_PREAMBLE).split(",")]
    if len(pre) != 10:
        raise RuntimeError(f"{CMD_PREAMBLE} returned {len(pre)} fields: {pre}")
    raw = scope.query_binary_values(CMD_DATA, datatype="h", is_big_endian=False, container=np.array)
    return t0, pre, raw.astype(np.int16, copy=False)


def to_volts(raw: np.ndarray, pre: list[float]) -> np.ndarray:
    return (raw.astype(np.float64) - pre[9]) * pre[7] + pre[8]


def stats(raw: np.ndarray, pre: list[float], volts_per_div: float, offset_v: float) -> dict:
    v = to_volts(raw, pre)
    peak = max(abs(v.max() - offset_v), abs(v.min() - offset_v))
    return {
        "rms_mV": float(np.sqrt(np.mean(v * v)) * 1e3),
        "mean_mV": float(v.mean() * 1e3),
        "min_mV": float(v.min() * 1e3),
        "max_mV": float(v.max() * 1e3),
        "clipped": bool(peak >= 0.99 * 4 * volts_per_div),   # 8 divisions full scale
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list visible VISA addresses and exit")
    ap.add_argument("--tag", help="short label used in the filename, e.g. A1_laser_off")
    ap.add_argument("--n", type=int, help="number of records (default: capture.n_records in config)")
    ap.add_argument("--note", default="", help="free text stored with the run")
    ap.add_argument("--config", default="daq_config.yaml")
    ap.add_argument("--no-configure", action="store_true",
                    help="leave the scope as set on the front panel (only transfer format is set)")
    ap.add_argument("--dry-run", action="store_true", help="capture one record, print stats, save nothing")
    args = ap.parse_args()

    if args.list:
        list_instruments()
        return 0
    if not args.tag:
        ap.error("--tag is required (or use --list to find the scope's address)")

    cfg = yaml.safe_load(Path(args.config).read_text())
    sc, cap = cfg["scope"], cfg["capture"]
    n_target = 1 if args.dry_run else (args.n or int(cap["n_records"]))
    vdiv, voff = float(sc["volts_per_div"]), float(sc["offset_v"])

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d_%H%M%S}_{args.tag}"

    scope = connect(cfg)
    exit_code = 0
    t0s: list[float] = []
    pres: list[list[float]] = []
    raws: list[np.ndarray] = []
    clipped: list[bool] = []
    t_start = time.time()
    try:
        readback = configure(scope, sc, full=not args.no_configure)
        if args.no_configure:
            vdiv = float(readback["volts_per_div"])
        log(f"Capturing {n_target} records -> {out_dir / stem}.npz   (Ctrl-C to stop early and save)")

        try:
            for i in range(n_target):
                t0, pre, raw = capture_one(scope)
                if raws and len(raw) != len(raws[0]):
                    raise RuntimeError(f"record {i}: {len(raw)} samples, previous records had {len(raws[0])}")
                if not raws and len(raw) != int(sc["points"]):
                    log(f"WARNING: scope returned {len(raw)} points, config asks for {sc['points']} "
                        "(this model caps at 25000 in HRESolution, 50000 in NORMal)")
                s = stats(raw, pre, vdiv, voff)
                t0s.append(t0); pres.append(pre); raws.append(raw); clipped.append(s["clipped"])
                if args.dry_run:
                    log(f"points={len(raw)}  dt={pre[4]:.4g} s  " +
                        "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
                elif (i + 1) % int(cap["progress_every"]) == 0 or i + 1 == n_target:
                    log(f"[{i + 1:4d}/{n_target}]  {time.time() - t_start:6.1f} s  "
                        f"rms {s['rms_mV']:.3f} mV  clipped {sum(clipped)}")
                if float(cap["inter_record_delay_s"]) > 0:
                    time.sleep(float(cap["inter_record_delay_s"]))
        except KeyboardInterrupt:
            log(f"Interrupted by user after {len(raws)} records")
            exit_code = 4
    finally:
        try:
            scope.write(CMD_RUN)
            scope.close()
        except Exception as e:  # noqa: BLE001
            log(f"WARNING: could not return scope to RUN: {e}")

    if args.dry_run:
        log("Dry run: nothing saved")
        return exit_code
    if not raws:
        log("No records captured; nothing saved")
        return exit_code or 1

    meta = {
        "tag": args.tag,
        "note": args.note,
        "start_time": datetime.fromtimestamp(t_start).isoformat(timespec="seconds"),
        "stop_time": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "configured_scope": not args.no_configure,
        "scope_readback": readback,
        "config": cfg,
        "volts_formula": "v = (raw - yreference) * yincrement + yorigin  "
                         "(preamble columns: 4=xincrement, 7=yincrement, 8=yorigin, 9=yreference)",
    }
    rms_all = [stats(r, p, vdiv, voff)["rms_mV"] for r, p in zip(raws, pres)]
    summary = {
        "n_records": len(raws),
        "points_per_record": len(raws[0]),
        "sample_interval_s": pres[0][4],
        "clipped_records": int(sum(clipped)),
        "rms_mV_first": rms_all[0],
        "rms_mV_last": rms_all[-1],
        "rms_mV_mean": float(np.mean(rms_all)),
        "interrupted": exit_code == 4,
    }

    npz_path = out_dir / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        raw=np.stack(raws),
        t0_unix=np.array(t0s, dtype=np.float64),
        preamble=np.array(pres, dtype=np.float64),
        clipped=np.array(clipped, dtype=bool),
        meta_json=json.dumps(meta),
    )
    (out_dir / f"{stem}.json").write_text(json.dumps({"summary": summary, **meta}, indent=2))
    log(f"Saved {summary['n_records']} records to {npz_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
