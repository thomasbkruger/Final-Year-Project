"""psd_rpm.py - C1 rotating-target spectra, one curve per RPM.

Each RPM was captured twice; one of the two captures is plotted (--capture,
1 = the earlier, by default). Every figure is drawn twice, as a Welch PSD and
as a plain FFT amplitude spectrum:

  * an overlay of every RPM's spectrum on one axis, with the stationary
    reference (C1_ref, averaged in linear power over its captures) as a dashed
    baseline;
  * one figure per RPM, all on a shared y-range so the files can be compared
    side by side.

The FFT is one Hann-windowed transform per whole record (df = fs / record
length), averaged across records in power - averaging the complex spectra
would cancel the Doppler return, whose phase is random from record to record.
It is scaled to peak amplitude, so a pure tone reads as its amplitude in mV;
broadband noise reads lower than on the PSD, since each bin holds only one
bin's width of it.

Only captures from one acquisition date are used - by default the most recent
date that has RPM captures.

    python psd_rpm.py
    python psd_rpm.py --date 20261002 --nperseg 25000
    python psd_rpm.py --capture 2              # the second capture of each RPM

RPM is parsed from the capture tag (C1_600RPM): the metadata block on these
runs still carries target "none" / 0 m/s from an earlier config.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from prepare_welch import load_run
from Welch import welch_psd

# dataviz reference palette, light mode. RPM is ordered, but the curves overlap
# heavily and a one-hue ramp made neighbours hard to tell apart, so each RPM
# takes a categorical slot, assigned in RPM order. The peaks themselves still
# march left to right, which carries the order.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
       "#4a3aa7", "#e34948"]                     # categorical slots 1-8
C_INK = "#0b0b0b"
C_MUTED = "#52514e"
C_GRID = "#d8d7d3"
C_SURFACE = "#fcfcfb"

RPM_TAG = re.compile(r"_(\d+)\s*RPM", re.IGNORECASE)

# How each kind of spectrum turns its stored power into the plotted quantity.
KINDS = {
    "psd": {"scale": 1e9, "ylabel": r"voltage noise (nV/$\sqrt{\mathrm{Hz}}$)",
            "name": "PSD"},
    "fft": {"scale": 1e3, "ylabel": "amplitude (mV)", "name": "FFT"},
}


def rpm_from_tag(tag: str) -> int | None:
    m = RPM_TAG.search(tag)
    return int(m.group(1)) if m else None


def fft_power(records: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Peak-amplitude FFT of each whole record, averaged across records in power.

    Returns f (Hz) and the mean |A|^2 (V^2), so sqrt gives amplitude in V.
    """
    records = np.atleast_2d(records)
    n = records.shape[-1]
    w = np.hanning(n)
    x = np.fft.rfft((records - records.mean(axis=-1, keepdims=True)) * w, axis=-1)
    a = 2 * np.abs(x) / w.sum()                 # one-sided, coherent-gain corrected
    a[:, 0] /= 2
    return np.fft.rfftfreq(n, 1 / fs), (a ** 2).mean(axis=0)


def series_colours(n: int) -> list[str]:
    if n > len(CAT):
        raise ValueError(f"{n} RPMs but only {len(CAT)} categorical colours; "
                         "split the figure rather than inventing hues")
    return CAT[:n]


def collect(data_dir: Path, series: str, ref_tag: str, date: str | None,
            nperseg: int) -> dict:
    """PSD and FFT of every RPM capture and reference capture from one date."""
    runs = []
    for p in sorted(data_dir.glob("*.npz")):
        if p.stem.endswith(("_volts", "_psd")):
            continue
        tag = p.stem[16:]                        # cheap pre-filter from the name
        if not tag.startswith(series):
            continue
        runs.append(p)

    rpm_dates = {p.stem[:8] for p in runs if rpm_from_tag(p.stem[16:]) is not None}
    if not rpm_dates:
        return {"date": date, "f": {}, "rpm": {}, "ref": []}
    date = date or max(rpm_dates)

    f_axes, by_rpm, ref = {}, {}, []
    for p in runs:
        if p.stem[:8] != date:
            continue
        run = load_run(p)
        tag = run.meta.get("tag", p.stem[16:])
        rpm = rpm_from_tag(tag)
        if rpm is None and tag != ref_tag:
            continue
        f_psd, psd, _ = welch_psd(run.v, run.fs, nperseg=nperseg)
        f_fft, fft = fft_power(run.v, run.fs)
        for kind, f in (("psd", f_psd), ("fft", f_fft)):
            if kind not in f_axes:
                f_axes[kind] = f
            elif not np.array_equal(f, f_axes[kind]):
                raise ValueError(f"{p.name}: frequency axis differs; cannot average")
        entry = {"psd": psd, "fft": fft,
                 "time": f"{p.stem[9:11]}:{p.stem[11:13]}:{p.stem[13:15]}"}
        if rpm is None:
            ref.append(entry)
        else:
            by_rpm.setdefault(rpm, []).append(entry)
    return {"date": date, "f": f_axes, "rpm": dict(sorted(by_rpm.items())), "ref": ref}


def amp(power: np.ndarray, m: np.ndarray, kind: str) -> np.ndarray:
    """V^2/Hz -> nV/rtHz for the PSD, V^2 -> mV for the FFT."""
    return np.sqrt(power[m]) * KINDS[kind]["scale"]


def style_axes(ax, kind: str, title: str | None = None) -> None:
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ax.set_xlabel("frequency (Hz)", color=C_INK)
    ax.set_ylabel(KINDS[kind]["ylabel"], color=C_INK)
    ax.grid(True, which="major", color=C_GRID, lw=0.7, alpha=0.9)
    ax.grid(True, which="minor", color=C_GRID, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=9, which="both")
    # The y-range spans barely a decade, so label the 2s and 5s as well.
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=C_INK)


def plot_overlay(d: dict, kind: str, out: Path, series: str, ref_tag: str) -> None:
    import matplotlib.pyplot as plt

    f = d["f"][kind]
    m = f > 0
    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)

    if d["ref"]:
        ref = np.mean([e[kind] for e in d["ref"]], axis=0)
        ax.loglog(f[m], amp(ref, m, kind), color=C_MUTED, lw=1.2, ls=(0, (5, 4)),
                  zorder=2, label=f"stationary, {ref_tag} (mean of {len(d['ref'])})")
    for colour, (rpm, e) in zip(series_colours(len(d["rpm"])), d["rpm"].items()):
        ax.loglog(f[m], amp(e[kind], m, kind), color=colour, lw=1.4, zorder=3,
                  label=f"{rpm} RPM")

    style_axes(ax, kind, f"{series} rotating target {KINDS[kind]['name']}")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=C_INK,
                    bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                    title="target speed", alignment="left")
    leg.get_title().set_fontsize(9)
    leg.get_title().set_color(C_MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"plot -> {out}")


def plot_each(d: dict, kind: str, out_dir: Path, series: str, ref_tag: str) -> None:
    import matplotlib.pyplot as plt

    f = d["f"][kind]
    m = f > 0
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = np.mean([e[kind] for e in d["ref"]], axis=0) if d["ref"] else None

    # One y-range for every file, so a taller curve means more signal, not a
    # different autoscale.
    every = [e[kind] for e in d["rpm"].values()]
    if ref is not None:
        every.append(ref)
    lo = min(amp(p, m, kind).min() for p in every)
    hi = max(amp(p, m, kind).max() for p in every)
    ylim = (lo / 1.15, hi * 1.15)

    for colour, (rpm, e) in zip(series_colours(len(d["rpm"])), d["rpm"].items()):
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        fig.patch.set_facecolor(C_SURFACE)
        ax.set_facecolor(C_SURFACE)

        if ref is not None:
            ax.loglog(f[m], amp(ref, m, kind), color=C_MUTED, lw=1.1, ls=(0, (5, 4)),
                      zorder=2, label=f"stationary, {ref_tag}")
        ax.loglog(f[m], amp(e[kind], m, kind), color=colour, lw=1.6, zorder=4,
                  label=f"{rpm} RPM")

        ax.set_ylim(*ylim)
        style_axes(ax, kind, f"{series} - {rpm} RPM, {KINDS[kind]['name']}")
        ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=C_INK)
        fig.tight_layout()
        out = out_dir / f"{kind}_{series}_{rpm}RPM.png"
        fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"plot -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--series", default="C1")
    ap.add_argument("--ref", default="C1_ref", help="tag of the stationary reference")
    ap.add_argument("--date", help="YYYYmmdd; default is the latest date with RPM captures")
    ap.add_argument("--nperseg", type=int, default=12500,
                    help="Welch segment length; 12500 gives 10 kHz bins at 125 MS/s")
    ap.add_argument("--capture", type=int, default=1,
                    help="which capture of each RPM to plot, in time order (1 or 2)")
    ap.add_argument("--out-dir", type=Path, default=Path("."),
                    help="overlays go here, per-RPM figures in psd_rpm/ and fft_rpm/ under it")
    args = ap.parse_args()

    d = collect(args.data_dir, args.series, args.ref, args.date, args.nperseg)
    if not d["rpm"]:
        print(f"no {args.series} RPM captures in {args.data_dir}"
              + (f" on {args.date}" if args.date else ""))
        return 1

    picked = {}
    for rpm, caps in d["rpm"].items():
        if not 1 <= args.capture <= len(caps):
            print(f"{rpm} RPM has {len(caps)} capture(s); cannot pick #{args.capture}")
            return 1
        picked[rpm] = caps[args.capture - 1]
    d["rpm"] = picked

    import matplotlib
    matplotlib.use("Agg")

    f, f_fft = d["f"]["psd"], d["f"]["fft"]
    print(f"{args.series} on {d['date']}, PSD nperseg={args.nperseg} "
          f"(df = {f[1]/1e3:g} kHz), FFT df = {f_fft[1]/1e3:g} kHz")
    ref = np.mean([e["psd"] for e in d["ref"]], axis=0) if d["ref"] else None
    if ref is None:
        print(f"note: no {args.ref} captures on {d['date']}; plotting RPMs alone")
    else:
        print(f"  {args.ref}: {len(d['ref'])} captures")
    sel = f >= 20e3                              # skip the DC end; 50 RPM sits ~60 kHz
    sel_fft = f_fft >= 20e3
    for rpm, e in d["rpm"].items():
        mean = e["psd"]
        line = (f"  {rpm:5d} RPM  capture {e['time']}  "
                f"total {np.sqrt(np.trapezoid(mean, f))*1e3:6.2f} mV rms")
        if ref is not None:
            ratio = mean[sel] / ref[sel]
            k = np.argmax(ratio)
            line += (f"   largest excess over ref: {10*np.log10(ratio[k]):5.1f} dB "
                     f"at {f[sel][k]/1e6:6.2f} MHz")
        fft = e["fft"]
        j = np.argmax(fft[sel_fft])
        line += (f"   FFT peak {np.sqrt(fft[sel_fft][j])*1e3:5.2f} mV "
                 f"at {f_fft[sel_fft][j]/1e6:6.3f} MHz")
        print(line)

    for kind in KINDS:
        plot_overlay(d, kind, args.out_dir / f"{kind}_rpm_overlay.png",
                     args.series, args.ref)
        plot_each(d, kind, args.out_dir / f"{kind}_rpm", args.series, args.ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
