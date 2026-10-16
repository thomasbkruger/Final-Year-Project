"""psd_rpm.py - C1 rotating-target spectra, one curve per RPM.

Each RPM was captured twice; one of the two captures is plotted (--capture,
1 = the earlier, by default). Every figure is drawn twice, as a Welch PSD and
as a plain FFT amplitude spectrum:

  * an overlay of every RPM's spectrum on one axis, with the stationary
    reference (C1_ref, averaged in linear power over its captures) as a dashed
    baseline;
  * one figure per RPM, all on a shared y-range so the files can be compared
    side by side.

The PSD figures are drawn in V^2/Hz on linear axes, ranged to the Doppler
content (--fmax) because a linear frequency axis out to Nyquist would put every
peak in the leftmost few percent. The shared y-range is kept, so a 2000 RPM
peak really does read as a seventh of the 100 RPM one. The FFT figures stay on
log axes in mV, where the peaks span decades.

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
    python psd_rpm.py --series "" --ref PIIN_4m --date 20261009   # 4 m fibre set

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

# Matches C1_600RPM and 500rpm_4m alike: the number opens the tag or follows
# an underscore.
RPM_TAG = re.compile(r"(?:^|_)(\d+)\s*RPM", re.IGNORECASE)

# How each kind of spectrum turns its stored power into the plotted quantity,
# and the axis scales it is drawn on. The PSD is plotted as stored, in V^2/Hz
# on linear axes; the FFT keeps amplitude in mV on log axes, where its peaks
# span decades and a density unit would be wrong.
KINDS = {
    "psd": {"root": False, "scale": 1.0, "ylabel": r"PSD (V$^2$/Hz)",
            "name": "PSD", "xscale": "linear", "yscale": "linear"},
    "fft": {"root": True, "scale": 1e3, "ylabel": "amplitude (mV)",
            "name": "FFT", "xscale": "log", "yscale": "log"},
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
        # The reference matches by prefix, so numbered repeats (PIIN_4m_1,
        # PIIN_4m_2, ...) all join the stationary baseline.
        if rpm is None and not tag.startswith(ref_tag):
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
    """Stored power -> plotted quantity: V^2/Hz as stored for the PSD,
    V^2 -> mV amplitude for the FFT."""
    spec = KINDS[kind]
    p = power[m]
    return (np.sqrt(p) if spec["root"] else p) * spec["scale"]


def shown_as(name: str, series: str, label: str) -> str:
    """Tag or title text with the series prefix under its display name.

    The captures are tagged C1, but the figures call the series C; selection
    still matches the real prefix, so only what is drawn changes.
    """
    return name.replace(series, label, 1) if series and label != series else name


def content_fmax(d: dict, kind: str, headroom: float = 1.5) -> float:
    """Upper frequency worth showing: above the highest Doppler peak, with room.

    A log axis compresses the empty top end to nothing, but a linear one gives
    it half the figure, so the linear plots are ranged to the signal instead.
    """
    f = d["f"][kind]
    sel = f >= 20e3                              # skip the DC end
    peaks = [f[sel][np.argmax(e[kind][sel])] for e in d["rpm"].values()]
    return max(max(peaks) * headroom, 1e6)


def style_axes(ax, kind: str, title: str | None = None, xscale: str | None = None,
               yscale: str | None = None, ylabel: str | None = None) -> None:
    """Shared axis styling. The scales and y label default to the kind's own.

    acquire_windtunnel.py draws the same quantity but wants it on log axes,
    where the whole spectrum out to Nyquist is the point, rather than the
    linear axes the RPM figures use to show the Doppler peaks. It overrides
    them here rather than keeping a second copy of this styling.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    spec = KINDS[kind]
    xscale = xscale or spec["xscale"]
    yscale = yscale or spec["yscale"]
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel("frequency (Hz)", color=C_INK)
    ax.set_ylabel(ylabel or spec["ylabel"], color=C_INK)
    ax.grid(True, which="major", color=C_GRID, lw=0.7, alpha=0.9)
    ax.grid(True, which="minor", color=C_GRID, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, which="both")
    if yscale == "log" and spec["yscale"] == "log":
        # The RPM y-range spans barely a decade, so label the 2s and 5s too.
        # A full spectrum spans several and is left to the default locator.
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    if xscale == "linear":
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e6:g}"))
        ax.set_xlabel("frequency (MHz)", color=C_INK)
    if title:
        ax.set_title(title, loc="left", color=C_INK)


def plot_overlay(d: dict, kind: str, out: Path, series: str, ref_tag: str,
                 fmax: float | None = None, label: str | None = None) -> None:
    import matplotlib.pyplot as plt

    import figstyle

    f = d["f"][kind]
    m = f > 0
    fig, ax = plt.subplots(figsize=(figstyle.TEXT_WIDTH, 3.6))
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)

    # A linear PSD axis is scaled by a power of ten, named in the axis label
    # rather than in an offset above the axis.
    curves = [e[kind] for e in d["rpm"].values()]
    exp = (int(np.floor(np.log10(max(amp(c, m, kind).max() for c in curves))))
           if KINDS[kind]["yscale"] == "linear" else 0)
    y = lambda power: amp(power, m, kind) / 10.0 ** exp
    ylabel = (rf"PSD ($10^{{{exp}}}$ V$^2$/Hz)" if exp else None)

    if d["ref"]:
        ref = np.mean([e[kind] for e in d["ref"]], axis=0)
        ax.plot(f[m], y(ref), color=C_MUTED, lw=1.2, ls=(0, (5, 4)),
                zorder=2,
                label=f"stationary, {shown_as(ref_tag, series, label or series)} "
                      f"(mean of {len(d['ref'])})")
    for colour, (rpm, e) in zip(series_colours(len(d["rpm"])), d["rpm"].items()):
        ax.plot(f[m], y(e[kind]), color=colour, lw=1.4, zorder=3,
                label=f"{rpm} RPM")

    if KINDS[kind]["xscale"] == "linear":
        ax.set_xlim(0, fmax or content_fmax(d, kind))
        ax.set_ylim(0, None)
    # Untitled: the report caption names the figure.
    style_axes(ax, kind, ylabel=ylabel)
    # The peaks sit low and to the left on the linear axes, leaving the top
    # right empty; on log axes the spectrum fills the plot, so go outside.
    inside = KINDS[kind]["xscale"] == "linear"
    leg = ax.legend(frameon=False, labelcolor=C_INK, title="target speed",
                    alignment="left",
                    **({"loc": "upper right"} if inside else
                       {"loc": "upper left", "bbox_to_anchor": (1.01, 1.0),
                        "borderaxespad": 0}))
    leg.get_title().set_color(C_MUTED)
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"plot -> {out}")


def plot_each(d: dict, kind: str, out_dir: Path, series: str, ref_tag: str,
              fmax: float | None = None, label: str | None = None) -> None:
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
    ylim = (lo / 1.15, hi * 1.15) if KINDS[kind]["yscale"] == "log" else (0, hi * 1.05)

    for colour, (rpm, e) in zip(series_colours(len(d["rpm"])), d["rpm"].items()):
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        fig.patch.set_facecolor(C_SURFACE)
        ax.set_facecolor(C_SURFACE)

        if ref is not None:
            ax.plot(f[m], amp(ref, m, kind), color=C_MUTED, lw=1.1, ls=(0, (5, 4)),
                    zorder=2,
                    label=f"stationary, {shown_as(ref_tag, series, label or series)}")
        ax.plot(f[m], amp(e[kind], m, kind), color=colour, lw=1.6, zorder=4,
                label=f"{rpm} RPM")

        ax.set_ylim(*ylim)
        if KINDS[kind]["xscale"] == "linear":
            ax.set_xlim(0, fmax or content_fmax(d, kind))
        style_axes(ax, kind, f"{label or series} - {rpm} RPM, "
                             f"{KINDS[kind]['name']}")
        # Peaks sit at the left on a linear axis and fill the lower band on a
        # log one, so the top right corner is the only reliably empty one.
        ax.legend(frameon=False, labelcolor=C_INK,
                  loc="upper right" if KINDS[kind]["xscale"] == "linear"
                  else "lower left")
        fig.tight_layout()
        out = out_dir / f"{kind}_{series or label}_{rpm}RPM.png"
        fig.savefig(out, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"plot -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--series", default="C1",
                    help="tag prefix that selects the captures; \"\" for tags "
                         "with no prefix, such as 500rpm_4m")
    ap.add_argument("--label", default="C",
                    help="name for the series in the figures; kept separate "
                         "from --series so selection stays exact. Output "
                         "filenames follow --series, or this when --series "
                         "is empty.")
    ap.add_argument("--ref", default="C1_ref",
                    help="tag prefix of the stationary reference captures")
    ap.add_argument("--date", help="YYYYmmdd; default is the latest date with RPM captures")
    ap.add_argument("--nperseg", type=int, default=12500,
                    help="Welch segment length; 12500 gives 10 kHz bins at 125 MS/s")
    ap.add_argument("--capture", type=int, default=1,
                    help="which capture of each RPM to plot, in time order (1 or 2)")
    ap.add_argument("--fmax", type=float,
                    help="upper frequency on the linear (PSD) axes, Hz; "
                         "default is 1.5x the highest Doppler peak")
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
    import figstyle
    figstyle.use()

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
                     args.series, args.ref, args.fmax, args.label)
        plot_each(d, kind, args.out_dir / f"{kind}_rpm", args.series, args.ref,
                  args.fmax, args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
