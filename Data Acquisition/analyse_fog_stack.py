"""analyse_fog_stack.py - stack fog-on wind-tunnel captures and look for the Doppler peak.

    python analyse_fog_stack.py --tag WT_fog_5mps
    python analyse_fog_stack.py --tag WT_fog_10mps --speed 10
    python analyse_fog_stack.py data/20261007_113004_test_5mps.npz data/...npz

Every record of every capture is given its own Welch PSD (Hann, 50 % overlap)
and divided by its own median over 1-10 MHz, so each record's floor sits at
about 1 whatever its LO power or gain that day. The normalised PSDs are then
averaged with equal weight per record, across all the captures together.

A broad baseline is taken from that average: a running median of log(PSD)
over --baseline-mhz, smoothed and exponentiated. The ratio PSD / baseline is
then ~1.0 everywhere except where something narrower than the median window
stands above the floor. A Doppler return is one such feature. Note that a
return much wider than --baseline-mhz is absorbed into the baseline.

Noise level, from the ratio itself (robust to a peak):
    sigma_bin    = 1.4826 * MAD(ratio)
    sigma_smooth = sigma_bin / sqrt(--smooth)

The expected line is f_D = 2 v cos(elev) / 1550 nm (5.28 MHz at 5 m/s, 35 deg).
Within f_D +/- 20 % the largest smoothed ratio is reported with its frequency,
the speed it implies and its height in sigma_smooth.

Captures are streamed CHUNK records at a time (prepare_welch_big.iter_volts,
the same volts conversion as prepare_welch.load_run), so memory stays low
for any number of records. Writes data/<tag>_stack.png and data/<tag>_stack.npz.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

import Acquisition
from acquire_windtunnel import captures
from prepare_welch_big import iter_volts, open_run
from psd_rpm import C_INK, C_MUTED, C_SURFACE, CAT, style_axes
from Welch_big import welch_psd, welch_stats

WAVELENGTH = 1550e-9
CHUNK = 100
NORM_BAND = (1e6, 10e6)     # each record is divided by its median PSD here
F_MIN = 0.3e6               # below this the spectrum is DC and 1/f, not floor
BASELINE_SMOOTH = 21        # bins of uniform smoothing on the log median
BAND_FRAC = 0.20            # Doppler search band, f_D +/- 20 %


def odd(n: float) -> int:
    n = max(int(round(n)), 1)
    return n if n % 2 else n + 1


def stack(paths: list[Path], nperseg: int) -> dict:
    """Mean of the median-normalised per-record Welch PSDs over every capture."""
    f = total = None
    n_rec = n_seg = 0
    for p in paths:
        info = open_run(p)
        if info.clipped.any():
            Acquisition.log(f"  WARNING: {p.name}: {int(info.clipped.sum())} clipped "
                            "records are included")
        n_here = 0
        for _, v in iter_volts(info, chunk=CHUNK):
            bf, _, psd = welch_psd(v, info.fs, nperseg=nperseg)
            if f is None:
                f, total = bf, np.zeros_like(bf)
                norm = (f >= NORM_BAND[0]) & (f <= NORM_BAND[1])
            elif not np.array_equal(bf, f):
                raise ValueError(f"{p.name}: sample rate differs from the first capture")
            psd /= np.median(psd[:, norm], axis=1, keepdims=True)
            total += psd.sum(axis=0)
            n_here += len(psd)
        segs = welch_stats(info.n_samples, nperseg, 0.5, n_here)
        n_rec += n_here
        n_seg += segs["total_averages"]
        Acquisition.log(f"  {p.name}: {n_here} records x "
                        f"{segs['segments_per_record']} segments")
    return {"f": f, "psd": total / n_rec, "n_records": n_rec, "n_segments": n_seg}


def analyse(f: np.ndarray, psd: np.ndarray, args) -> dict:
    """Baseline, ratio, noise and the Doppler search, all within F_MIN..fmax."""
    keep = (f >= F_MIN) & (f <= args.fmax)
    f, psd = f[keep], psd[keep]
    df = f[1] - f[0]

    med_bins = odd(args.baseline_mhz * 1e6 / df)
    log_base = median_filter(np.log(psd), size=med_bins, mode="reflect")
    baseline = np.exp(uniform_filter1d(log_base, BASELINE_SMOOTH, mode="reflect"))

    ratio = psd / baseline
    ratio_smooth = uniform_filter1d(ratio, args.smooth, mode="reflect")
    sigma_bin = 1.4826 * np.median(np.abs(ratio - np.median(ratio)))
    sigma_smooth = sigma_bin / np.sqrt(args.smooth)

    cos_e = np.cos(np.radians(args.elev))
    f_d = 2 * args.speed * cos_e / WAVELENGTH
    band = (f_d * (1 - BAND_FRAC), f_d * (1 + BAND_FRAC))
    sel = np.flatnonzero((f >= band[0]) & (f <= band[1]))
    if sel.size == 0:
        raise ValueError(f"Doppler band {band[0]/1e6:.2f}-{band[1]/1e6:.2f} MHz "
                         f"lies outside {F_MIN/1e6:g}-{args.fmax/1e6:g} MHz")
    i = sel[np.argmax(ratio_smooth[sel])]
    return {"f": f, "psd": psd, "baseline": baseline, "ratio": ratio,
            "ratio_smooth": ratio_smooth, "sigma_bin": sigma_bin,
            "sigma_smooth": sigma_smooth, "f_D": f_d, "band": band,
            "med_bins": med_bins, "peak_f": f[i], "peak_ratio": ratio_smooth[i],
            "peak_speed": f[i] * WAVELENGTH / (2 * cos_e),
            "peak_snr": (ratio_smooth[i] - 1) / sigma_smooth}


def plot(r: dict, title: str, args, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, band = r["f"], r["band"]
    s = r["sigma_smooth"]
    fig, (top, bot) = plt.subplots(2, 1, figsize=(9.2, 8.4),
                                   gridspec_kw={"height_ratios": [1, 1.25]})
    fig.patch.set_facecolor(C_SURFACE)
    band_label = f"expected f$_D$ ≈ {r['f_D']/1e6:.2f} MHz ({args.speed:g} m/s)"

    top.set_facecolor(C_SURFACE)
    top.axvspan(*band, color=CAT[3], alpha=0.18, lw=0, zorder=1, label=band_label)
    top.plot(f, r["psd"], color=CAT[0], lw=1.2, zorder=3, label="normalised mean PSD")
    top.plot(f, r["baseline"], color=C_MUTED, lw=1.2, ls=(0, (5, 4)), zorder=4,
             label=f"baseline (running median, {args.baseline_mhz:g} MHz)")
    style_axes(top, "psd", title, xscale="log", yscale="log",
               ylabel="PSD / record median (1–10 MHz)")
    top.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=C_INK)

    bot.set_facecolor(C_SURFACE)
    bot.axvspan(*band, color=CAT[3], alpha=0.18, lw=0, zorder=1, label=band_label)
    bot.plot(f, r["ratio"], color=CAT[0], lw=0.5, alpha=0.35, zorder=2,
             label="per bin")
    bot.plot(f, r["ratio_smooth"], color=CAT[0], lw=1.8, zorder=3,
             label=f"smoothed ({args.smooth} bins)")
    bot.axhline(1.0, color=C_INK, lw=0.9, zorder=4)
    for k in (1, -1):
        bot.axhline(1 + 5 * k * s, color=CAT[7], lw=1.0, ls=(0, (5, 4)), zorder=4,
                    label=f"1 ± 5σ$_{{smooth}}$ ({5 * s * 100:.2f} %)" if k > 0 else None)
    style_axes(bot, "psd", "PSD / baseline", xscale="linear", yscale="linear",
               ylabel="ratio (linear)")
    bot.set_xlim(0, args.fmax)
    bot.set_ylim(1 - 8 * s, 1 + 8 * s)
    bot.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=C_INK, ncol=2)

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help="capture .npz files to stack")
    ap.add_argument("--tag", help="stack every capture whose filename tag is exactly this")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--speed", type=float, default=5.0, help="wind speed, m/s (default 5)")
    ap.add_argument("--elev", type=float, default=35.0,
                    help="beam elevation, deg (default 35)")
    ap.add_argument("--nperseg", type=int, default=2500,
                    help="Welch segment length (default 2500: 50 kHz bins at 125 MSa/s)")
    ap.add_argument("--baseline-mhz", type=float, default=4.0,
                    help="running-median width of the baseline, MHz (default 4)")
    ap.add_argument("--smooth", type=int, default=11,
                    help="bins of uniform smoothing on the ratio (default 11)")
    ap.add_argument("--fmax", type=float, default=35e6,
                    help="upper frequency limit, Hz (default 35e6)")
    args = ap.parse_args()

    paths = list(args.files)
    if args.tag:
        paths += sorted(p for p in captures(args.data_dir) if p.stem[16:] == args.tag)
    if not paths:
        ap.error(f"no captures tagged {args.tag!r} in {args.data_dir}" if args.tag
                 else "give --tag or one or more capture files")
    bad = [p for p in paths if p.stem.endswith(("_psd", "_volts", "_stack"))]
    if bad:
        ap.error(f"not raw captures: {', '.join(p.name for p in bad)}")
    tags = {p.stem[16:] for p in paths}
    tag = args.tag or (tags.pop() if len(tags) == 1 else "stack")
    out_dir = args.data_dir if args.tag else paths[0].parent

    Acquisition.log(f"Stacking {len(paths)} capture(s), nperseg={args.nperseg}")
    s = stack(paths, args.nperseg)
    Acquisition.log(f"  total: {len(paths)} captures, {s['n_records']} records, "
                    f"{s['n_segments']} Welch segments "
                    f"(expected per-bin scatter ~{100 / np.sqrt(s['n_segments']):.2f} %)")

    r = analyse(s["f"], s["psd"], args)
    title = f"{tag}: {s['n_records']} records, {args.speed:g} m/s"
    out_png = out_dir / f"{tag}_stack.png"
    out_npz = out_dir / f"{tag}_stack.npz"
    plot(r, title, args, out_png)
    np.savez_compressed(out_npz, f=r["f"], psd=r["psd"], baseline=r["baseline"],
                        ratio=r["ratio"], ratio_smooth=r["ratio_smooth"],
                        sigma_bin=r["sigma_bin"], sigma_smooth=r["sigma_smooth"],
                        f_D=r["f_D"], files=np.array([str(p) for p in paths]))
    Acquisition.log(f"  baseline median over {r['med_bins']} bins; "
                    f"sigma_bin {r['sigma_bin']*100:.3f} %, "
                    f"sigma_smooth {r['sigma_smooth']*100:.3f} %")
    Acquisition.log(f"  saved {out_png.name}, {out_npz.name}")
    print(f"{tag}: {s['n_records']} rec, f_D {r['f_D']/1e6:.2f} MHz, "
          f"band max {r['peak_ratio']:.4f} at {r['peak_f']/1e6:.2f} MHz "
          f"({r['peak_speed']:.2f} m/s), {r['peak_snr']:+.1f} sigma_smooth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
