"""noise_vs_rpm.py - Doppler peak power against the local noise, per RPM.

For every RPM capture this reports, in V^2/Hz:

  * the PSD at the peak centroid;
  * the mean noise PSD in a band around that peak, with the peak itself
    excluded;
  * the standard deviation of the noise there, and five times it.

The peak is located by its power-weighted centroid, not by the bin of maximum
PSD, using peak_vs_rpm.peak_of so that this agrees with the Doppler fit. A
single bin is a noisy estimator, and the 50 RPM peak is two-lobed, so the
argmax lands on whichever lobe happens to be taller. The centroid is taken on
the excess over the stationary reference, which keeps the broadband floor from
dragging it toward the middle of the window.

Noise is measured next to each peak rather than once for the whole spectrum,
because the floor is not flat: it rises towards DC and rolls off above ~20 MHz,
so a single figure would understate the noise at 50 RPM and overstate it at
2000 RPM. The excluded guard band is set from the measured FWHM, widened where
needed so it always covers the half-maximum points either side of the centroid.
The noise band also stops short of DC (--noise-fmin): the first few bins above
the peak-search limit carry a low-frequency spike - 20x the floor at 20 kHz in
the 9 Oct 1000 RPM capture - which would otherwise dominate the std of any peak
whose band reaches down that far.

    python noise_vs_rpm.py
    python noise_vs_rpm.py --guard-mult 4 --window 3e6
    python noise_vs_rpm.py --capture 2 --out noise_vs_rpm_cap2.txt

Captures are selected exactly as in psd_rpm.py, so the numbers correspond to
the figures that script draws.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from peak_vs_rpm import peak_of
from psd_rpm import collect

FMIN = 20e3          # skip the DC end; the 50 RPM peak sits near 60 kHz
NOISE_FMIN = 50e3    # noise band starts here, clear of the near-DC spike


def measure(f: np.ndarray, psd: np.ndarray, ref: np.ndarray | None,
            window: float, guard_mult: float,
            noise_fmin: float = NOISE_FMIN) -> dict:
    """Centroid peak power and the noise statistics beside it, for one capture."""
    excess = np.clip(psd - ref, 0, None) if ref is not None else psd
    p = peak_of(f, excess, f_min=FMIN)
    centroid = p["centroid"]
    fwhm = p["fwhm_hi"] - p["fwhm_lo"]

    # Nearest bin to the centroid; the centroid rarely lands on a bin centre.
    j = int(np.argmin(np.abs(f - centroid)))

    # Keep the whole peak out of the noise band, not just its half-maximum part.
    guard = max(guard_mult * fwhm / 2,
                centroid - p["fwhm_lo"], p["fwhm_hi"] - centroid)

    off = np.abs(f - centroid)
    noise = (f >= noise_fmin) & (off > guard) & (off <= window)
    peak_sel = (f >= FMIN) & (off <= guard)

    below = int((noise & (f < centroid)).sum())
    above = int((noise & (f > centroid)).sum())
    mean = float(psd[noise].mean())
    std = float(psd[noise].std(ddof=1))
    # Excess power carried by the peak itself, above the local noise.
    excess_pwr = float(np.trapezoid(np.clip(psd[peak_sel] - mean, 0, None),
                                    f[peak_sel]))
    return {"centroid": centroid, "f_argmax": p["argmax"], "fwhm": fwhm,
            "guard": guard, "peak": float(psd[j]), "mean": mean, "std": std,
            "five_sigma": 5 * std, "n_noise": int(noise.sum()),
            "n_below": below, "n_above": above, "excess_power": excess_pwr}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--series", default="C1", help="tag prefix selecting the captures")
    ap.add_argument("--label", default="C", help="name for the series in the report")
    ap.add_argument("--ref", default="C1_ref")
    ap.add_argument("--date", help="YYYYmmdd; default is the latest date with RPM captures")
    ap.add_argument("--nperseg", type=int, default=12500,
                    help="Welch segment length; matches psd_rpm.py")
    ap.add_argument("--capture", type=int, default=1,
                    help="which capture of each RPM to use, in time order")
    ap.add_argument("--window", type=float, default=2e6,
                    help="half-width of the noise band either side of the peak, Hz")
    ap.add_argument("--guard-mult", type=float, default=3.0,
                    help="guard band excluded around the centroid, in FWHMs")
    ap.add_argument("--noise-fmin", type=float, default=NOISE_FMIN,
                    help="lowest frequency in the noise band, Hz; keeps the "
                         "near-DC spike out of the noise statistics")
    ap.add_argument("--out", type=Path, default=Path("noise_vs_rpm.txt"))
    args = ap.parse_args()

    d = collect(args.data_dir, args.series, args.ref, args.date, args.nperseg)
    if not d["rpm"]:
        print(f"no {args.series} RPM captures in {args.data_dir}"
              + (f" on {args.date}" if args.date else ""))
        return 1

    f = d["f"]["psd"]
    ref = np.mean([e["psd"] for e in d["ref"]], axis=0) if d["ref"] else None
    if ref is None:
        print(f"note: no {args.ref} captures on {d['date']}; the centroid is "
              "taken on the raw PSD, so the broadband floor pulls it slightly")

    rows = []
    for rpm, caps in sorted(d["rpm"].items()):
        if not 1 <= args.capture <= len(caps):
            print(f"{rpm} RPM has {len(caps)} capture(s); cannot pick #{args.capture}")
            return 1
        cap = caps[args.capture - 1]
        r = measure(f, cap["psd"], ref, args.window, args.guard_mult,
                    args.noise_fmin)
        r.update(rpm=rpm, time=cap["time"])
        rows.append(r)

    head = (
        f"Doppler peak power and local noise, {args.label} series\n"
        f"generated: {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}\n"
        f"captures:  {d['date']}, capture #{args.capture} of each RPM, "
        f"Welch nperseg={args.nperseg} (df = {(f[1]-f[0])/1e3:g} kHz)\n"
        f"peak:      power-weighted centroid (peak_vs_rpm.peak_of) of the "
        f"excess over {args.ref};\n"
        f"           the power quoted is the PSD of the bin nearest that "
        f"centroid, not the\n"
        f"           largest bin and not the integral under the peak\n"
        f"noise:     {args.guard_mult:g} x FWHM guard around the centroid, out "
        f"to {args.window/1e6:g} MHz either side,\n"
        f"           nothing below {args.noise_fmin/1e3:g} kHz (near-DC spike)\n"
        f"units:     V^2/Hz throughout\n"
    )
    cols = (f"\n{'RPM':>6} {'centroid':>9} {'FWHM':>8} {'PSD at':>12} "
            f"{'mean noise':>12} {'std':>12} {'5 x std':>12} {'mean+5sd':>12} "
            f"{'cent/mean':>10} {'bins':>6}\n"
            f"{'':>6} {'(MHz)':>9} {'(MHz)':>8} {'centroid':>12} {'(V^2/Hz)':>12} "
            f"{'(V^2/Hz)':>12} {'(V^2/Hz)':>12} {'(V^2/Hz)':>12} {'(dB)':>10} {'':>6}\n"
            + "-" * 116 + "\n")

    body = ""
    for r in rows:
        thresh = r["mean"] + r["five_sigma"]
        body += (f"{r['rpm']:6d} {r['centroid']/1e6:9.3f} {r['fwhm']/1e6:8.3f} "
                 f"{r['peak']:12.4e} {r['mean']:12.4e} {r['std']:12.4e} "
                 f"{r['five_sigma']:12.4e} {thresh:12.4e} "
                 f"{10*np.log10(r['peak']/r['mean']):10.1f} {r['n_noise']:6d}\n")

    missed = [r["rpm"] for r in rows if r["peak"] <= r["mean"] + r["five_sigma"]]
    tail = ("\nevery centroid clears mean + 5 sd\n" if not missed else
            "\ncentroids NOT clearing mean + 5 sd: "
            + ", ".join(f"{m} RPM" for m in missed) + "\n")

    # A peak close to DC has no room for a noise band below it, so its mean is
    # read off the high side only, where the floor sits lower.
    lopsided = [r for r in rows if min(r["n_below"], r["n_above"]) < 0.25 * r["n_noise"]]
    if lopsided:
        tail += ("\nnoise band one-sided - too close to DC to fit a band below "
                 "the peak - so the\nfloor is read off the high side only and is "
                 "likely a slight underestimate:\n")
        for r in lopsided:
            tail += (f"  {r['rpm']:5d} RPM  {r['n_below']} bins below the peak / "
                     f"{r['n_above']} above\n")

    tail += "\nintegrated excess power under each peak, above the local noise:\n"
    for r in rows:
        tail += (f"  {r['rpm']:5d} RPM  {r['excess_power']:.4e} V^2  "
                 f"(guard +/-{r['guard']/1e6:.3f} MHz, capture {r['time']})\n")

    tail += ("\ncentroid against the largest bin, as a check on how far the two "
             "disagree:\n")
    for r in rows:
        tail += (f"  {r['rpm']:5d} RPM  centroid {r['centroid']/1e6:6.3f} MHz  "
                 f"argmax {r['f_argmax']/1e6:6.3f} MHz  "
                 f"offset {(r['centroid']-r['f_argmax'])/1e3:+7.1f} kHz\n")

    report = head + cols + body + tail
    args.out.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
