"""noise_vs_windspeed.py - Doppler peak power against the local noise, per tunnel speed.

The wind-tunnel counterpart of noise_vs_rpm.py, reporting the same quantities
in V^2/Hz for the dense-fog captures wind_tunnel_doppler.py plots (one per
speed, the 9 Oct capture at 3.5 m/s), followed by the mist runs. Each mist
speed gets one row per number of captures averaged - the first capture, the
first two, and so on in time order - since its return is weak enough that the
averaging matters:

  * the PSD at the peak centroid;
  * the mean noise PSD in a band around that peak, with the peak itself
    excluded;
  * the standard deviation of the noise there, and five times it.

The noise statistics come from noise_vs_rpm.measure, unchanged. Only the peak
is found differently: there is no no-flow reference to subtract, so the
centroid is taken on the excess over each PSD's own running median
(wind_tunnel_doppler.locate), exactly as in the Doppler-peak figure.

A second table repeats the test relative to that running median, as the
floor-normalised figures plot it: PSD / floor in dB over the same noise band,
the peak's height above the band's mean, and that height in units of the
band's scatter. The raw std counts the floor's slope across the band as noise,
and averaging cannot reduce that; dividing by the floor removes it, leaving
the random scatter that averaging does reduce.

    python noise_vs_windspeed.py
    python noise_vs_windspeed.py --guard-mult 4 --window 3e6
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from acquire_windtunnel import psd_of
from noise_vs_rpm import GUARD_MULT, NOISE_FMIN, WINDOW, measure, report
from wind_tunnel_doppler import (BASELINE_BINS, F_MAX, F_MIN, captures, locate,
                                  select)


def relative_to_floor(rows: list[dict], width: int) -> str:
    """The 5 sd test again, on PSD / own floor in dB over the same noise band."""
    text = ("\nrelative to each PSD's own noise floor (PSD / running median, in dB, "
            "as in the\nfloor-normalised figures), over the same noise band:\n"
            f"\n{'U':>{width}} {'peak':>9} {'band mean':>10} {'band sd':>9} "
            f"{'(peak-mean)/sd':>15} {'clears 5 sd':>12}\n"
            f"{'':>{width}} {'(dB)':>9} {'(dB)':>10} {'(dB)':>9} {'':>15} {'':>12}\n"
            + "-" * (width + 60) + "\n")
    for r in rows:
        r["rel_sigmas"] = (r["rel_peak_dB"] - r["rel_mean_dB"]) / r["rel_sd_dB"]
        text += (f"{r['x']:>{width}} {r['rel_peak_dB']:9.3f} {r['rel_mean_dB']:10.3f} "
                 f"{r['rel_sd_dB']:9.4f} {r['rel_sigmas']:15.1f} "
                 f"{'yes' if r['rel_sigmas'] > 5 else 'NO':>12}\n")
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--date", default="20261012")
    ap.add_argument("--label", default="D", help="name for the series in the report")
    ap.add_argument("--nperseg", type=int, default=2500,
                    help="Welch segment length; matches wind_tunnel_doppler.py")
    ap.add_argument("--window", type=float, default=WINDOW,
                    help="half-width of the noise band either side of the peak, Hz")
    ap.add_argument("--guard-mult", type=float, default=GUARD_MULT,
                    help="guard band excluded around the centroid, in FWHMs")
    ap.add_argument("--noise-fmin", type=float, default=NOISE_FMIN,
                    help="lowest frequency in the noise band, Hz")
    ap.add_argument("--out", type=Path, default=Path("noise_vs_windspeed.txt"))
    args = ap.parse_args()

    caps = select(args.data_dir, args.date)
    if not caps:
        print(f"no wind-tunnel captures in {args.data_dir} on {args.date}")
        return 1

    def row(psd, f, x: str, time: str) -> dict:
        p = locate(f, psd)
        r = measure(f, psd, None, args.window, args.guard_mult,
                    args.noise_fmin, peak=p, floor=p["floor"])
        r.update(x=x, time=time)
        return r

    rows = []
    for c in caps:
        d = psd_of(c["path"], args.nperseg)
        f = d["f"]
        rows.append(row(d["psd"], f,
                        f"{'~' if c['approx'] else ''}{c['speed']:g} m/s",
                        c["path"].stem[:15]))

    for c in select(args.data_dir, args.date, mist=True):
        tag = c["path"].stem[16:]
        runs = [psd_of(p, args.nperseg) | {"stem": p.stem[:15]}
                for p in captures(args.data_dir, args.date, tag)]
        for n in range(1, len(runs) + 1):
            used = runs[:n]
            # Averaged in linear power, weighted by records: one long capture.
            records = sum(d["n"] for d in used)
            psd = sum(d["psd"] * d["n"] for d in used) / records
            time = used[0]["stem"] + (f"..{used[-1]['stem'][9:]}" if n > 1 else "")
            rows.append(row(psd, f, f"{c['speed']:g} m/s mist x{n}", time))

    extra = sorted({c["path"].stem[:8] for c in caps} - {args.date})
    head = (
        f"Doppler peak power and local noise, {args.label} series (wind tunnel)\n"
        f"generated: {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}\n"
        f"captures:  {args.date}"
        + (f" (plus {', '.join(extra)})" if extra else "")
        + f", one per speed setpoint; Welch nperseg={args.nperseg} "
        f"(df = {(f[1]-f[0])/1e3:g} kHz)\n"
        f"mist xN:   the first N captures of that speed, averaged in power\n"
        f"peak:      power-weighted centroid (peak_vs_rpm.peak_of) of the excess "
        f"over the PSD's\n"
        f"           own running median ({BASELINE_BINS} bins), searched over "
        f"{F_MIN/1e6:g}-{F_MAX/1e6:g} MHz; the power quoted\n"
        f"           is the PSD of the bin nearest that centroid, not the largest "
        f"bin and not\n"
        f"           the integral under the peak\n"
        f"noise:     {args.guard_mult:g} x FWHM guard around the centroid, out "
        f"to {args.window/1e6:g} MHz either side,\n"
        f"           nothing below {args.noise_fmin/1e3:g} kHz (near-DC spike)\n"
        f"units:     V^2/Hz throughout\n"
    )
    width = max(len(r["x"]) for r in rows)
    text = report(head, rows, "", ("U", ""), width=width)
    text += relative_to_floor(rows, width)
    args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
