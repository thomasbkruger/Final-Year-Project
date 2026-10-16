"""noise_vs_windspeed.py - Doppler peak power against the local noise, per tunnel speed.

The wind-tunnel counterpart of noise_vs_rpm.py. For the dense-fog captures
wind_tunnel_doppler.py plots (one per speed, plus the 9 Oct 3.1 m/s capture) it
reports the same quantities in V^2/Hz:

  * the PSD at the peak centroid;
  * the mean noise PSD in a band around that peak, with the peak itself
    excluded;
  * the standard deviation of the noise there, and five times it.

The noise statistics come from noise_vs_rpm.measure, unchanged. Only the peak
is found differently: there is no no-flow reference to subtract, so the
centroid is taken on the excess over each PSD's own running median
(wind_tunnel_doppler.locate), exactly as in the Doppler-peak figure.

The mist runs follow in a table of their own, measured relative to that
running median instead, as the floor-normalised figures plot it: PSD / floor
in dB over the same noise band, the peak's height above the band's mean, and
that height in units of the band's scatter. The raw std counts the floor's
slope across the band as noise, which swamps the weak mist peaks and which
averaging cannot reduce; dividing by the floor removes it, leaving the random
scatter that averaging does reduce. Each mist speed gets one row per number of
captures averaged - the first capture, the first two, and so on in time order.

    python noise_vs_windspeed.py
    python noise_vs_windspeed.py --guard-mult 4 --window 3e6
    python noise_vs_windspeed.py --set rotated     # -> noise_vs_windspeed_rotated.txt

--set rotated reports the 16 Oct rotated-rig captures, picked as
wind_tunnel_doppler.py --set rotated picks them (0.2 V/div only, repeats at a
speed averaged, shown as xN). That set has no mist runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from acquire_windtunnel import psd_of
from noise_vs_rpm import GUARD_MULT, NOISE_FMIN, WINDOW, measure, report
from wind_tunnel_doppler import (BASELINE_BINS, F_MAX, F_MIN, SETS, captures,
                                  locate, measure_all, select)


def relative_to_floor(rows: list[dict], width: int, what: str = "mist runs") -> str:
    """The 5 sd test on PSD / own floor in dB, over the same noise band."""
    text = (f"\n\n{what}, relative to each PSD's own noise floor (PSD / running "
            "median, in dB,\nas in the floor-normalised figures), over the same "
            "noise band; xN = the\nfirst N captures of that speed, averaged "
            "in power:\n"
            f"\n{'U':>{width}} {'centroid':>9} {'FWHM':>8} {'peak':>9} "
            f"{'band mean':>10} {'band sd':>9} {'(peak-mean)/sd':>15} "
            f"{'clears 5 sd':>12}\n"
            f"{'':>{width}} {'(MHz)':>9} {'(MHz)':>8} {'(dB)':>9} {'(dB)':>10} "
            f"{'(dB)':>9} {'':>15} {'':>12}\n"
            + "-" * (width + 78) + "\n")
    for r in rows:
        r["rel_sigmas"] = (r["rel_peak_dB"] - r["rel_mean_dB"]) / r["rel_sd_dB"]
        text += (f"{r['x']:>{width}} {r['centroid']/1e6:9.3f} {r['fwhm']/1e6:8.3f} "
                 f"{r['rel_peak_dB']:9.3f} {r['rel_mean_dB']:10.3f} "
                 f"{r['rel_sd_dB']:9.4f} {r['rel_sigmas']:15.1f} "
                 f"{'yes' if r['rel_sigmas'] > 5 else 'NO':>12}\n")
    text += "\ncaptures:\n"
    for r in rows:
        text += f"  {r['x']:>{width}}  {r['time']}\n"
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--set", choices=SETS, default="unrotated",
                    help="which rig configuration to report")
    ap.add_argument("--date", help="YYYYmmdd; default is the set's own date")
    ap.add_argument("--label", default="D", help="name for the series in the report")
    ap.add_argument("--nperseg", type=int, default=2500,
                    help="Welch segment length; matches wind_tunnel_doppler.py")
    ap.add_argument("--window", type=float, default=WINDOW,
                    help="half-width of the noise band either side of the peak, Hz")
    ap.add_argument("--guard-mult", type=float, default=GUARD_MULT,
                    help="guard band excluded around the centroid, in FWHMs")
    ap.add_argument("--noise-fmin", type=float, default=NOISE_FMIN,
                    help="lowest frequency in the noise band, Hz")
    ap.add_argument("--out", type=Path,
                    help="default noise_vs_windspeed.txt, with the set's suffix")
    args = ap.parse_args()
    cfg = SETS[args.set]
    date = args.date or cfg["date"]
    out = args.out or Path(f"noise_vs_windspeed{cfg['out']}.txt")

    caps = select(args.data_dir, date, suffix=cfg["suffix"], extra=cfg["extra"],
                  vdiv=cfg["vdiv"], average=cfg["average"])
    if not caps:
        print(f"no {args.set} wind-tunnel captures in {args.data_dir} on {date}")
        return 1

    def row(psd, f, x: str, time: str) -> dict:
        p = locate(f, psd)
        r = measure(f, psd, None, args.window, args.guard_mult,
                    args.noise_fmin, peak=p, floor=p["floor"])
        r.update(x=x, time=time)
        return r

    f = measure_all(caps, args.nperseg)
    rows = [row(c["psd"], f,
                f"{'~' if c['approx'] else ''}{c['speed']:g} m/s"
                + (f" x{c['n_caps']}" if c["n_caps"] > 1 else ""),
                c["capture"])
            for c in caps]

    mist = []
    for c in select(args.data_dir, date, mist=True) if cfg["mist"] else []:
        tag = c["path"].stem[16:]
        runs = [psd_of(p, args.nperseg) | {"stem": p.stem[:15]}
                for p in captures(args.data_dir, date, tag)]
        for n in range(1, len(runs) + 1):
            used = runs[:n]
            # Averaged in linear power, weighted by records: one long capture.
            records = sum(d["n"] for d in used)
            psd = sum(d["psd"] * d["n"] for d in used) / records
            time = used[0]["stem"] + (f"..{used[-1]['stem'][9:]}" if n > 1 else "")
            mist.append(row(psd, f, f"{c['speed']:g} m/s mist x{n}", time))

    extra = sorted({c["path"].stem[:8] for c in caps} - {date})
    each = ("repeats at a speed averaged in power (xN)" if cfg["average"]
            else "one per speed setpoint")
    head = (
        f"Doppler peak power and local noise, {args.label} series (wind tunnel)\n"
        f"generated: {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}\n"
        f"captures:  {date}, {args.set} rig"
        + (f" (plus {', '.join(extra)})" if extra else "")
        + (f", {cfg['vdiv']:g} V/div only" if cfg["vdiv"] else "")
        + f", {each};\n"
        f"           Welch nperseg={args.nperseg} (df = {(f[1]-f[0])/1e3:g} kHz)\n"
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
        f"units:     V^2/Hz"
        + (", except the mist table (dB relative to the floor)" if mist
           else ", except the last table (dB relative to the floor)")
        + f"\n\n{cfg['name']}, raw PSD:\n"
    )
    width = max(len(r["x"]) for r in rows + mist)
    text = report(head, rows, "", ("U", ""), width=width)
    if mist:
        text += relative_to_floor(mist, width)
    else:
        # No mist runs to put there, but the weakest returns (21 m/s rotated)
        # sit on a sloping floor whose slope the raw std counts as noise, so
        # the floor-relative test is given for every speed as well.
        text += relative_to_floor(rows, width, f"{cfg['name']} runs")
    out.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
