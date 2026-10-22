# Data acquisition and processing

Scripts for capturing signals from the oscilloscope and turning them into the
noise spectra, Doppler peaks and figures used in the report.

The workflow is always the same: **capture** a run with the scope, **process**
it into a spectrum, and **plot** the result. Each script below does one of
those jobs.

## Setup

- Python 3 with `numpy`, `scipy`, `matplotlib`, `pyyaml` and `pyvisa`.
- Windows: install Keysight IO Libraries Suite so the scope is visible over USB.
  Plug into the scope's rear USB-B port.
- Before capturing, edit the `metadata:` block in `daq_config.yaml` so the run
  is saved with a description of the experiment. The scope settings (volts per
  division, record length, sample rate) live in the same file, with a table of
  what the scope can do.

Run every script from this folder.

## Folders

| Folder | Contents |
|---|---|
| `data/` | Raw captures (`<date>_<time>_<tag>.npz` plus a readable `.json` summary), and the quick-look spectra `acquire_windtunnel.py` writes next to them |
| `figures/` | Every figure the report uses, plus the per-RPM figures in `figures/psd_rpm/` and three drawings made outside these scripts (`fig_rig_side3d.pdf`, `fig_doppler_vs_rpm.pdf`, `fig_velocity_both.pdf`) |
| `*.txt` in this folder | Number tables printed by the `noise_vs_*` and `psd_piin` scripts |

## Capturing

**`Acquisition.py`** - records waveforms from the scope and saves them. It does
no processing. `--tag` names the run and ends up in the filename.

```
python Acquisition.py --list                          # find the scope's address
python Acquisition.py --tag test --dry-run            # one record, print stats, save nothing
python Acquisition.py --tag A1_laser_off              # 100 records (from the config)
python Acquisition.py --tag B1_48mA --n 50 --note "..."
python Acquisition.py --tag manual --no-configure     # keep the scope's front-panel settings
```

Ctrl-C stops early and still saves what was captured.

**`acquire_windtunnel.py`** - the same capture, but 1000 records by default,
and it computes and plots the spectrum as soon as the capture finishes. Use it
for anything that needs many averages. `--ref` overlays an earlier run (a
no-flow or laser-off capture) for comparison.

```
python acquire_windtunnel.py --tag WT_noflow
python acquire_windtunnel.py --tag WT_10mps --ref WT_noflow
python acquire_windtunnel.py --tag WT_long --n 10000
python acquire_windtunnel.py --psd data/<file>.npz    # redo the spectrum of an existing capture
```

All `Acquisition.py` options pass straight through. Add `--no-psd` to capture only.

## Processing (library code)

These are imported by the other scripts rather than run on their own.

| Script | Purpose |
|---|---|
| `prepare_welch.py` | Loads a capture and converts the scope's raw numbers to volts |
| `prepare_welch_big.py` | The same, a block of records at a time, for captures too big to hold in memory |
| `Welch.py` | Turns voltage records into a power spectral density: how much noise there is at each frequency |
| `Welch_big.py` | The same, streaming, for very large captures (used by `acquire_windtunnel.py`) |
| `figstyle.py` | Figure size and font settings so plots match the report's text |

`prepare_welch.py` also works from the command line if you want a converted
copy of a capture:

```
python prepare_welch.py data/<file>.npz          # writes <file>_volts.npz
python prepare_welch.py --all --describe         # print what each capture contains
```

**`run_welch.py`** - quick look at one or more captures: prints the noise level
in a frequency band and can plot it.

```
python run_welch.py data/<file>.npz
python run_welch.py --all --plot
python run_welch.py data/<file>.npz --band 1e6 10e6 --nperseg 5000
```

## Plotting, by experiment

Each script picks the right captures out of `data/` by their tags, so they
normally run with no arguments. Figures go to `figures/`.

### A: noise floor vs laser current

**`psd_vs_current.py`** - how the noise floor grows with laser drive current,
for the balanced detector (default) or the single-ended one.

```
python psd_vs_current.py                      # -> figures/psd_vs_current.png
python psd_vs_current.py --detector single    # -> figures/psd_vs_current_single.png
```

### B: laser interference noise (PIIN)

**`psd_piin.py`** - the PIIN spectrum against the balanced reference and the
electronic floor. Several capture series can be overlaid; this is how the
patch-cable-length figure is made. A `.txt` with the band numbers is written
alongside.

```
python psd_piin.py                            # -> figures/psd_piin.png, psd_piin.txt
python psd_piin.py --series PIIN_ref PIIN_2m PIIN_3m PIIN_4m \
    --label "no matching" "2m patch cable" "3m patch cable" "4m patch cable" \
    --out figures/psd_piin_lengths.png
```

### C: rotating target

**`psd_rpm.py`** - the spectrum at each motor speed on one axis, plus one
figure per speed in `figures/psd_rpm/`.

```
python psd_rpm.py                             # latest set of RPM captures
python psd_rpm.py --series "" --ref PIIN_4m --date 20261009
```

**`peak_vs_rpm.py`** - where the Doppler peak sits at each speed, with a
straight-line fit.

```
python peak_vs_rpm.py                         # -> figures/peak_vs_rpm.png and .csv
```

**`noise_vs_rpm.py`** - how far each peak stands above the noise around it.

```
python noise_vs_rpm.py                        # -> noise_vs_rpm.txt
```

### D: wind tunnel

**`wind_tunnel_doppler.py`** - all the wind-tunnel figures: spectra at each
tunnel speed, measured peak against set speed, a grid of every test case, and
the effect of averaging on the weak mist runs. `--set rotated` repeats it for
the captures taken with the rig turned to face upstream.

```
python wind_tunnel_doppler.py                 # -> figures/psd_windtunnel_*.png, peak_vs_windtunnel_speed.png
python wind_tunnel_doppler.py --set rotated   # -> the same with a _rotated suffix
```

**`noise_vs_windspeed.py`** - peak height against surrounding noise at each
tunnel speed, for the dense-fog and mist runs.

```
python noise_vs_windspeed.py                  # -> noise_vs_windspeed.txt
python noise_vs_windspeed.py --set rotated    # -> noise_vs_windspeed_rotated.txt
```

```

