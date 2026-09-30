# Oscilloscope data acquisition (Keysight EDUX1002G)

Captures raw waveform records from channel 1 over USB and saves them with your
experiment metadata. No processing is done here.

## Install

1. Windows: install Keysight IO Libraries Suite (or NI-VISA).
   Linux/macOS: `pip install pyvisa-py pyusb` and set `visa_backend: "@py"` in the config.
2. `pip install -r requirements.txt`

## Find the VISA address

```
python Acquisition.py --list
```

This prints every visible instrument with its identity string. Paste the address into
`visa_address` in `daq_config.yaml`. Use the rear USB-B port on the scope.

The address for the lab's scope (serial CN58022160) is already in the config:

```
USB0::0x2A8D::0x179B::CN58022160::0::INSTR
```

## What this scope can actually do

Measured on the EDU-X 1002G with channel 2 off. Memory caps at 25 000 points in
HRESolution and 50 000 in NORMal, so the sample rate is set by the timebase:

| acquire_type | seconds_per_div | sample rate | points | record | Nyquist |
|---|---|---|---|---|---|
| HRESolution | 20.0e-6 | 125 MSa/s | 25000 | 200 us | 62.5 MHz |
| HRESolution | 100.0e-6 | 25 MSa/s | 25000 | 1 ms | 12.5 MHz |
| HRESolution | 400.0e-6 | 5 MSa/s | 20001 | 4 ms | 2.5 MHz |
| HRESolution | 1.0e-3 | 2.5 MSa/s | 25000 | 10 ms | 1.25 MHz |
| NORMal | 100.0e-6 | 50 MSa/s | 50000 | 1 ms | 25 MHz |
| NORMal | 400.0e-6 | 10 MSa/s | 40000 | 4 ms | 5 MHz |

Points are capped, so the sample rate is roughly `points / (10 * seconds_per_div)`,
quantised to 1-2-5 steps. Shortening the timebase buys bandwidth and costs frequency
resolution: the Welch bin floor is `1 / record length`. Change `seconds_per_div`,
`points` and `expected_sample_rate` together, using a row above.

Throughput is about 13 records/s, so 100 records takes roughly 8 seconds.

## Use

1. Edit the `metadata:` block in `daq_config.yaml` to describe the run.
2. Run:

```
python Acquisition.py --list                                       # find the scope's VISA address
python Acquisition.py --tag test --dry-run                         # connect, capture one record, print stats
python Acquisition.py --tag A1_laser_off                           # 100 records (from config)
python Acquisition.py --tag A1_leg_A_10mA --n 50 --note "leg A"    # override count, add a note
python Acquisition.py --tag manual --no-configure                  # keep front-panel scope settings
```

Ctrl-C stops early and still saves what was captured.

## Output

Each run writes to `data/`:

- `YYYYMMDD_HHMMSS_<tag>.npz` with arrays `raw` (int16, records x points), `t0_unix`,
  `preamble` (records x 10), `clipped`, and `meta_json`.
- `YYYYMMDD_HHMMSS_<tag>.json`: the same metadata plus a short summary (record count,
  rms of first/last record in mV, clipped count). Open this to check a run without loading the data.

Convert to volts in the processing script:

```python
import numpy as np
d = np.load("data/....npz")
pre = d["preamble"]                       # per record: [..., 4=xincrement, ..., 7=yinc, 8=yorig, 9=yref]
v = (d["raw"] - pre[:, 9:10]) * pre[:, 7:8] + pre[:, 8:9]
dt = pre[0, 4]
```

## Feeding the data to Welch.py

`prepare_welch.py` converts a raw capture into volts plus a sample rate. It does
the conversion only: no windowing, no PSD, no averaging.

Import it from `Welch.py`, which avoids a second copy of the data on disk:

```python
from scipy.signal import welch
from prepare_welch import load_run

run = load_run("data/20260925_123928_A1_laser_off.npz")
f, pxx = welch(run.v, fs=run.fs, nperseg=4096, axis=-1)
psd = pxx.mean(axis=0)          # ensemble average across records
```

`run.v` is float64 volts with shape `(n_records, n_samples)`, and `run.fs` is the
sample rate in Hz. Passing the 2-D array straight to `welch` with `axis=-1` gives
one periodogram per record; average those rather than concatenating the records,
since the records are not contiguous in time.

Or convert to a file first:

```
python prepare_welch.py data/20260925_123928_A1_laser_off.npz
python prepare_welch.py --all --drop-clipped
python prepare_welch.py --all --describe            # print contents, write nothing
python prepare_welch.py data/....npz --csv          # also dump record 0 as time_s,volts
```

That writes `<name>_volts.npz` containing:

| key | type | meaning |
|---|---|---|
| `v` | float64 (n_records, n_samples) | volts |
| `fs` | float64 scalar | sample rate, Hz |
| `dt` | float64 scalar | sample interval, s |
| `t0_unix` | float64 (n_records,) | wall-clock time of each record |
| `clipped` | bool (n_records,) | record touched the ADC rail |
| `meta_json` | str | metadata carried from the capture |

The converter refuses a file whose sample interval changed mid-run, since Welch
needs one sample rate. Each record is converted with its own preamble, so a scale
change between records cannot corrupt the result.

## Running the Welch analysis

`Welch.py` is a library plus a synthetic self-test. Running it directly checks the
scaling end to end against white noise of known density, and prints nothing about
your data:

```
python Welch.py
```

To analyse a real capture, use `run_welch.py`, which loads the file through
`prepare_welch`, runs `Welch.welch_psd`, and reports the band-averaged floor:

```
python run_welch.py data/20260925_123928_A1_laser_off.npz
python run_welch.py --all
python run_welch.py --all --plot
python run_welch.py data/....npz --band 1e6 60e6 --nperseg 5000 --save
python run_welch.py data/....npz --plot-file floor.png
```

Useful options:

| flag | meaning |
|---|---|
| `--nperseg` | segment length; resolution is `fs / nperseg` (default 2500) |
| `--band F_LO F_HI` | band to average, Hz (default 5e6 20e6) |
| `--gain` | transimpedance V/A; defaults from the capture metadata |
| `--droop` | apply the HRESolution sinc-squared correction |
| `--drop-clipped` | exclude records flagged as clipped |
| `--save` | write `<name>_psd.npz` with `f`, `psd`, `psd_per_record` |
| `--plot` / `--plot-file` | show or save the floor, nV/sqrt(Hz) and pA/sqrt(Hz) |

The transimpedance gain is taken from the capture metadata: 39 kV/A when
`terminator_50ohm` is false, 19.5 kV/A when true. The gain used is printed on
every run, because getting it wrong scales the input-referred current noise by two.

This rig currently runs **unterminated**, so the gain is 39 kV/A. The EDU-X 1002G
has a 1 Mohm input only, confirmed by `:CHANnel1:IMPedance FIFTy` being rejected
with "Data out of range", so the only way to terminate is an external 50 ohm
feed-through at the scope input. If you fit one, set `terminator_50ohm: true`.

Running unterminated doubles the signal at the scope, which helps at 5 mV/div, but
leaves the PD100B's 50 ohm output driving an unmatched 1 Mohm load through the coax.
The resulting reflections shape the frequency response, so treat the high-frequency
end of the floor as suspect until a terminated run confirms it.

Choosing `--nperseg` is a trade between resolution and error bar. At 25 000 points
per record, `--nperseg 2500` gives 19 segments per record; raising it to 12500
leaves only 3, and the runner warns below 4 segments. For finer resolution,
lengthen the record with a slower timebase rather than raising `--nperseg`.

To call it from your own code instead:

```python
from prepare_welch import load_run
from Welch import welch_psd, band_average, to_input_current

run = load_run("data/20260925_123928_A1_laser_off.npz")
f, psd, per = welch_psd(run.v, run.fs, nperseg=2500)
band, sem = band_average(f, per, 5e6, 20e6)
i_rt = (to_input_current(band, 39e3)) ** 0.5       # A/sqrt(Hz), unterminated
```

## Troubleshooting

- **`VI_ERROR_LIBRARY_NFOUND` on startup**: the script already works around this. Keysight's
  VISA implementation sits in `System32` but its dependencies live under
  `C:\Program Files\Keysight\IO Libraries Suite`, and Python 3.8+ no longer searches `PATH`
  for dependent DLLs. `_enable_keysight_dlls()` at the top of the script adds those folders.
  If Keysight is installed elsewhere, edit the `root` path in that function.
  Note `visa_backend: "@py"` does **not** work on Windows here, because the Keysight driver
  owns the USB device and pyvisa-py cannot claim it. Leave `visa_backend` empty.
- **Timeout on data transfer**: raise `scope.timeout` in `connect()` (default 20 s); check the USB cable.
- **Fewer points than expected**: channel 2 must be off (the script turns it off unless `--no-configure`).
  Also check the table above, since 25 000 points is the hard ceiling in HRESolution.
- **Wrong sample rate warning**: change `seconds_per_div` and `expected_sample_rate` together
  using a row from the table above.
- **Instrument not found**: rear USB-B port, cable, "Remote" shown on the scope, address matches Connection Expert.
- **Scope stuck in a menu**: press Back/Escape on the front panel and rerun.
- **Fallback without the PC**: front panel Save > Waveform > CSV to a USB stick (columns: time_s, volts).
