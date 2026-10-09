"""
Welch PSD analysis for the Phase A / B noise measurements, for large captures.

Everything in Welch.py is here unchanged. What is added is a streaming path:
welch_psd_streaming consumes blocks of records and keeps only the running sum
of their PSDs, so the per-record PSDs never all exist at once.

That array is the second memory problem with a 10 000-record capture. Welch.py
returns psd_per_record, (n_records, n_freqs); at nperseg 12500 that is
10000 x 6251 x 8 = 500 MB, and the caller in acquire_windtunnel.py only ever
took its mean. Streaming holds one block's worth instead.

The per-record band average is still available, because it is only one float
per record: pass band=(f_lo, f_hi) and welch_psd_streaming returns the same
per-record values band_average would have been given, so the standard error
across records is unchanged.

Input : blocks of scope records, each (n_records_in_block, n_points), in VOLTS
        (already converted from ADC codes using the scope's
        :WAVeform:PREamble  y_increment / y_origin / y_reference).
Output: one-sided PSD in V^2/Hz, the input-referred current noise in A^2/Hz,
        and a band-averaged value with a statistical error bar.

"""

import numpy as np
from scipy import signal

# ---------------------------------------------------------------- constants
G_TIA = 39e3          # PD100B transimpedance into high-Z (V/A). Use 19.5e3 into 50 ohm.
Q = 1.602176634e-19   # electron charge (C)
R_PD = 0.9            # PD100B responsivity (A/W), nominal


def welch_psd(records, fs, nperseg=2500, overlap=0.5, window="hann",
              hres_droop_correct=False):
    """
    Welch PSD of each record, averaged across records.

    records : (n_records, n_points) array, volts
    fs      : sample rate the scope reports (Hz)
    nperseg : segment length -> resolution df = fs / nperseg
    Returns f (Hz), psd_mean (V^2/Hz), psd_per_record (n_records, n_freqs)

    Each record is processed on its own, then the records' PSDs are averaged.
    Records are NOT concatenated: there are gaps between captures, and a
    segment spanning a gap would add a spurious step.
    """
    records = np.atleast_2d(np.asarray(records, dtype=float))
    noverlap = int(nperseg * overlap)
    f, p = signal.welch(records, fs=fs, window=window, nperseg=nperseg,
                        noverlap=noverlap, detrend="constant",
                        scaling="density", return_onesided=True, axis=-1)
    if hres_droop_correct:
        # HRESolution block averaging behaves like a boxcar of length 1/fs:
        # |H(f)|^2 = sinc^2(f/fs).  Only apply once you have confirmed the
        # droop shape on a flat (laser-off) floor.
        h2 = np.sinc(f / fs) ** 2
        p = p / h2
    return f, p.mean(axis=0), p


def welch_psd_streaming(blocks, fs, nperseg=2500, overlap=0.5, window="hann",
                        hres_droop_correct=False, band=None):
    """
    Mean Welch PSD over a stream of record blocks, in constant memory.

    blocks : iterable of (m, n_points) float arrays in volts, any m per block
    band   : optional (f_lo, f_hi); when given, the per-record mean PSD in that
             band is collected, which is what band_average needs for its
             standard error. One float per record, so it costs nothing.

    Returns a dict:
        f            frequencies (Hz)
        psd          mean PSD across every record (V^2/Hz)
        n_records    how many records went into it
        band_per_record  (n_records,) or None

    The result is identical to welch_psd over the concatenation of the blocks:
    each record's PSD depends only on that record, so grouping cannot change
    the mean. Summing and dividing once at the end also avoids the drift a
    running mean would accumulate over 10 000 updates.
    """
    f = None
    total = None
    n = 0
    bands = [] if band is not None else None

    for blk in blocks:
        bf, _, p = welch_psd(blk, fs, nperseg=nperseg, overlap=overlap,
                             window=window, hres_droop_correct=hres_droop_correct)
        if f is None:
            f, total = bf, p.sum(axis=0)
        else:
            if not np.array_equal(bf, f):
                raise ValueError("frequency axis changed between blocks; "
                                 "every block must share one sample rate")
            total += p.sum(axis=0)
        n += p.shape[0]
        if bands is not None:
            sel = (f >= band[0]) & (f <= band[1])
            bands.append(p[:, sel].mean(axis=1))

    if n == 0:
        raise ValueError("no records in the stream")
    return {"f": f, "psd": total / n, "n_records": n,
            "band_per_record": np.concatenate(bands) if bands else None}


def band_average(f, psd_per_record, f_lo=5e6, f_hi=20e6):
    """
    Mean PSD in [f_lo, f_hi], computed per record, then averaged.
    Averaging is done in LINEAR units (V^2/Hz), never in dB.
    Returns (mean, standard error of the mean) in the PSD's units.
    """
    sel = (f >= f_lo) & (f <= f_hi)
    per_rec = psd_per_record[:, sel].mean(axis=1)
    n = per_rec.size
    sem = per_rec.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return per_rec.mean(), sem


def band_average_streamed(band_per_record):
    """band_average's result from the per-record values welch_psd_streaming kept."""
    per_rec = np.asarray(band_per_record)
    n = per_rec.size
    sem = per_rec.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return per_rec.mean(), sem


def to_input_current(psd_v, gain=G_TIA):
    """V^2/Hz at the scope -> A^2/Hz at the TIA input."""
    return psd_v / gain ** 2


def shot_noise_psd(p_lo_total_w, resp=R_PD):
    """Expected shot-noise current PSD (A^2/Hz) for total LO power on both diodes."""
    return 2 * Q * resp * p_lo_total_w


def welch_stats(n_points, nperseg, overlap, n_records):
    """Number of segments averaged and the resulting per-bin uncertainty."""
    step = nperseg - int(nperseg * overlap)
    segs_per_record = 1 + (n_points - nperseg) // step
    k = segs_per_record * n_records
    # For Hann with 50 % overlap the segments are nearly independent;
    # per-bin relative std ~ 1/sqrt(K)
    rel = 1 / np.sqrt(k)
    return {"segments_per_record": segs_per_record, "total_averages": k,
            "per_bin_rel_std": rel, "per_bin_dB": 10 * np.log10(1 + rel)}
