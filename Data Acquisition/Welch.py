"""
Welch PSD analysis for the Phase A / B noise measurements.

Input : a 2-D array of scope records, shape (n_records, n_points), in VOLTS
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


