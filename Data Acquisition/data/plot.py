import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, uniform_filter1d
d = np.load("test_5mps_psd.npz"); f, p = d["f"], d["psd"]; df = f[1]-f[0]
m = (f >= 0.3e6) & (f <= 35e6); fm, pm = f[m], p[m]
L = np.log(pm)
base = uniform_filter1d(median_filter(L, size=81, mode="nearest"), 21, mode="nearest")
r = 10/np.log(10)*(L-base)
S = 11
rs = uniform_filter1d(r, S, mode="nearest")
s1 = 1.4826*np.median(np.abs(r-np.median(r)))          # per-bin scatter (dB)
sS = s1/np.sqrt(S)                                       # expected after S-bin smoothing
fD = 5*np.cos(np.radians(35))*2/1550e-9
INK, MUTE, BLUE, GRID = "#1b1f24", "#6e7781", "#2f6fd6", "#e6e8eb"
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 7.2), gridspec_kw=dict(height_ratios=[1, 1.15], hspace=0.38))
for a in (a1, a2):
    a.grid(color=GRID, lw=0.6); a.tick_params(colors=MUTE, labelsize=9)
    for s in a.spines.values(): s.set_color("#c9ced4")
a1.loglog(fm, pm, color=BLUE, lw=1.4, label="test_5mps PSD")
a1.loglog(fm, np.exp(base), color=INK, lw=1.0, ls=(0, (5, 3)), label="smooth baseline (4 MHz running median)")
a1.axvspan(0.8*fD, 1.2*fD, color="#f2c14e", alpha=0.25, lw=0)
a1.set_ylabel("PSD (V²/Hz)", color=INK); a1.set_xlabel("frequency (Hz)", color=INK)
a1.set_title("(a) PSD and fitted baseline", loc="left", fontsize=11, color=INK)
a1.legend(frameon=False, fontsize=9, loc="lower left")
a2.plot(fm/1e6, r, color="#b9c7dc", lw=0.7, label="per bin (50 kHz)")
a2.plot(fm/1e6, rs, color=BLUE, lw=1.6, label=f"smoothed over {S} bins ({S*df/1e6:.2f} MHz)")
for k in (5, -5): a2.axhline(k*sS, color=MUTE, lw=1, ls="--")
a2.axhline(0, color=INK, lw=0.8)
a2.axvspan(0.8*fD/1e6, 1.2*fD/1e6, color="#f2c14e", alpha=0.25, lw=0)
a2.text(fD/1e6, 0.155, f"expected f_D ≈ {fD/1e6:.1f} MHz", ha="center", fontsize=9, color=INK)
a2.text(34.5, 5*sS+0.008, "+5σ (smoothed)", ha="right", fontsize=8.5, color=MUTE)
a2.set_xlim(0, 35); a2.set_ylim(-0.18, 0.18)
a2.set_xlabel("frequency (MHz)", color=INK); a2.set_ylabel("PSD / baseline (dB)", color=INK)
a2.set_title("(b) Normalised residual: a Doppler return would be a bump in the shaded band", loc="left", fontsize=11, color=INK)
a2.legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle("test_5mps (500 records, nperseg 2500): no Doppler peak above 5σ", x=0.075, ha="left", fontsize=12.5, color=INK)
fig.savefig("test_5mps_normalised.png", dpi=160, bbox_inches="tight")
i = (fm > 0.8*fD) & (fm < 1.2*fD)
print(f"per-bin sigma {s1:.3f} dB, smoothed sigma {sS:.4f} dB, 5sigma {5*sS:.3f} dB = {100*(10**(5*sS/10)-1):.2f}%")
print(f"max smoothed in band {rs[i].max():+.3f} dB at {fm[i][rs[i].argmax()]/1e6:.2f} MHz; global |max| {np.abs(rs).max():.3f} dB")
print("floor at fD", np.interp(fD, fm, np.exp(base)))
