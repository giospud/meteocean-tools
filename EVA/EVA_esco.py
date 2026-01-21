import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto
from pathlib import Path
import seaborn as sns

p = Path("__file__").resolve().parents[1]

indir  = p / "EVA"
filein = 'Point_004708_out.dat'
outdir = indir / "plot"


Ith    = 24          # window's width [hr]
hth    = 2           # Hs threshold
thlim  = (180, 247.5)# directional fetch
nboot  = 1000        # bootstrap samples
alpha  = 0.05        # confidence level (1 - alpha)


def load_dicca(file_path):
    df = pd.read_csv(file_path, sep='\s+', header=None)
    date = pd.to_datetime(df.iloc[:,0].astype(str) + str(' ') + df.iloc[:,1].astype(str) + str(' ') + df.iloc[:,2].astype(str) + str(' ') + df.iloc[:,3])
    Hs = df[4].values
    Tp = df[5].values
    thp = df[7].values
    Ny = (date.iloc[-1] - date.iloc[0]).days / 365.25  # approx years
    return date, Hs, Tp, thp, Ny

date, Hs, Tp, thp, Ny = load_dicca(indir / filein)

def mov_wnd_peak(date, H, T, TH, Ith_hours=24):
    dt = pd.to_timedelta(np.median(np.diff(date))).total_seconds() / 3600
    wnd = int(Ith_hours / dt)
    peaks_H = []
    peaks_T = []
    peaks_TH = []
    for i in range(0, len(H) - wnd, wnd):
        idx = slice(i, i + wnd)
        if np.any(H[idx] > 0):
            imax = np.argmax(H[idx])
            peaks_H.append(H[idx][imax])
            peaks_T.append(T[idx][imax])
            peaks_TH.append(TH[idx][imax])
    return np.array(peaks_H), np.array(peaks_T), np.array(peaks_TH)

Hf, Tf, THf = mov_wnd_peak(date, Hs, Tp, thp, Ith)

mask = (THf >= thlim[0]) & (THf < thlim[1])
H = Hf[mask]
T = Tf[mask]
TH = THf[mask]

Hgp = H[H > hth]
Tgp = T[H > hth]

Hgp_sorted = np.sort(Hgp)
nExc = len(Hgp_sorted)             # Number of exceedances
lambda_ = nExc / Ny         # Mean number of events per year

# Empirical return period
ecdf = np.arange(1, nExc + 1) / (nExc + 1)
eTr = 1 / (lambda_ * (1 - ecdf))

# Fit GPD to exceedances
params = genpareto.fit(Hgp - hth)
shape, loc, scale = params  # loc should be ~0

Tr = np.arange(1, 1001)
P = 1 - 1 / (lambda_ * Tr)  # prob. of non-exceedance
He = genpareto.ppf(P, c=shape, loc=0, scale=scale) + hth

He_boot = np.full((nboot, len(Tr)), np.nan)

for i in range(nboot):
    sample = np.random.choice(Hgp, size=len(Hgp), replace=True)
    try:
        shape_b, loc_b, scale_b = genpareto.fit(sample - hth)
        He_boot[i, :] = genpareto.ppf(P, c=shape_b, loc=0, scale=scale_b) + hth
    except Exception:
        continue

# --- Plot Hs vs Tr
He_low  = np.nanquantile(He_boot, alpha / 2, axis=0)
He_high = np.nanquantile(He_boot, 1 - alpha / 2, axis=0)

plt.figure(figsize=(10, 5))
plt.semilogx(eTr, Hgp_sorted, 'o', color=[.7, .7, .7], markerfacecolor='w', label='Data')
plt.semilogx(Tr, He, '-k', label='GPD fit')
plt.semilogx(Tr, He_low, '--k', label='CI 95%')
plt.semilogx(Tr, He_high, '--k')
plt.xlabel('Tr [y]')
plt.ylabel('Hs [m]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(outdir / f"{filein[:12]}_HsTr_fit.png", bbox_inches='tight')
# plt.show()


# --- Plot Hs vs Tp
Tr_plt = [1, 25, 100, 1000]
He_Tr = [He[t - 1] for t in Tr_plt]

plt.figure(figsize=(8, 5))
for val in He_Tr:
    plt.axvline(val, color='g', linestyle='-')

plt.scatter(H, T, color=[.7, .7, .7], edgecolor='none', alpha=0.4)
sns.kdeplot(x=Hgp, y=Tgp, fill=True, cmap="Blues", bw_adjust=0.5, alpha=0.6)

for i, tr in enumerate(Tr_plt):
    plt.text(He_Tr[i], 0.2, f'{tr}y', rotation=90, color='g')

plt.xlabel('Hs [m]')
plt.ylabel('Tp [s]')
plt.grid()
plt.tight_layout()
plt.savefig(outdir / f"{filein[:12]}_HsTp.png", bbox_inches='tight')
# plt.show()


# --- Plot orientatiom
labels = ['NE', 'EA', 'SE', 'SO', 'SW', 'WE', 'NW', 'NW']  # 8 labels for 0–315°
angles = np.arange(0, 361, 45)  # 0, 45, 90, ..., 315

plt.figure(figsize=(10, 4))
sns.kdeplot(x=THf, y=Hf, fill=True, cmap="Blues", bw_adjust=0.7, alpha=0.8)

for i, (angle, label) in enumerate(zip(angles, labels)):
    plt.axvline(angle, color='k', linestyle=':')
    x_pos = 0.11 + i * 0.13
    plt.text(x_pos, 0.95, label, transform=plt.gca().transAxes, ha='center')

plt.xticks(np.arange(22.5, 360 + 1, 45))
plt.xlim([0, 360])
plt.xlabel(r'$\theta_p$ [°N]')
plt.ylabel(r'$H_s$ [m]')
plt.tight_layout()
plt.savefig(outdir / f"{filein[:12]}_HsTh_density.png", bbox_inches='tight')
# plt.show()


# --- Plot Hs vs Th (All values / Selected values)
plt.figure(figsize=(10, 4))
plt.plot(THf, Hf, 'o', color=[.7, .7, .7], label='All peaks')
D = T
plt.plot(TH, H, 'ko', label='Selected sector')
# Add sector lines
for angle in np.arange(0, 361, 45):
    style = '--' if (angle + 22.5) % 90 == 0 else ':'
    plt.axvline(angle, color='k', linestyle=style)
plt.xlabel(r'$\theta_p$ [°N]')
plt.ylabel(r'$H_s$ [m]')
plt.xticks(np.arange(22.5, 360 + 1, 45))
plt.xlim([0, 360])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(outdir / f"{filein[:12]}_HsTh_all.png", bbox_inches='tight')
# plt.show()


# --- Mean Residual Life plot
qth = np.concatenate([np.arange(0.1, 0.96, 0.05), np.arange(0.96, 1.00, 0.01)])
hth = np.quantile(H, qth)
mrl = np.full_like(hth, np.nan)         # average amount by which wave height exceeds threshold
lambda_ = np.full_like(hth, np.nan)
ci = np.full((len(hth), 2), np.nan)
for i, u in enumerate(hth):
    datafit = H[H > u] - u
    mrl[i] = np.mean(datafit)
    lambda_[i] = len(datafit) / Ny
    s = np.std(datafit, ddof=1)
    ci[i, 0] = mrl[i] - 1.96 * s / np.sqrt(len(datafit))
    ci[i, 1] = mrl[i] + 1.96 * s / np.sqrt(len(datafit))

# Color class bins
rng = np.arange(0, 105, 5)
uplim = rng[np.argmin(np.abs(rng - np.floor(np.max(lambda_))))]
nclass = uplim // 5
lbl = f"{thlim[0]}°N - {thlim[1]}°N"
fout = f"{filein[6:12]}_{thlim[0]}_{thlim[1]}_DT_{Ith}_MRL"
plt.figure()
plt.plot(hth, mrl, 'k-', label='MRL')
plt.plot(hth, ci[:, 0], 'k--')
plt.plot(hth, ci[:, 1], 'k--')

# Color-coded scatter points
sc = plt.scatter(hth, mrl, c=lambda_, cmap='jet', edgecolor='k', s=50)
plt.colorbar(sc, label=r'$\lambda$ [values/year]')
plt.clim(0, uplim)
plt.xlabel('Threshold $u$ [m]')
plt.ylabel('Mean Residual Life [m]')
plt.text(0.05, 0.05, lbl, transform=plt.gca().transAxes)
plt.grid(True)
plt.tight_layout()
plt.savefig(outdir / f"{filein[:12]}_MRL.png", bbox_inches='tight')
# plt.show()


# --- Plot Goodness-of-fit for different thresholds
qth = [0.50, 0.75, 0.90, 0.95, 0.98, 0.99]
hth = np.quantile(H, qth)

fout = f"{filein[6:12]}_{thlim[0]}_{thlim[1]}_DT_{Ith}_QQ"
plt.figure(figsize=(12, 8))

for i, u in enumerate(hth):
    datafit = H[H > u] - u
    n = len(datafit)
    if n < 5: continue  # Skip thresholds with too few data

    # Fit GPD
    sh, loc, sc = genpareto.fit(datafit, floc=0)

    prob = np.arange(1, n + 1) / (n + 1)
    thr_q = genpareto.ppf(prob, c=sh, scale=sc, loc=0) + u
    emp_q = np.sort(datafit) + u

    plt.subplot(2, 3, i + 1)
    plt.plot(thr_q, emp_q, 'o')
    plt.plot([0, 10], [0, 10], '--', color=[.7, .7, .7], linewidth=1.2)
    plt.axis('equal')
    plt.axis([0, 10, 0, 10])

    if i in [3, 4, 5]:
        plt.xlabel('Theoretical $H_s$ [m]')
    if i in [0, 3]:
        plt.ylabel('Empirical $H_s$ [m]')

    plt.text(0.05, 0.9, f'$H_{{th}}$ = {u:.2f} m', transform=plt.gca().transAxes)
    plt.text(0.05, 0.75, f'$\\lambda$ = {n / Ny:.2f}', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.gca().tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(outdir / f"{filein[:12]}_GoF.png", bbox_inches='tight')
