# Source: paper2_meteorology_build.ipynb, code cell 87 (final version of step 42 as run).
# ============================================================
# Step 42 — is the inversion clustered, and is Hong Kong in the cluster?
#
# The screen, on corrected coordinates, finds 22 of 365 cities with a
# positive and significant boundary-layer coefficient. Thirteen sit
# below 26N in the subtropical south, several within 120 km of the
# coast. Hong Kong is at 22.3N and 21 km from the coast.
#
# That is an eyeball pattern. This turns it into a number, because
# Section 6 cannot claim a regional regularity from a list read by eye.
#
#   1. Inversion rate by latitude band and by distance to coast.
#   2. A linear probability model of inversion on latitude and coast
#      distance, with robust standard errors.
#   3. The predicted inversion probability AT HONG KONG'S COORDINATES,
#      which is the number the paper needs: does the pattern place
#      Hong Kong in the failure region, or does it still predict an
#      ordinary first stage there?
#   4. Region labels from the province file, so the two clusters
#      (subtropical south, southwest plateau) are named rather than
#      asserted.
#   5. Figures 4 and 5 regenerated on clean coordinates.
#
# EXPLORATORY throughout. Built after the Hong Kong result and after
# seeing the screen. Reported as such.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DATA = '/content/drive/MyDrive/bilateral/data'
OUT = '/content/drive/MyDrive/bilateral/figures'
os.makedirs(OUT, exist_ok=True)
HK_LAT, HK_LON, HK_COAST = 22.3193, 114.1694, 21.0

INK, ACCENT, WARM, MUTED, BAND = ('#1a1a1a', '#1f4e79', '#b4471f',
                                  '#8a8a8a', '#e6e6e6')
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'savefig.bbox': 'tight',
})

COAST = np.array([
    (39.9, 124.4), (39.0, 121.7), (38.9, 117.7), (37.8, 120.7),
    (36.1, 120.4), (34.6, 119.2), (32.0, 121.7), (31.2, 121.5),
    (30.0, 122.1), (28.0, 121.1), (26.1, 119.3), (24.5, 118.1),
    (23.4, 116.7), (22.5, 114.1), (21.6, 111.0), (21.2, 110.4),
    (21.5, 109.1)])

R = pd.read_csv(os.path.join(DATA, 'city_screen_results.csv'),
                dtype={'city': str})
R['inv'] = ((R['b'] > 0) & (R['t'] > 1.96)).astype(float)
print('cities %d, inverted %d (%.1f%%)'
      % (len(R), int(R['inv'].sum()), 100 * R['inv'].mean()))

# Province labels, for naming the clusters rather than asserting them
prov = None
for f in ('city_coords_province.csv',):
    p = os.path.join(DATA, f)
    if os.path.exists(p):
        prov = pd.read_csv(p, dtype={'city': str})[['city', 'province']]
        R = R.merge(prov, on='city', how='left')
        break
if prov is None:
    R['province'] = ''
    print('no province file; cluster names will be blank')

# ------------------------------------------------------------
# 1. Rates by band
# ------------------------------------------------------------
print('\n' + '=' * 72)
print('1. INVERSION RATE BY LATITUDE')
print('=' * 72)
bands = [(0, 24), (24, 28), (28, 32), (32, 36), (36, 90)]
print('   %-14s %7s %9s %9s' % ('latitude', 'cities', 'inverted', 'rate'))
for lo, hi in bands:
    s = R[(R['lat'] >= lo) & (R['lat'] < hi)]
    if not len(s):
        continue
    print('   %-14s %7d %9d %8.1f%%'
          % ('%d-%d N' % (lo, hi), len(s), int(s['inv'].sum()),
             100 * s['inv'].mean()))

print('\n' + '=' * 72)
print('2. INVERSION RATE BY DISTANCE TO COAST, cities below 28N only')
print('=' * 72)
south = R[R['lat'] < 28]
cuts = [(0, 150), (150, 400), (400, 1000), (1000, 99999)]
print('   %-16s %7s %9s %9s' % ('km to coast', 'cities', 'inverted', 'rate'))
for lo, hi in cuts:
    s = south[(south['coast_km'] >= lo) & (south['coast_km'] < hi)]
    if not len(s):
        continue
    print('   %-16s %7d %9d %8.1f%%'
          % ('%d-%d' % (lo, min(hi, 9999)), len(s), int(s['inv'].sum()),
             100 * s['inv'].mean()))

# ------------------------------------------------------------
# 3. Linear probability model, and Hong Kong's predicted rate
# ------------------------------------------------------------
print('\n' + '=' * 72)
print('3. PROBABILITY OF INVERSION ON GEOGRAPHY')
print('=' * 72)
g = R.dropna(subset=['lat', 'coast_km', 'inv'])
X = np.column_stack([np.ones(len(g)), g['lat'].values,
                     np.log1p(g['coast_km'].values)])
y = g['inv'].values
b, *_ = np.linalg.lstsq(X, y, rcond=None)
e = y - X @ b
n, k = X.shape
XtXi = np.linalg.pinv(X.T @ X)
V = XtXi @ (X.T @ (X * (e ** 2)[:, None])) @ XtXi * (n / (n - k))
se = np.sqrt(np.maximum(np.diag(V), 1e-300))
for i, nm in enumerate(['constant', 'latitude', 'log(1+km to coast)']):
    print('   %-22s %+.5f  (se %.5f, t %+.2f)'
          % (nm, b[i], se[i], b[i] / se[i]))
print('   R2 %.3f, %d cities'
      % (1 - (e @ e) / np.sum((y - y.mean()) ** 2), n))

hk_x = np.array([1.0, HK_LAT, np.log1p(HK_COAST)])
hk_p = float(hk_x @ b)
hk_se = float(np.sqrt(hk_x @ V @ hk_x))
print('\n   PREDICTED INVERSION PROBABILITY AT HONG KONG')
print('   (%.2f N, %.0f km from coast):  %.3f  (se %.3f)'
      % (HK_LAT, HK_COAST, hk_p, hk_se))
print('   unconditional rate across all cities:   %.3f' % R['inv'].mean())
print('   ratio: %.1fx the national rate' % (hk_p / max(R['inv'].mean(), 1e-9)))
print("""
   A predicted probability well above the national rate means geography
   puts Hong Kong in the failure region, and its uninstrumentability is
   an instance of a regional pattern rather than an anomaly. A predicted
   probability near the national rate means the pattern does NOT explain
   Hong Kong, and Section 6 must say the two findings are separate.""")

# ------------------------------------------------------------
# 4. Naming the clusters
# ------------------------------------------------------------
print('\n' + '=' * 72)
print('4. THE INVERTED CITIES BY REGION')
print('=' * 72)
inv = R[R['inv'] > 0].sort_values(['lat'])
print('   %-14s %-16s %7s %8s %8s' % ('city', 'province', 'lat', 'coast',
                                      'b'))
for r in inv.itertuples():
    print('   %-14s %-16s %7.2f %8.0f %8.3f'
          % (r.city, str(getattr(r, 'province', ''))[:16], r.lat,
             r.coast_km, r.b))
sub = inv[inv['lat'] < 26]
print('\n   below 26N: %d of %d inverted cities (%.0f%%)'
      % (len(sub), len(inv), 100.0 * len(sub) / max(len(inv), 1)))
print('   of those, within 200 km of the coast: %d'
      % int((sub['coast_km'] < 200).sum()))

# ------------------------------------------------------------
# 5. Figures
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 4.6))
pos = R['inv'] > 0
ax.scatter(R.loc[~pos, 'lon'], R.loc[~pos, 'lat'], s=22, color=ACCENT,
           edgecolor='white', linewidth=0.4, zorder=2,
           label='negative (source-like)')
ax.scatter(R.loc[pos, 'lon'], R.loc[pos, 'lat'], s=42, color=WARM,
           marker='^', edgecolor='white', linewidth=0.5, zorder=3,
           label='positive (receptor-like)')
ax.plot(COAST[:, 1], COAST[:, 0], color=MUTED, lw=0.8, zorder=1)
ax.plot([HK_LON], [HK_LAT], marker='*', ms=15, color=INK, zorder=4,
        markeredgecolor='white', markeredgewidth=0.6)
ax.annotate('Hong Kong', (HK_LON, HK_LAT), textcoords='offset points',
            xytext=(8, -2), fontsize=8, color=INK)
sh = R[R['city'] == '上海']
if len(sh):
    ax.annotate('Shanghai', (float(sh['lon'].iloc[0]),
                             float(sh['lat'].iloc[0])),
                textcoords='offset points', xytext=(7, 3), fontsize=8,
                color=INK)
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.legend(frameon=False, fontsize=8, loc='lower left')
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, 'fig4_city_screen.%s' % ext), dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.hist(R['b'], bins=40, color=BAND, edgecolor=MUTED, linewidth=0.5)
ax.axvline(0, color=INK, lw=0.9)
if len(sh):
    v = float(sh['b'].iloc[0])
    ax.axvline(v, color=ACCENT, lw=1.6)
    ax.annotate('Shanghai', (v, ax.get_ylim()[1] * 0.9),
                textcoords='offset points', xytext=(4, 0), fontsize=8,
                color=ACCENT)
ax.set_xlabel('coefficient on log boundary layer height')
ax.set_ylabel('cities')
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, 'fig5_blh_distribution.%s' % ext),
                dpi=300)
plt.close(fig)
print('\nfigures 4 and 5 rewritten to %s' % OUT)
