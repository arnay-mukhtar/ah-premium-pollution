# Source: paper2_meteorology_build.ipynb, code cell 78 (final version of step 40 as run).
# ============================================================
# Step 40 — the three tables a referee will ask for
#
# A. OLS BESIDE 2SLS. An IV paper that never shows the OLS estimate
#    invites the question of what the instrument is actually doing. If
#    OLS and 2SLS agree, the endogeneity correction is doing little and
#    the reader should know. If they differ, that difference is the
#    paper's case for instrumenting at all.
#
# B. THE FULL FIRST STAGE. The paper reports only the boundary-layer
#    coefficient. All four instruments belong in a table with their
#    standard errors, computed on DAILY data, where the first stage
#    lives.
#
# C. THE HONG KONG ATTEMPTS. Section 6 claims six measure families
#    crossed with three control sets all fail, and shows no numbers.
#    Every other claim in the paper carries its evidence. This one
#    carries the paper's second contribution and currently carries a
#    sentence. This builds the table.
#
# C needs the Hong Kong pollution series. The script searches for it and
# reports what it finds rather than guessing.
# ============================================================

import os
import numpy as np
import pandas as pd

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DATA = '/content/drive/MyDrive/bilateral/data'
START, END = '2014-05-13', '2019-12-31'
LAGS = 7

INST = ['log_blh', 'u', 'v', 'surface_pressure']
CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']


def _dk(S, lags):
    om = S.T @ S
    for j in range(1, int(lags) + 1):
        if j >= len(S):
            break
        w = 1.0 - j / (lags + 1.0)
        G = S[j:].T @ S[:-j]
        om += w * (G + G.T)
    return om


def ols_dk(y, X, codes, G, lags=LAGS):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    n, k = X.shape
    S = np.zeros((G, k))
    np.add.at(S, codes, X * e[:, None])
    XtXi = np.linalg.pinv(X.T @ X)
    V = (XtXi @ _dk(S, lags) @ XtXi
         * (G / (G - 1.0)) * ((n - 1.0) / (n - k)))
    return b, np.sqrt(np.maximum(np.diag(V), 1e-300))


def tsls_dk(y, x, Z, W, codes, G, lags=LAGS):
    n = len(y)
    Zf = np.column_stack([np.ones(n), Z, W])
    Xf = np.column_stack([x, np.ones(n), W])
    P, *_ = np.linalg.lstsq(Zf, Xf, rcond=None)
    Xh = Zf @ P
    A = Xh.T @ Xf
    b = np.linalg.solve(A, Xh.T @ y)
    e = y - Xf @ b
    S = np.zeros((G, Xh.shape[1]))
    np.add.at(S, codes, Xh * e[:, None])
    Ai = np.linalg.inv(A)
    V = (Ai @ _dk(S, lags) @ Ai.T
         * (G / (G - 1.0)) * ((n - 1.0) / (n - Xf.shape[1])))
    return float(b[0]), float(np.sqrt(V[0, 0]))


def fe_matrix(s, cols=('month', 'year', 'dow')):
    cols = [c for c in cols if s[c].nunique() > 1]
    if not cols:
        return np.zeros((len(s), 0))
    return pd.get_dummies(s[list(cols)].astype(str),
                          drop_first=True).astype(float).values


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------
prem = pd.read_csv(os.path.join(DATA, 'ah_premium_clean.csv'),
                   parse_dates=['date'], dtype={'h_code': str},
                   usecols=['date', 'h_code', 'd_prem_raw'])
fs = pd.read_csv(os.path.join(DATA, 'shanghai_first_stage_fullyear.csv'),
                 parse_dates=['date'])
d = prem.merge(fs, on='date', how='inner')
d = d[(d['date'] >= START) & (d['date'] <= END)]
d = d.dropna(subset=['d_prem_raw', 'log_pm25'] + INST + CONTROLS).copy()
d = d.sort_values(['date', 'h_code'])
sd_pm = d.groupby('date')['log_pm25'].first().std()
BP = 10000.0 * sd_pm
_, codes = np.unique(d['date'].values, return_inverse=True)
G = codes.max() + 1
print('sample: %s firm-days, %d days, sd(log PM2.5) = %.4f'
      % (format(len(d), ','), G, sd_pm))

W = np.column_stack([d[CONTROLS].values, fe_matrix(d)])

# ------------------------------------------------------------
# A. OLS against 2SLS
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('TABLE A.  OLS AGAINST 2SLS, outcome = daily change in log premium')
print('=' * 74)
Xo = np.column_stack([d['log_pm25'].values, np.ones(len(d)), W])
bo, so = ols_dk(d['d_prem_raw'].values, Xo, codes, G)
bi, si = tsls_dk(d['d_prem_raw'].values, d['log_pm25'].values,
                 d[INST].values, W, codes, G)
print('   %-28s %12s %12s %8s %11s'
      % ('estimator', 'coef', 'se (DK7)', 't', 'bp per sd'))
print('   %-28s %12.6f %12.6f %8.2f %11.2f'
      % ('OLS', bo[0], so[0], bo[0] / so[0], bo[0] * BP))
print('   %-28s %12.6f %12.6f %8.2f %11.2f'
      % ('2SLS (registered)', bi, si, bi / si, bi * BP))
print('   %-28s %12.6f %12s %8s %11.2f'
      % ('2SLS minus OLS', bi - bo[0], '', '', (bi - bo[0]) * BP))
print('''
   Both near zero means pollution is close to exogenous here and the
   instrument changes little; the paper should say so plainly rather
   than implying the IV rescued something. A large gap is the case for
   instrumenting and belongs in the text.''')

# ------------------------------------------------------------
# B. The full first stage, on daily data
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('TABLE B.  FIRST STAGE, all instruments, daily')
print('=' * 74)
day = d.drop_duplicates('date').copy()
Wd = np.column_stack([day[CONTROLS].values, fe_matrix(day)])
Xd = np.column_stack([np.ones(len(day)), day[INST].values, Wd])
yd = day['log_pm25'].values
bd, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
ed = yd - Xd @ bd
nd, kd = Xd.shape
XtXi = np.linalg.pinv(Xd.T @ Xd)
Vd = XtXi @ (Xd.T @ (Xd * (ed ** 2)[:, None])) @ XtXi * (nd / (nd - kd))
sed = np.sqrt(np.maximum(np.diag(Vd), 1e-300))
print('   dependent variable: log PM2.5, %d trading days' % nd)
print('   %-24s %12s %12s %8s' % ('instrument', 'coef', 'se (HC1)', 't'))
for j, nm in enumerate(INST, start=1):
    print('   %-24s %12.5f %12.5f %8.2f'
          % (nm, bd[j], sed[j], bd[j] / sed[j]))
print('\n   controls (same regression):')
for j, nm in enumerate(CONTROLS, start=1 + len(INST)):
    print('   %-24s %12.5f %12.5f %8.2f'
          % (nm, bd[j], sed[j], bd[j] / sed[j]))

br, *_ = np.linalg.lstsq(np.delete(Xd, range(1, 1 + len(INST)), axis=1),
                         yd, rcond=None)
er = yd - np.delete(Xd, range(1, 1 + len(INST)), axis=1) @ br
print('\n   partial R2 of the four instruments: %.4f'
      % (1.0 - (ed @ ed) / (er @ er)))

# ------------------------------------------------------------
# C. The Hong Kong attempts
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('TABLE C.  HONG KONG, every dispersion measure tried')
print('=' * 74)
files = sorted(os.listdir(DATA))
print('   files in the data folder:')
for f in files:
    print('      %s' % f)

hk_pm = None
for f in files:
    if not f.endswith('.csv'):
        continue
    low = f.lower()
    if not any(k in low for k in ('hk', 'hong', 'epd')):
        continue
    try:
        t = pd.read_csv(os.path.join(DATA, f), nrows=5)
    except Exception:
        continue
    cols = [c.lower() for c in t.columns]
    if any('pm' in c for c in cols):
        print('\n   candidate with a PM column: %s -> %s'
              % (f, list(t.columns)))
        if hk_pm is None:
            hk_pm = f

if hk_pm is None:
    print('''
   No Hong Kong pollution file found by name. The listing above is the
   whole folder: send it and I will point at the right file, or at the
   script that built the HK series originally.

   Table C cannot be built without it, and Section 6 needs Table C more
   than the paper needs anything else outstanding.''')
else:
    print('\n   using %s as the Hong Kong pollution source' % hk_pm)
    print('   (re-run once I have wired the measure families in)')
