# Source: paper2_meteorology_build.ipynb, code cell 28 (final version of step 17 as run).
# ============================================================
# Step 17 — is the Zenodo series simply noisier?
#
# Step 16 section C: same days, same instruments, Zenodo partial R2
# 0.117, quotsoft 0.265, with near-identical log_blh coefficients
# (-0.546 vs -0.575). A slope that holds while R2 halves is the
# fingerprint of classical measurement error in the DEPENDENT variable.
#
# The test is a calibration, not a formula. Partial R2 is normalised by
# the reduced-model fit, so it falls FASTER than the raw reliability:
# simulation shows a partial R2 ratio of 0.47 corresponds to an R2
# between the series of about 0.86, not 0.47. So rather than assume a
# mapping, this adds noise to the quotsoft series at a grid of levels,
# runs the real specification on each, finds the level reproducing
# Zenodo's partial R2, and asks whether the correlation that level
# implies matches the correlation the two series actually have.
#
# Also redoes step 16 section A. Inside one winter the year dummies are
# collinear with the month dummies (December is one year, January and
# February the next), which made that design rank-deficient.
#
# Pollution and meteorology only.
# ============================================================

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DATA = '/content/drive/MyDrive/bilateral/data'
rng = np.random.default_rng(11)

# ------------------------------------------------------------
# 1. Inputs
# ------------------------------------------------------------
zen = (pd.read_csv(os.path.join(DATA, 'cn_pm25_daily_24h.csv'),
                   index_col='date', parse_dates=True)['Shanghai']
       .rename('zen').reset_index())

qs = pd.read_csv(os.path.join(DATA, 'quotsoft_shanghai_hourly.csv'),
                 parse_dates=['ts'])
gq = qs.groupby(qs['ts'].dt.normalize())['pm25']
qsd = (gq.mean().where(gq.count() >= 8).rename('qs').reset_index()
       .rename(columns={'ts': 'date'}).dropna())

om = pd.read_csv(os.path.join(DATA, 'shanghai_om_fullyear_hourly.csv'),
                 parse_dates=['time'])
rad = np.deg2rad(om['wind_direction_10m'])
om['u'] = -om['wind_speed_10m'] * np.sin(rad)
om['v'] = -om['wind_speed_10m'] * np.cos(rad)
om['date'] = om['time'].dt.normalize()
AGG = ['boundary_layer_height', 'wind_speed_10m', 'surface_pressure',
       'temperature_2m', 'precipitation', 'cloud_cover',
       'shortwave_radiation', 'u', 'v']
g = om.groupby('date')
met = g[AGG].mean().reset_index()
met['n_hours'] = g.size().values
met = met[met['n_hours'] >= 8].rename(
    columns={'boundary_layer_height': 'blh'})

base = met.merge(zen, on='date', how='left').merge(qsd, on='date',
                                                   how='left')
base['month'] = base['date'].dt.month
base['year'] = base['date'].dt.year
base['dow'] = base['date'].dt.dayofweek
base['winter'] = np.where(base['month'] == 12, base['year'],
                          base['year'] - 1)

CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']
INST = ['log_blh', 'u', 'v', 'surface_pressure']


def effective_f(y, Z, W):
    Wc = sm.add_constant(W, has_constant='add')
    P = Wc.values @ np.linalg.pinv(Wc.values)
    yt = y.values - P @ y.values
    Zt = Z.values - P @ Z.values
    ZtZ = Zt.T @ Zt
    pi = np.linalg.solve(ZtZ, Zt.T @ yt)
    e = yt - Zt @ pi
    n, k = Zt.shape
    br = np.linalg.inv(ZtZ)
    V = (br @ (Zt.T @ (Zt * (e ** 2)[:, None])) @ br
         * (n / max(n - k - Wc.shape[1], 1)))
    return float(pi @ ZtZ @ pi / np.trace(ZtZ @ V))


def design(s, fe_cols):
    fe = pd.get_dummies(s[fe_cols].astype(str), drop_first=True).astype(float)
    W = pd.concat([s[CONTROLS].reset_index(drop=True),
                   fe.reset_index(drop=True)], axis=1)
    X = sm.add_constant(pd.concat([s[INST].reset_index(drop=True), W], axis=1))
    return X, W


def fit(y, X, W, s):
    full = sm.OLS(y, X).fit(cov_type='HC1')
    red = sm.OLS(y, X.drop(columns=INST)).fit(cov_type='HC1')
    pr2 = (full.rsquared - red.rsquared) / (1 - red.rsquared)
    return (effective_f(y, s[INST].reset_index(drop=True), W), pr2,
            int(full.nobs), full.params['log_blh'], full.tvalues['log_blh'])


def prep(d, pm):
    s = d.dropna(subset=[pm, 'blh'] + CONTROLS + ['u', 'v',
                                                  'surface_pressure'])
    s = s[s[pm] > 0].copy()
    s['log_pm25'] = np.log(s[pm])
    s['log_blh'] = np.log(s['blh'].clip(lower=1))
    return s


# ------------------------------------------------------------
# 2. The two series side by side
# ------------------------------------------------------------
b = base.dropna(subset=['zen', 'qs'])
print('=' * 70)
print('THE TWO SERIES, %d OVERLAPPING DAYS' % len(b))
print('=' * 70)
lz, lq = np.log(b['zen']), np.log(b['qs'])
r_actual = float(np.corrcoef(lz, lq)[0, 1])
print('   correlation (levels)  %.4f' % b['zen'].corr(b['qs']))
print('   correlation (logs)    %.4f   -> R2 %.4f'
      % (r_actual, r_actual ** 2))
print('   mean   zenodo %6.2f   quotsoft %6.2f'
      % (b['zen'].mean(), b['qs'].mean()))
print('   sd     zenodo %6.2f   quotsoft %6.2f'
      % (b['zen'].std(), b['qs'].std()))
sl = sm.OLS(lz, sm.add_constant(lq)).fit()
print('   log(zen) on log(qs): slope %.4f (1.0 = pure noise difference)'
      % sl.params.iloc[1])
d_ = b['zen'] - b['qs']
print('   difference: mean %+.2f  sd %.2f  max abs %.1f'
      % (d_.mean(), d_.std(), d_.abs().max()))
print('\n   10 largest disagreements:')
t = b.assign(diff=d_).reindex(d_.abs().sort_values(ascending=False).index)
print(t.head(10)[['date', 'zen', 'qs', 'diff']].to_string(index=False))

# ------------------------------------------------------------
# 3. Observed partial R2 on identical days
# ------------------------------------------------------------
print('\n' + '=' * 70)
print('SAME DAYS, EACH SERIES AS THE DEPENDENT VARIABLE')
print('=' * 70)
obs = {}
for pm, lab in [('zen', 'Zenodo'), ('qs', 'quotsoft')]:
    s = prep(b, pm)
    X, W = design(s, ['month', 'year', 'dow'])
    r = fit(s['log_pm25'].reset_index(drop=True), X, W, s)
    obs[pm] = r[1]
    print('   %-9s N %4d  F_eff %6.2f  partial R2 %.4f  log_blh %+.4f  t %.2f'
          % (lab, r[2], r[0], r[1], r[3], r[4]))
ratio = obs['zen'] / obs['qs']
print('\n   ratio zen/qs = %.4f' % ratio)

# ------------------------------------------------------------
# 4. Calibration: how much noise reproduces that ratio?
# ------------------------------------------------------------
print('\n' + '=' * 70)
print('CALIBRATION: noise added to quotsoft, real design matrix')
print('=' * 70)
s0 = prep(b, 'qs')
X0, W0 = design(s0, ['month', 'year', 'dow'])
y0 = s0['log_pm25'].reset_index(drop=True)
sd0 = float(y0.std())

print('   %8s %12s %10s %12s' % ('noise sd', 'partial R2', 'ratio',
                                 'corr w/ qs'))
grid, best = [], None
for f in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
    yn = y0 + rng.normal(0, f * sd0, len(y0))
    full = sm.OLS(yn, X0).fit()
    red = sm.OLS(yn, X0.drop(columns=INST)).fit()
    p = (full.rsquared - red.rsquared) / (1 - red.rsquared)
    c = float(np.corrcoef(yn, y0)[0, 1])
    grid.append((f, p, p / obs['qs'], c))
    print('   %8.2f %12.4f %10.3f %12.4f' % (f, p, p / obs['qs'], c))
    if best is None or abs(p / obs['qs'] - ratio) < abs(best[2] - ratio):
        best = (f, p, p / obs['qs'], c)

print('\n   observed ratio                      %.3f' % ratio)
print('   closest simulated noise level       %.2f x sd' % best[0])
print('   correlation that noise level implies %.4f' % best[3])
print('   correlation the two series ACTUALLY have %.4f' % r_actual)
print("""
   VERDICT
     Implied and actual within about 0.03: classical measurement error
     in the Zenodo series explains the whole gap. quotsoft is simply the
     better measurement and should be the source for the full build.
     Actual correlation MUCH higher than implied: the two series agree
     too well for noise to explain the regression gap, and something
     systematic is going on that must be found before either is used.""")

# ------------------------------------------------------------
# 5. Winter by winter, year FE dropped
# ------------------------------------------------------------
print('\n' + '=' * 70)
print('WINTER BY WINTER (Zenodo), year FE dropped: they caused the')
print('rank deficiency in step 16')
print('=' * 70)
print('%-12s %6s %9s %12s %9s %7s %9s'
      % ('winter', 'N', 'F_eff', 'partial R2', 'log_blh', 't', 'mean PM'))
for yr in range(2013, 2020):
    sub = base[(base['winter'] == yr) & (base['month'].isin([12, 1, 2]))]
    s = prep(sub, 'zen')
    if len(s) < 60:
        print('%-12s %6d  too few days' % ('%d-%d' % (yr, yr + 1), len(s)))
        continue
    X, W = design(s, ['month', 'dow'])
    if np.linalg.matrix_rank(X.values) < X.shape[1]:
        print('%-12s %6d  still rank-deficient' % ('%d-%d' % (yr, yr + 1),
                                                   len(s)))
        continue
    r = fit(s['log_pm25'].reset_index(drop=True), X, W, s)
    print('%-12s %6d %9.2f %12.4f %9.4f %7.2f %9.1f'
          % ('%d-%d' % (yr, yr + 1), r[2], r[0], r[1], r[3], r[4],
             sub['zen'].mean()))
