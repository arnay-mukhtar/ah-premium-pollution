# Source: paper2_meteorology_build.ipynb, code cell 25 (final version of step 13 as run).
# ============================================================
# Step 13 — the first stage on the REAL estimation sample
#
# Step 11 computed the first stage on every day with pollution and
# meteorology: 622 for the 24h window. But the outcome only exists on
# days both markets traded, which is 390. Weekends remove 29% and
# Chinese New Year falls inside DJF. An IV first stage must be estimated
# on the sample the second stage will use, so step 11's F statistics
# were optimistic by roughly the ratio 622/390.
#
# This recomputes them on trading days, Shanghai-listed pairs only
# (Shenzhen costs zero days and has no instrument of its own).
#
# DISCIPLINE: reads ONLY the 'date' and 'a_code' columns of ah_panel.csv.
# No price, ratio or return column is loaded at all, so no relationship
# between pollution and the outcome can be inspected here even by
# accident. Verified by the column check printed below.
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

DIR = '/content/drive/MyDrive/pollution-ah'
DATA = '/content/drive/MyDrive/bilateral/data'

WIN_START, WIN_END = 7, 16
MIN_WIN_HOURS, MIN_DAY_HOURS = 7, 8
SAMPLE_START, SAMPLE_END = '2013-12-01', '2020-02-29'

# ------------------------------------------------------------
# 1. Trading days only, from the date column and nothing else
# ------------------------------------------------------------
USE = ['date', 'a_code']
cal = pd.read_csv(os.path.join(DIR, 'ah_panel.csv'), usecols=USE,
                  parse_dates=['date'], dtype={'a_code': str})
print('columns loaded from ah_panel.csv: %s' % list(cal.columns))
assert set(cal.columns) == set(USE), 'unexpected columns loaded'
print('  (no price, ratio or return column is read by this script)')

cal['a_code'] = cal['a_code'].str.zfill(6)
sh = cal[cal['a_code'].str.startswith('6')]
trading = pd.Index(sorted(sh['date'].unique()), name='date')
print('\nShanghai-listed trading days in panel: %d' % len(trading))

# ------------------------------------------------------------
# 2. Rebuild the first-stage inputs (as step 11)
# ------------------------------------------------------------
snd = pd.read_csv(os.path.join(DATA, 'sounding_daily.csv'),
                  parse_dates=['date'])
snd = snd[snd['city'] == 'shanghai'].drop(columns='city')

w = pd.read_csv(os.path.join(DATA, 'shanghai_weather_hourly.csv'),
                parse_dates=['time', 'date'])
b = pd.read_csv(os.path.join(DATA, 'shanghai_blh_hourly.csv'),
                parse_dates=['time', 'date'])
met_h = w.merge(b.drop(columns=['date', 'hour']), on='time', how='inner')

FIELDS = [f for f in ['temperature_2m', 'precipitation', 'wind_speed_10m',
                      'surface_pressure', 'cloud_cover',
                      'shortwave_radiation', 'u', 'v', 'blh_land_mean']
          if f in met_h.columns]

CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']
SETS = {
    'ERA5      ': ['log_blh', 'u', 'v', 'surface_pressure'],
    'soundings ': ['dT_200_100', 'dT_1000_300'],
    'combined  ': ['log_blh', 'u', 'v', 'surface_pressure',
                   'dT_200_100', 'dT_1000_300'],
}


def effective_f(y, Z, W):
    """Montiel Olea-Pflueger effective F, one endogenous regressor.

    Controls partialled out first, then
        F_eff = pi'(Z'Z)pi / tr((Z'Z) V_robust(pi))
    which equals the classical joint F under homoskedasticity.
    """
    Wc = sm.add_constant(W, has_constant='add')
    P = Wc.values @ np.linalg.pinv(Wc.values)
    yt = y.values - P @ y.values
    Zt = Z.values - P @ Z.values
    ZtZ = Zt.T @ Zt
    pi = np.linalg.solve(ZtZ, Zt.T @ yt)
    e = yt - Zt @ pi
    n, k = Zt.shape
    bread = np.linalg.inv(ZtZ)
    meat = Zt.T @ (Zt * (e ** 2)[:, None])
    V = bread @ meat @ bread * (n / max(n - k - Wc.shape[1], 1))
    return float(pi @ ZtZ @ pi / np.trace(ZtZ @ V))


def run(d, label):
    out = []
    for name, inst in SETS.items():
        s = d.dropna(subset=inst + ['log_pm25'] + CONTROLS).copy()
        if len(s) < 60:
            out.append((name, np.nan, np.nan, np.nan, len(s)))
            continue
        fe = pd.get_dummies(s[['month', 'year', 'dow']].astype(str),
                            drop_first=True).astype(float)
        W = pd.concat([s[CONTROLS].reset_index(drop=True),
                       fe.reset_index(drop=True)], axis=1)
        X = sm.add_constant(pd.concat([s[inst].reset_index(drop=True), W],
                                      axis=1))
        y = s['log_pm25'].reset_index(drop=True)
        full = sm.OLS(y, X).fit(cov_type='HC1')
        red = sm.OLS(y, X.drop(columns=inst)).fit(cov_type='HC1')
        pr2 = (full.rsquared - red.rsquared) / (1 - red.rsquared)
        R = np.zeros((len(inst), X.shape[1]))
        for i, nm in enumerate(inst):
            R[i, list(X.columns).index(nm)] = 1.0
        jf = float(np.squeeze(full.f_test(R).fvalue))
        out.append((name, jf, effective_f(y, s[inst].reset_index(drop=True),
                                          W), pr2, int(full.nobs)))
    for name, jf, fe_, pr2, n in out:
        print('%-10s %-11s %6d %9s %9s %11s'
              % (label, name, n,
                 '%.2f' % jf if jf == jf else 'n/a',
                 '%.2f' % fe_ if fe_ == fe_ else 'n/a',
                 '%.4f' % pr2 if pr2 == pr2 else 'n/a'))
    return out


print('\n' + '=' * 78)
print('SHANGHAI FIRST STAGE   outcome: log PM2.5')
print('=' * 78)
print('%-10s %-11s %6s %9s %9s %11s'
      % ('sample', 'instruments', 'N', 'joint F', 'F_eff', 'partial R2'))

results = {}
for tag, fn, lo, hi, need in [('24h', 'cn_pm25_daily_24h.csv', 0, 24,
                               MIN_DAY_HOURS),
                              ('win', 'cn_pm25_daily_win.csv', WIN_START,
                               WIN_END, MIN_WIN_HOURS)]:
    C = pd.read_csv(os.path.join(DATA, fn), index_col='date',
                    parse_dates=True)
    poll = C['Shanghai'].rename('pm25').reset_index()

    m = met_h[(met_h['hour'] >= lo) & (met_h['hour'] < hi)]
    g = m.groupby('date')
    met = g[FIELDS].mean().reset_index()
    met['n_hours'] = g.size().values
    met = met[met['n_hours'] >= need]

    d = met.merge(poll, on='date', how='inner').merge(snd, on='date',
                                                      how='left')
    d = d[(d['date'] >= SAMPLE_START) & (d['date'] <= SAMPLE_END)]
    d = d[d['pm25'] > 0].copy()
    d['log_pm25'] = np.log(d['pm25'])
    d['log_blh'] = np.log(d['blh_land_mean'].clip(lower=1))
    d['month'] = d['date'].dt.month
    d['year'] = d['date'].dt.year
    d['dow'] = d['date'].dt.dayofweek

    results[(tag, 'all days')] = run(d, '%s all' % tag)
    print()
    dt = d[d['date'].isin(trading)]
    results[(tag, 'trading')] = run(dt, '%s TRADING' % tag)
    print()

# ------------------------------------------------------------
# 3. What the November/March extension would buy
# ------------------------------------------------------------
print('=' * 78)
print('PROJECTION: November and March added (quotsoft, 11 months)')
print('=' * 78)
best = max((r for r in results[('24h', 'trading')] if r[1] == r[1]),
           key=lambda r: r[2])
name, jf, fe_, pr2, n = best
print('current best on trading days: %s  N %d  F_eff %.2f  partial R2 %.4f'
      % (name.strip(), n, fe_, pr2))
for mult, why in [(1.54, 'Nov+Mar 2014-2019 (approx +54% days)')]:
    print('   at N ~ %d (%s): F_eff ~ %.1f'
          % (round(n * mult), why, fe_ * mult))
print('\n   F scales roughly linearly with N at fixed partial R2, so this')
print('   is an estimate, not a promise. The real number needs the data.')

print("""
HOW TO READ IT
  The 'TRADING' rows are the only ones that mean anything. The 'all'
  rows are printed solely to show the size of the gap, since step 11
  reported the optimistic version.

  If F_eff on trading days sits near 8, the instrument is weak by any
  standard and the pre-registration must commit to Anderson-Rubin
  confidence sets, which stay valid at any instrument strength, and to
  reporting them as the primary inference rather than 2SLS t-statistics.
""")
