# Source: paper2_meteorology_build.ipynb, code cell 33 (final version of step 19 as run).
# ============================================================
# Step 19 — the first stage that decides the paper
#
# 1,559 trading days against the 390 the winter-only Zenodo build gave.
# Step 13's best was F_eff 8.22 with partial R2 0.074.
#
# Boundary layer comes from Open-Meteo, full year. That is licensed by
# step 16, which found CDS box, CDS point and Open-Meteo point give
# identical results on the same days (0.2641, 0.2646, 0.2647). Since the
# source does not matter, there is no need to pull twelve months of CDS.
#
# Reports:
#   - the headline first stage on all trading days
#   - by year, because the relationship may not be stable over a period
#     when Shanghai's mean PM2.5 fell by a third
#   - by season, to confirm step 15's finding holds out of sample
#   - with and without 2020, so the COVID decision is made on evidence
#
# Pollution and meteorology only. Reads only 'date' and 'a_code' from
# the panel; no price, ratio or return column is loaded.
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
DIR = '/content/drive/MyDrive/pollution-ah'
CITY = '上海'

# ------------------------------------------------------------
# 1. Inputs
# ------------------------------------------------------------
D = pd.read_csv(os.path.join(DATA, 'quotsoft_cities_daily.csv'),
                usecols=['date', CITY], parse_dates=['date'])
poll = D.rename(columns={CITY: 'pm25'}).dropna()
print('pollution: %d days, %s to %s'
      % (len(poll), poll['date'].min().date(), poll['date'].max().date()))

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
print('meteorology: %d days, %s to %s'
      % (len(met), met['date'].min().date(), met['date'].max().date()))

snd = pd.read_csv(os.path.join(DATA, 'sounding_daily.csv'),
                  parse_dates=['date'])
snd = snd[snd['city'] == 'shanghai'].drop(columns='city')

cal = pd.read_csv(os.path.join(DIR, 'ah_panel.csv'),
                  usecols=['date', 'a_code'], parse_dates=['date'],
                  dtype={'a_code': str})
cal['a_code'] = cal['a_code'].str.zfill(6)
trading = set(cal.loc[cal['a_code'].str.startswith('6'), 'date'])
print('panel trading days: %d' % len(trading))

d = met.merge(poll, on='date', how='inner').merge(snd, on='date', how='left')
d = d[d['pm25'] > 0].copy()
d['log_pm25'] = np.log(d['pm25'])
d['log_blh'] = np.log(d['blh'].clip(lower=1))
d['month'] = d['date'].dt.month
d['year'] = d['date'].dt.year
d['dow'] = d['date'].dt.dayofweek
d['season'] = d['month'].map({12: 'DJF', 1: 'DJF', 2: 'DJF',
                              3: 'MAM', 4: 'MAM', 5: 'MAM',
                              6: 'JJA', 7: 'JJA', 8: 'JJA',
                              9: 'SON', 10: 'SON', 11: 'SON'})
dt = d[d['date'].isin(trading)].copy()
print('merged: %d days, of which %d are trading days' % (len(d), len(dt)))

# ------------------------------------------------------------
# 2. Machinery
# ------------------------------------------------------------
CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']
SETS = {
    'ERA5      ': ['log_blh', 'u', 'v', 'surface_pressure'],
    'combined  ': ['log_blh', 'u', 'v', 'surface_pressure',
                   'dT_200_100', 'dT_1000_300'],
}


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


def run(sub, inst, fe_cols):
    s = sub.dropna(subset=inst + ['log_pm25'] + CONTROLS)
    if len(s) < 60:
        return None
    fe = pd.get_dummies(s[fe_cols].astype(str), drop_first=True).astype(float)
    W = pd.concat([s[CONTROLS].reset_index(drop=True),
                   fe.reset_index(drop=True)], axis=1)
    X = sm.add_constant(pd.concat([s[inst].reset_index(drop=True), W], axis=1))
    if np.linalg.matrix_rank(X.values) < X.shape[1]:
        return 'RANK'
    y = s['log_pm25'].reset_index(drop=True)
    full = sm.OLS(y, X).fit(cov_type='HC1')
    red = sm.OLS(y, X.drop(columns=inst)).fit(cov_type='HC1')
    pr2 = (full.rsquared - red.rsquared) / (1 - red.rsquared)
    return (effective_f(y, s[inst].reset_index(drop=True), W), pr2,
            int(full.nobs), full.params['log_blh'], full.tvalues['log_blh'])


def show(label, sub, fe_cols=('month', 'year', 'dow')):
    for name, inst in SETS.items():
        r = run(sub, inst, list(fe_cols))
        if not isinstance(r, tuple):
            print('%-14s %-11s %s' % (label, name, r or 'too few'))
            continue
        print('%-14s %-11s %6d %9.2f %12.4f %10.4f %8.2f'
              % (label, name, r[2], r[0], r[1], r[3], r[4]))


HDR = ('%-14s %-11s %6s %9s %12s %10s %8s'
       % ('sample', 'instruments', 'N', 'F_eff', 'partial R2', 'log_blh', 't'))

print('\n' + '=' * 82)
print('HEADLINE   Shanghai first stage, trading days, full year')
print('=' * 82)
print(HDR)
show('ALL', dt)
show('pre-2020', dt[dt['year'] <= 2019])
show('2020 only', dt[dt['year'] == 2020])

print('\n' + '=' * 82)
print('BY YEAR   (ERA5 set; is the relationship stable?)')
print('=' * 82)
print('%-14s %6s %9s %12s %10s %8s %9s'
      % ('year', 'N', 'F_eff', 'partial R2', 'log_blh', 't', 'mean PM'))
for y in sorted(dt['year'].unique()):
    sub = dt[dt['year'] == y]
    r = run(sub, SETS['ERA5      '], ['month', 'dow'])
    if not isinstance(r, tuple):
        print('%-14s %6d  %s' % (y, len(sub), r or 'too few'))
        continue
    print('%-14s %6d %9.2f %12.4f %10.4f %8.2f %9.1f'
          % (y, r[2], r[0], r[1], r[3], r[4], sub['pm25'].mean()))

print('\n' + '=' * 82)
print('BY SEASON   (does step 15 hold out of sample?)')
print('=' * 82)
print('%-14s %6s %9s %12s %10s %8s'
      % ('season', 'N', 'F_eff', 'partial R2', 'log_blh', 't'))
for s_ in ['DJF', 'MAM', 'JJA', 'SON']:
    sub = dt[dt['season'] == s_]
    r = run(sub, SETS['ERA5      '], ['month', 'year', 'dow'])
    if not isinstance(r, tuple):
        print('%-14s %6d  %s' % (s_, len(sub), r or 'too few'))
        continue
    print('%-14s %6d %9.2f %12.4f %10.4f %8.2f'
          % (s_, r[2], r[0], r[1], r[3], r[4]))

dt.to_csv(os.path.join(DATA, 'shanghai_first_stage_fullyear.csv'),
          index=False)
print('\nwritten: shanghai_first_stage_fullyear.csv  (%d rows)' % len(dt))

print("""
READING IT
  The ALL row is the headline. Step 13's best was F_eff 8.22 with
  partial R2 0.074 on 390 days.

  BY YEAR matters as much as the headline. Shanghai's mean PM2.5 fell by
  a third over this period, so if partial R2 drifts with it the pooled
  figure is an average of a changing relationship and year-by-year
  stability has to be argued in the pre-registration rather than assumed.

  log_blh must stay NEGATIVE everywhere. A positive year or season is
  the Hong Kong failure appearing in Shanghai and would need explaining
  before anything is filed.
""")
