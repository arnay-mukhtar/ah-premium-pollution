# Source: paper2_meteorology_build.ipynb, code cell 26 (final version of step 15 as run).
# ============================================================
# Step 15 — does the instrument survive outside winter?
#
# THE DECISION THIS SETTLES. Winter gives 390 trading days and F_eff
# 8.22. The full year gives roughly 1,500. F scales with N x partial R2,
# so four times the days at half the explanatory power still roughly
# doubles F. But the lidar literature reports the boundary layer /
# PM2.5 relationship is "nearly flat in summer", so the full year might
# buy days and lose more in strength than it gains.
#
# Two years of pilot data answers it. Pull it before committing to
# either the eleven-month winter extension or a seven-year full-year
# job, because doing the winter one first wastes it if the answer is
# full year.
#
# NOTE: uses Open-Meteo boundary layer height, not Copernicus. Paper 1
# documented Open-Meteo runs high in the upper tail but correlates 0.973
# in logs with CDS. That bias is common across seasons, so it cannot
# manufacture a seasonal pattern, which is all this pilot is asking.
# The real build would use CDS.
#
# Pollution and meteorology only. Nothing touches the A-H ratio.
# ============================================================

import io
import os
import time

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DATA = '/content/drive/MyDrive/bilateral/data'
DIR = '/content/drive/MyDrive/pollution-ah'
CACHE = os.path.join(DATA, 'quotsoft_shanghai_hourly.csv')

YEARS = [2017, 2018]          # widen if you want a sharper read

# quotsoft names cities in Chinese; the Zenodo deposit used English.
# If the two sources are ever combined, they need a name mapping.
CITY = '上海'                  # Shanghai
CITY_ALT = ['Shanghai', '上海市']
BASE = 'https://quotsoft.net/air/data'

# ------------------------------------------------------------
# 1. Probe ONE file before downloading hundreds
# ------------------------------------------------------------
print('probing the archive format ...')
r = requests.get('%s/china_cities_20180115.csv' % BASE, timeout=60)
print('  HTTP %s, %d bytes' % (r.status_code, len(r.content)))
if r.status_code != 200:
    raise SystemExit('Archive unreachable from Colab. Send me this output.')

probe = pd.read_csv(io.BytesIO(r.content))
print('  %d columns, %d rows' % (probe.shape[1], len(probe)))
types = sorted(probe['type'].unique())
print('  all %d types: %s' % (len(types), types))
print('  hours: %d distinct' % probe['hour'].nunique())

for cand in [CITY] + CITY_ALT:
    if cand in probe.columns:
        CITY = cand
        break
else:
    hits = [c for c in probe.columns if '海' in str(c)]
    print('  Shanghai column NOT FOUND. Columns containing 海: %s' % hits)
    print('  first 30 columns: %s' % list(probe.columns[:30]))
    raise SystemExit('Send this output and I will set the name.')

print('  using city column: %r' % CITY)
if 'PM2.5' not in set(types):
    raise SystemExit('No PM2.5 type. Send the type list above.')
print('  sample values: %s'
      % probe.loc[probe['type'] == 'PM2.5', CITY].head(5).tolist())
print('  format OK\n')

# ------------------------------------------------------------
# 2. Pull the pilot years
# ------------------------------------------------------------
if os.path.exists(CACHE):
    hourly = pd.read_csv(CACHE, parse_dates=['ts'])
    print('cached: %d hourly rows' % len(hourly))
else:
    days = pd.date_range('%d-01-01' % YEARS[0], '%d-12-31' % YEARS[-1],
                         freq='D')
    rows, bad = [], 0
    for i, day in enumerate(days, 1):
        try:
            rr = requests.get('%s/china_cities_%s.csv'
                              % (BASE, day.strftime('%Y%m%d')), timeout=30)
            if rr.status_code != 200:
                bad += 1
                continue
            t = pd.read_csv(io.BytesIO(rr.content),
                            usecols=['date', 'hour', 'type', CITY])
            t = t[t['type'] == 'PM2.5']
            rows.append(t[['date', 'hour', CITY]])
        except Exception:
            bad += 1
        if i % 100 == 0:
            print('  %d/%d days, %d unavailable' % (i, len(days), bad))
        time.sleep(0.15)

    hourly = pd.concat(rows, ignore_index=True)
    hourly['ts'] = (pd.to_datetime(hourly['date'], format='%Y%m%d')
                    + pd.to_timedelta(hourly['hour'], unit='h'))
    hourly = hourly[['ts', CITY]].rename(columns={CITY: 'pm25'})
    hourly['pm25'] = pd.to_numeric(hourly['pm25'], errors='coerce')
    hourly = hourly.drop_duplicates('ts').sort_values('ts')
    hourly.to_csv(CACHE, index=False)
    print('\n%d hourly rows, %d days unavailable -> cached' % (len(hourly), bad))

g = hourly.groupby(hourly['ts'].dt.normalize())['pm25']
poll = g.mean().where(g.count() >= 8).rename('pm25').reset_index()
poll = poll.rename(columns={'ts': 'date'}).dropna()
print('daily PM2.5: %d days' % len(poll))
print(poll.groupby(poll['date'].dt.month)['pm25'].mean().round(1).to_string())

# ------------------------------------------------------------
# 3. Meteorology, Open-Meteo, same years
# ------------------------------------------------------------
FIELDS = ['boundary_layer_height', 'wind_speed_10m', 'wind_direction_10m',
          'surface_pressure', 'temperature_2m', 'precipitation',
          'cloud_cover', 'shortwave_radiation']
rq = requests.get('https://archive-api.open-meteo.com/v1/archive', params={
    'latitude': 31.2304, 'longitude': 121.4737,
    'start_date': '%d-01-01' % YEARS[0], 'end_date': '%d-12-31' % YEARS[-1],
    'hourly': ','.join(FIELDS), 'timezone': 'Asia/Shanghai'}, timeout=300)
rq.raise_for_status()
m = pd.DataFrame(rq.json()['hourly'])
m['time'] = pd.to_datetime(m['time'])
rad = np.deg2rad(m['wind_direction_10m'])
m['u'] = -m['wind_speed_10m'] * np.sin(rad)
m['v'] = -m['wind_speed_10m'] * np.cos(rad)
m['date'] = m['time'].dt.normalize()
print('\nmeteorology: %d hours, BLH missing %d'
      % (len(m), int(m['boundary_layer_height'].isna().sum())))

AGG = ['boundary_layer_height', 'wind_speed_10m', 'surface_pressure',
       'temperature_2m', 'precipitation', 'cloud_cover',
       'shortwave_radiation', 'u', 'v']
gg = m.groupby('date')
met = gg[AGG].mean().reset_index()
met['n_hours'] = gg.size().values
met = met[met['n_hours'] >= 8]

d = met.merge(poll, on='date', how='inner')
d = d[d['pm25'] > 0].copy()
d['log_pm25'] = np.log(d['pm25'])
d['log_blh'] = np.log(d['boundary_layer_height'].clip(lower=1))
d['month'] = d['date'].dt.month
d['year'] = d['date'].dt.year
d['dow'] = d['date'].dt.dayofweek
d['season'] = d['month'].map({12: 'DJF', 1: 'DJF', 2: 'DJF',
                              3: 'MAM', 4: 'MAM', 5: 'MAM',
                              6: 'JJA', 7: 'JJA', 8: 'JJA',
                              9: 'SON', 10: 'SON', 11: 'SON'})
print('merged: %d days' % len(d))

# ------------------------------------------------------------
# 4. First stage, by season and all year
# ------------------------------------------------------------
INST = ['log_blh', 'u', 'v', 'surface_pressure']
CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']


def effective_f(y, Z, W):
    Wc = sm.add_constant(W, has_constant='add')
    P = Wc.values @ np.linalg.pinv(Wc.values)
    yt = y.values - P @ y.values
    Zt = Z.values - P @ Z.values
    ZtZ = Zt.T @ Zt
    pi = np.linalg.solve(ZtZ, Zt.T @ yt)
    e = yt - Zt @ pi
    n, k = Zt.shape
    bread = np.linalg.inv(ZtZ)
    V = (bread @ (Zt.T @ (Zt * (e ** 2)[:, None])) @ bread
         * (n / max(n - k - Wc.shape[1], 1)))
    return float(pi @ ZtZ @ pi / np.trace(ZtZ @ V))


def fs(sub, fe_cols):
    sub = sub.dropna(subset=INST + ['log_pm25'] + CONTROLS).copy()
    if len(sub) < 60:
        return None
    fe = pd.get_dummies(sub[fe_cols].astype(str),
                        drop_first=True).astype(float)
    W = pd.concat([sub[CONTROLS].reset_index(drop=True),
                   fe.reset_index(drop=True)], axis=1)
    X = sm.add_constant(pd.concat([sub[INST].reset_index(drop=True), W],
                                  axis=1))
    y = sub['log_pm25'].reset_index(drop=True)
    full = sm.OLS(y, X).fit(cov_type='HC1')
    red = sm.OLS(y, X.drop(columns=INST)).fit(cov_type='HC1')
    pr2 = (full.rsquared - red.rsquared) / (1 - red.rsquared)
    return (effective_f(y, sub[INST].reset_index(drop=True), W), pr2,
            int(full.nobs), full.params['log_blh'], full.tvalues['log_blh'])


print('\n' + '=' * 74)
print('FIRST STAGE BY SEASON   (ERA5-style set, Open-Meteo BLH)')
print('=' * 74)
print('%-10s %6s %9s %12s %10s %8s' %
      ('season', 'N', 'F_eff', 'partial R2', 'log_blh', 't'))
for s in ['DJF', 'MAM', 'JJA', 'SON']:
    r = fs(d[d['season'] == s], ['month', 'year', 'dow'])
    if r is None:
        print('%-10s %6s' % (s, 'n/a'))
        continue
    print('%-10s %6d %9.2f %12.4f %10.4f %8.2f'
          % (s, r[2], r[0], r[1], r[3], r[4]))

r = fs(d, ['month', 'year', 'dow'])
print('%-10s %6d %9.2f %12.4f %10.4f %8.2f'
      % ('ALL YEAR', r[2], r[0], r[1], r[3], r[4]))

# Trading-day version, the number that actually transfers
cal = pd.read_csv(os.path.join(DIR, 'ah_panel.csv'),
                  usecols=['date', 'a_code'], parse_dates=['date'],
                  dtype={'a_code': str})
cal['a_code'] = cal['a_code'].str.zfill(6)
trading = set(cal.loc[cal['a_code'].str.startswith('6'), 'date'])
rt = fs(d[d['date'].isin(trading)], ['month', 'year', 'dow'])
print('%-10s %6d %9.2f %12.4f %10.4f %8.2f'
      % ('ALL TRADE', rt[2], rt[0], rt[1], rt[3], rt[4]))

print("""
WHAT IT DECIDES
  Summer partial R2 above roughly half the winter figure, and ALL YEAR
  at or above winter: go full year. Roughly 1,500 trading days over
  2013-2020 instead of 390, and F_eff should land somewhere near 15-20.
  That is the difference between an instrument you apologise for and one
  you defend.

  Summer near zero and ALL YEAR below winter: the seasonal restriction
  is doing real work. Stay with winter, pull November and March only,
  and accept F_eff near 12.7 with Anderson-Rubin inference.

  Either way the log_blh coefficient must stay NEGATIVE in every season.
  A sign flip in any season is the Hong Kong problem appearing in
  Shanghai and would need explaining before anything is pre-registered.
""")
