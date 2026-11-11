# Source: paper2_meteorology_build.ipynb, code cell 41 (final version of step 25 as run).
# ============================================================
# Step 25 — Beijing, and the three-city picture
#
# Shanghai  exchange + large investor population
# Beijing   large investor population, NO exchange
# Urumqi    neither
#
# Entered together, these separate three readings of the result: the
# exchange's own city, mainland investor exposure generally, and
# national weather patterns.
#
# Two things have to hold for the test to work, and both are checked
# here rather than assumed:
#   1. Beijing needs its OWN strong first stage, or a null there is a
#      null by construction.
#   2. Beijing and Shanghai pollution must not be so correlated that a
#      joint regression cannot separate them. Above about 0.7 in logs
#      the test is uninformative whatever the coefficients say.
#
# Pollution and meteorology only.
# ============================================================

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
START, END = '2014-05-13', '2019-12-31'

CITIES = {
    '上海': {'lat': 31.2304, 'lon': 121.4737, 'tag': 'Shanghai',
             'cache': 'shanghai_om_fullyear_hourly.csv'},
    '北京': {'lat': 39.9042, 'lon': 116.4074, 'tag': 'Beijing',
             'cache': 'beijing_om_hourly.csv'},
    '乌鲁木齐': {'lat': 43.8256, 'lon': 87.6168, 'tag': 'Urumqi',
                 'cache': 'urumqi_om_hourly.csv'},
}

FIELDS = ['boundary_layer_height', 'wind_speed_10m', 'wind_direction_10m',
          'surface_pressure', 'temperature_2m', 'precipitation',
          'cloud_cover', 'shortwave_radiation']

D = pd.read_csv(os.path.join(DATA, 'quotsoft_cities_daily.csv'),
                parse_dates=['date'])
for c in CITIES:
    if c not in D.columns:
        raise SystemExit('%r not in the archive.' % c)
    print('%-10s %s  coverage %.1f%%  mean %.1f'
          % (CITIES[c]['tag'], c, 100 * D[c].notna().mean(), D[c].mean()))

# ------------------------------------------------------------
# 1. Can the three be told apart?
# ------------------------------------------------------------
print('\n' + '=' * 72)
print('PAIRWISE POLLUTION CORRELATION (logs)')
print('=' * 72)
names = list(CITIES)
print('   %-12s %s' % ('', ''.join('%12s' % CITIES[c]['tag']
                                   for c in names)))
for a in names:
    row = []
    for b in names:
        if a == b:
            # D[[a, a]] gives duplicate columns and s[a] is then a frame
            row.append(1.0)
            continue
        s = D[[a, b]].dropna()
        row.append(float(np.log(s[a]).corr(np.log(s[b]))) if len(s) > 500
                   else np.nan)
    print('   %-12s %s' % (CITIES[a]['tag'],
                           ''.join('%12.4f' % v for v in row)))
print('\n   Shanghai against Beijing is the number that matters. Above')
print('   about 0.7 and a joint regression cannot separate them,')
print('   whatever the coefficients come out as.')

# ------------------------------------------------------------
# 2. Meteorology for each
# ------------------------------------------------------------
def meteorology(city):
    c = CITIES[city]
    path = os.path.join(DATA, c['cache'])
    if os.path.exists(path):
        om = pd.read_csv(path, parse_dates=['time'])
    else:
        parts = []
        for y0 in range(2014, 2021, 2):
            y1 = min(y0 + 1, 2020)
            r = requests.get(
                'https://archive-api.open-meteo.com/v1/archive',
                params={'latitude': c['lat'], 'longitude': c['lon'],
                        'start_date': '%d-01-01' % y0,
                        'end_date': '%d-12-31' % y1,
                        'hourly': ','.join(FIELDS),
                        'timezone': 'Asia/Shanghai'}, timeout=600)
            r.raise_for_status()
            parts.append(pd.DataFrame(r.json()['hourly']))
            print('   %s %d-%d: %d hours' % (c['tag'], y0, y1,
                                             len(parts[-1])))
            time.sleep(2)
        om = pd.concat(parts, ignore_index=True).drop_duplicates('time')
        om['time'] = pd.to_datetime(om['time'])
        om.to_csv(path, index=False)
    rad = np.deg2rad(om['wind_direction_10m'])
    om['u'] = -om['wind_speed_10m'] * np.sin(rad)
    om['v'] = -om['wind_speed_10m'] * np.cos(rad)
    om['date'] = om['time'].dt.normalize()
    AGG = ['boundary_layer_height', 'wind_speed_10m', 'surface_pressure',
           'temperature_2m', 'precipitation', 'cloud_cover',
           'shortwave_radiation', 'u', 'v']
    g = om.groupby('date')
    m = g[AGG].mean().reset_index()
    m['n_hours'] = g.size().values
    return m[m['n_hours'] >= 8].rename(
        columns={'boundary_layer_height': 'blh'})


print('\nbuilding meteorology ...')
met = {c: meteorology(c) for c in CITIES}

# ------------------------------------------------------------
# 3. First stage for each, on the trading calendar
# ------------------------------------------------------------
cal = pd.read_csv(os.path.join(DIR, 'ah_panel.csv'),
                  usecols=['date', 'a_code'], parse_dates=['date'],
                  dtype={'a_code': str})
cal['a_code'] = cal['a_code'].str.zfill(6)
trading = set(cal.loc[cal['a_code'].str.startswith('6'), 'date'])

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


print('\n' + '=' * 72)
print('FIRST STAGE BY CITY, trading days, %s to %s' % (START, END))
print('=' * 72)
print('%-10s %6s %9s %12s %10s %8s'
      % ('city', 'N', 'F_eff', 'partial R2', 'log_blh', 't'))

built = {}
for city in CITIES:
    d = met[city].merge(
        D[['date', city]].rename(columns={city: 'pm25'}), on='date',
        how='inner')
    d = d[(d['date'] >= START) & (d['date'] <= END)]
    d = d[d['pm25'] > 0].copy()
    d['log_pm25'] = np.log(d['pm25'])
    d['log_blh'] = np.log(d['blh'].clip(lower=1))
    d['month'] = d['date'].dt.month
    d['year'] = d['date'].dt.year
    d['dow'] = d['date'].dt.dayofweek
    d = d[d['date'].isin(trading)]
    built[city] = d

    s = d.dropna(subset=INST + ['log_pm25'] + CONTROLS)
    fe = pd.get_dummies(s[['month', 'year', 'dow']].astype(str),
                        drop_first=True).astype(float)
    W = pd.concat([s[CONTROLS].reset_index(drop=True),
                   fe.reset_index(drop=True)], axis=1)
    X = sm.add_constant(pd.concat([s[INST].reset_index(drop=True), W],
                                  axis=1))
    y = s['log_pm25'].reset_index(drop=True)
    full = sm.OLS(y, X).fit(cov_type='HC1')
    red = sm.OLS(y, X.drop(columns=INST)).fit(cov_type='HC1')
    pr2 = (full.rsquared - red.rsquared) / (1 - red.rsquared)
    print('%-10s %6d %9.2f %12.4f %10.4f %8.2f'
          % (CITIES[city]['tag'], int(full.nobs),
             effective_f(y, s[INST].reset_index(drop=True), W), pr2,
             full.params['log_blh'], full.tvalues['log_blh']))

# ------------------------------------------------------------
# 4. Save a single three-city panel for the registered test
# ------------------------------------------------------------
out = None
for city in CITIES:
    tag = CITIES[city]['tag'].lower()
    k = built[city][['date', 'log_pm25'] + INST + CONTROLS].copy()
    k.columns = ['date'] + ['%s_%s' % (tag, c) for c in k.columns[1:]]
    out = k if out is None else out.merge(k, on='date', how='inner')

p = os.path.join(DATA, 'three_city_panel.csv')
out.to_csv(p, index=False)
print('\nwritten: %s  (%d days, %d columns)' % (p, len(out),
                                                out.shape[1]))

print("""
WHAT TO CHECK
  Beijing's own F_eff must be strong and its log_blh negative, or a null
  for Beijing means nothing.
  Shanghai-Beijing pollution correlation below about 0.7, or the joint
  regression cannot tell them apart.
  The day count on the merged panel should stay near 1,300. A large drop
  means one city's coverage is thinner than the others and the test
  would be run on a different sample than the primary result.
""")
