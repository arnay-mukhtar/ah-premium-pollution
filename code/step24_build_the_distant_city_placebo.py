# Source: paper2_meteorology_build.ipynb, code cell 39 (final version of step 24 as run).
# ============================================================
# Step 24 — build the distant-city placebo
#
# The registered falsification test needs pollution in a city far from
# Shanghai, instrumented by ITS OWN dispersion conditions, entered
# alongside Shanghai pollution. If investors respond to Shanghai air
# because they breathe it, a distant city carries no coefficient.
#
# A placebo is only informative if its OWN first stage is strong. A weak
# instrument fails to reject by construction, which proves nothing. So
# this builds the Urumqi series and tests its first stage before the
# placebo is relied on.
#
# Urumqi (43.8256 N, 87.6168 E) is chosen a priori on geography:
# ~3,300 km from Shanghai, a separate airshed behind the Tianshan with
# no synoptic connection, and no financial centre.
#
# Also finds the three cities least correlated with Shanghai, so the
# registered secondary version does not rest on one hand-picked city.
#
# Pollution and meteorology only. Nothing touches the A-H ratio.
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
CACHE = os.path.join(DATA, 'urumqi_om_hourly.csv')

PLACEBO = '乌鲁木齐'          # Urumqi
LAT, LON = 43.8256, 87.6168
HOME = '上海'                # Shanghai
START, END = '2014-05-13', '2019-12-31'

# ------------------------------------------------------------
# 1. Pollution: is the city in the archive?
# ------------------------------------------------------------
D = pd.read_csv(os.path.join(DATA, 'quotsoft_cities_daily.csv'),
                parse_dates=['date'])
cols = [c for c in D.columns if c != 'date']
print('cities in archive: %d' % len(cols))

if PLACEBO not in D.columns:
    print('%r NOT FOUND. Cities containing 乌 or 鲁: %s'
          % (PLACEBO, [c for c in cols if '乌' in c or '鲁' in c][:20]))
    raise SystemExit('Send this and I will pick another city.')

cov = D[PLACEBO].notna().mean()
print('%s: %d days, coverage %.1f%%, mean %.1f'
      % (PLACEBO, int(D[PLACEBO].notna().sum()), 100 * cov,
         D[PLACEBO].mean()))

# ------------------------------------------------------------
# 2. How independent is it of Shanghai?
# ------------------------------------------------------------
both = D[['date', HOME, PLACEBO]].dropna()
r_lvl = both[HOME].corr(both[PLACEBO])
r_log = np.log(both[HOME]).corr(np.log(both[PLACEBO]))
print('\ncorrelation with Shanghai: %.4f in levels, %.4f in logs (n %d)'
      % (r_lvl, r_log, len(both)))

keep = [c for c in cols if D[c].notna().mean() >= 0.90 and c != HOME]
cc = {}
for c in keep:
    s = D[[HOME, c]].dropna()
    if len(s) > 500:
        cc[c] = np.log(s[HOME]).corr(np.log(s[c]))
cc = pd.Series(cc).sort_values()
print('\ncities with >=90%% coverage: %d' % len(keep))
print('three least correlated with Shanghai (registered secondary set):')
print(cc.head(3).round(4).to_string())
print('\nUrumqi rank: %d of %d (correlation %.4f)'
      % (list(cc.index).index(PLACEBO) + 1 if PLACEBO in cc.index else -1,
         len(cc), cc.get(PLACEBO, np.nan)))

# ------------------------------------------------------------
# 3. Meteorology for the placebo city
# ------------------------------------------------------------
FIELDS = ['boundary_layer_height', 'wind_speed_10m', 'wind_direction_10m',
          'surface_pressure', 'temperature_2m', 'precipitation',
          'cloud_cover', 'shortwave_radiation']
if os.path.exists(CACHE):
    om = pd.read_csv(CACHE, parse_dates=['time'])
    print('\nmeteorology cached: %d hours' % len(om))
else:
    parts = []
    for y0 in range(2014, 2021, 2):
        y1 = min(y0 + 1, 2020)
        r = requests.get('https://archive-api.open-meteo.com/v1/archive',
                         params={'latitude': LAT, 'longitude': LON,
                                 'start_date': '%d-01-01' % y0,
                                 'end_date': '%d-12-31' % y1,
                                 'hourly': ','.join(FIELDS),
                                 'timezone': 'Asia/Shanghai'}, timeout=600)
        r.raise_for_status()
        parts.append(pd.DataFrame(r.json()['hourly']))
        print('  %d-%d: %d hours' % (y0, y1, len(parts[-1])))
        time.sleep(2)
    om = pd.concat(parts, ignore_index=True).drop_duplicates('time')
    om['time'] = pd.to_datetime(om['time'])
    om.to_csv(CACHE, index=False)
    print('\ncached -> %s' % CACHE)

# Urumqi runs on Beijing time officially; the archive is queried in the
# same zone as Shanghai so both cities' days are aligned to the trading
# calendar, which is what the placebo requires.
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

# ------------------------------------------------------------
# 4. Does the placebo city have a usable first stage?
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


def first_stage(d, label):
    s = d.dropna(subset=INST + ['log_pm25'] + CONTROLS)
    if len(s) < 100:
        print('%-22s too few days (%d)' % (label, len(s)))
        return
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
    print('%-22s %6d %9.2f %12.4f %10.4f %8.2f'
          % (label, int(full.nobs),
             effective_f(y, s[INST].reset_index(drop=True), W), pr2,
             full.params['log_blh'], full.tvalues['log_blh']))


d = met.merge(D[['date', PLACEBO]].rename(columns={PLACEBO: 'pm25'}),
              on='date', how='inner')
d = d[(d['date'] >= START) & (d['date'] <= END)]
d = d[d['pm25'] > 0].copy()
d['log_pm25'] = np.log(d['pm25'])
d['log_blh'] = np.log(d['blh'].clip(lower=1))
d['month'] = d['date'].dt.month
d['year'] = d['date'].dt.year
d['dow'] = d['date'].dt.dayofweek

print('\n' + '=' * 72)
print('PLACEBO CITY FIRST STAGE   (%s)' % PLACEBO)
print('=' * 72)
print('%-22s %6s %9s %12s %10s %8s'
      % ('sample', 'N', 'F_eff', 'partial R2', 'log_blh', 't'))
first_stage(d, 'all days')
first_stage(d[d['date'].isin(trading)], 'trading days')

out = os.path.join(DATA, 'placebo_city_panel.csv')
d.to_csv(out, index=False)
print('\nwritten: %s (%d rows)' % (out, len(d)))

print("""
WHAT DECIDES WHETHER THE PLACEBO IS USABLE
  F_eff comfortably above the Shanghai figure's order, log_blh negative
  and significant: the placebo is informative, because a null there
  means the distant city genuinely does not matter rather than that its
  instrument was too weak to say.

  F_eff in single figures or a wrong-signed coefficient: the placebo
  proves nothing and must not be reported as though it does. Pick
  another city from the least-correlated list above and rerun.

  Correlation with Shanghai should be low. If it is above about 0.5,
  the two cities share weather regimes and the placebo cannot separate
  a local channel from a national one.
""")
