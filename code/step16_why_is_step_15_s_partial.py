# Source: paper2_meteorology_build.ipynb, code cell 27 (final version of step 16 as run).
# ============================================================
# Step 16 — why is step 15's partial R2 3.6x step 13's?
#
# Step 13 (DJF trading days, 2013-2020): partial R2 0.074
# Step 15 (DJF, 2017-2018):              partial R2 0.265
#
# Same four instruments, same controls, same FE structure, same city.
# Restricting to winter does not reconcile them, so it is not seasonal.
# The weather controls and u/v/pressure come from Open-Meteo at the same
# point in both, so only three candidates remain:
#
#   A. YEARS        2013-2020 vs 2017-2018. Shanghai PM2.5 fell ~40% over
#                   the period, so meteorology becomes a larger share of
#                   the remaining variance. Would be a finding, not a bug.
#   B. POLLUTION    Zenodo station average vs quotsoft MEP city value.
#   C. BOUNDARY     CDS averaged over a 1.2 x 1.1 degree box vs
#      LAYER        Open-Meteo at a single point. Averaging over 130 km
#                   may be smoothing away the signal.
#
# This isolates all three, and pulls the full-year Open-Meteo series the
# full-year build needs regardless.
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
OM_CACHE = os.path.join(DATA, 'shanghai_om_fullyear_hourly.csv')

LAT, LON = 31.2304, 121.4737
FIELDS = ['boundary_layer_height', 'wind_speed_10m', 'wind_direction_10m',
          'surface_pressure', 'temperature_2m', 'precipitation',
          'cloud_cover', 'shortwave_radiation']

# ------------------------------------------------------------
# 1. Open-Meteo, full years 2013-2020, including BLH
# ------------------------------------------------------------
if os.path.exists(OM_CACHE):
    om = pd.read_csv(OM_CACHE, parse_dates=['time'])
    print('Open-Meteo cached: %d hours' % len(om))
else:
    parts = []
    for y0 in range(2013, 2021, 2):
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
    om.to_csv(OM_CACHE, index=False)
    print('cached -> %s  (%d hours)' % (OM_CACHE, len(om)))

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
met = met[met['n_hours'] >= 8]
met = met.rename(columns={'boundary_layer_height': 'blh_om'})
print('Open-Meteo daily: %d days' % len(met))

# ------------------------------------------------------------
# 2. CDS boundary layer, daily, same 24h rule
# ------------------------------------------------------------
cds = pd.read_csv(os.path.join(DATA, 'shanghai_blh_hourly.csv'),
                  parse_dates=['date'])
gc = cds.groupby('date')
cdsd = gc[['blh_land_mean', 'blh_all_mean', 'blh_point']].mean().reset_index()
cdsd['n_cds'] = gc.size().values
cdsd = cdsd[cdsd['n_cds'] >= 8]
print('CDS daily: %d days (winter months only, by construction)' % len(cdsd))

# ------------------------------------------------------------
# 3. Two pollution series
# ------------------------------------------------------------
zen = pd.read_csv(os.path.join(DATA, 'cn_pm25_daily_24h.csv'),
                  index_col='date', parse_dates=True)['Shanghai']
zen = zen.rename('pm_zenodo').reset_index()

qs = pd.read_csv(os.path.join(DATA, 'quotsoft_shanghai_hourly.csv'),
                 parse_dates=['ts'])
gq = qs.groupby(qs['ts'].dt.normalize())['pm25']
qsd = gq.mean().where(gq.count() >= 8).rename('pm_quotsoft').reset_index()
qsd = qsd.rename(columns={'ts': 'date'}).dropna()
print('Zenodo %d days, quotsoft %d days' % (len(zen), len(qsd)))

both = zen.merge(qsd, on='date', how='inner').dropna()
print('\n' + '=' * 70)
print('B. ARE THE TWO POLLUTION SERIES THE SAME THING?')
print('=' * 70)
print('overlapping days: %d' % len(both))
if len(both):
    print('   correlation        %.4f' % both['pm_zenodo'].corr(
        both['pm_quotsoft']))
    print('   mean   zenodo %.1f   quotsoft %.1f'
          % (both['pm_zenodo'].mean(), both['pm_quotsoft'].mean()))
    print('   sd     zenodo %.1f   quotsoft %.1f'
          % (both['pm_zenodo'].std(), both['pm_quotsoft'].std()))
    d_ = (both['pm_zenodo'] - both['pm_quotsoft'])
    print('   difference: mean %.2f   sd %.2f   max abs %.1f'
          % (d_.mean(), d_.std(), d_.abs().max()))
    print('   (correlation above 0.98 means the source is NOT the cause)')

# ------------------------------------------------------------
# 4. The specification, run every way
# ------------------------------------------------------------
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


def spec(d, pm_col, blh_col):
    inst = ['log_blh', 'u', 'v', 'surface_pressure']
    s = d.dropna(subset=[pm_col, blh_col] + CONTROLS + ['u', 'v',
                                                        'surface_pressure'])
    s = s[s[pm_col] > 0].copy()
    if len(s) < 60:
        return None
    s['log_pm25'] = np.log(s[pm_col])
    s['log_blh'] = np.log(s[blh_col].clip(lower=1))
    s['month'] = s['date'].dt.month
    s['year'] = s['date'].dt.year
    s['dow'] = s['date'].dt.dayofweek
    fe = pd.get_dummies(s[['month', 'year', 'dow']].astype(str),
                        drop_first=True).astype(float)
    W = pd.concat([s[CONTROLS].reset_index(drop=True),
                   fe.reset_index(drop=True)], axis=1)
    X = sm.add_constant(pd.concat([s[inst].reset_index(drop=True), W], axis=1))
    y = s['log_pm25'].reset_index(drop=True)
    full = sm.OLS(y, X).fit(cov_type='HC1')
    red = sm.OLS(y, X.drop(columns=inst)).fit(cov_type='HC1')
    pr2 = (full.rsquared - red.rsquared) / (1 - red.rsquared)
    return (effective_f(y, s[inst].reset_index(drop=True), W), pr2,
            int(full.nobs), full.params['log_blh'], full.tvalues['log_blh'])


base = met.merge(cdsd, on='date', how='left')
base = base.merge(zen, on='date', how='left').merge(qsd, on='date', how='left')

cal = pd.read_csv(os.path.join(DIR, 'ah_panel.csv'),
                  usecols=['date', 'a_code'], parse_dates=['date'],
                  dtype={'a_code': str})
cal['a_code'] = cal['a_code'].str.zfill(6)
trading = set(cal.loc[cal['a_code'].str.startswith('6'), 'date'])

DJF1718 = base[(base['date'] >= '2017-01-01') & (base['date'] <= '2018-12-31')
               & (base['date'].dt.month.isin([12, 1, 2]))]

print('\n' + '=' * 70)
print('C. SAME DAYS, EVERY COMBINATION   (DJF 2017-2018)')
print('=' * 70)
print('%-12s %-14s %6s %9s %12s %9s %7s'
      % ('pollution', 'boundary layer', 'N', 'F_eff', 'partial R2',
         'log_blh', 't'))
for pm, pml in [('pm_zenodo', 'Zenodo'), ('pm_quotsoft', 'quotsoft')]:
    for bl, bll in [('blh_land_mean', 'CDS land box'),
                    ('blh_point', 'CDS point'),
                    ('blh_om', 'Open-Meteo pt')]:
        r = spec(DJF1718, pm, bl)
        if r is None:
            print('%-12s %-14s %6s' % (pml, bll, 'n/a'))
            continue
        print('%-12s %-14s %6d %9.2f %12.4f %9.4f %7.2f'
              % (pml, bll, r[2], r[0], r[1], r[3], r[4]))

# ------------------------------------------------------------
# 5. Is it the years?
# ------------------------------------------------------------
print('\n' + '=' * 70)
print('A. THE SAME SPECIFICATION, WINTER BY WINTER   (Zenodo + CDS box)')
print('=' * 70)
print('%-12s %6s %9s %12s %9s %7s %10s'
      % ('winter', 'N', 'F_eff', 'partial R2', 'log_blh', 't', 'mean PM'))
w = base.copy()
w['winter'] = np.where(w['date'].dt.month == 12, w['date'].dt.year,
                       w['date'].dt.year - 1)
for yr in range(2013, 2020):
    s = w[(w['winter'] == yr) & (w['date'].dt.month.isin([12, 1, 2]))]
    r = spec(s, 'pm_zenodo', 'blh_land_mean')
    mp = s['pm_zenodo'].mean()
    if r is None:
        print('%-12s %6s %38s %10.1f'
              % ('%d-%d' % (yr, yr + 1), len(s), 'too few days', mp)
              if mp == mp else '%-12s too few' % yr)
        continue
    print('%-12s %6d %9.2f %12.4f %9.4f %7.2f %10.1f'
          % ('%d-%d' % (yr, yr + 1), r[2], r[0], r[1], r[3], r[4], mp))

print("""
READING IT
  Section B: correlation above 0.98 rules the pollution source out.
  Section C: if the three boundary layer rows differ sharply on the SAME
    days and SAME pollution, the 1.2x1.1 degree box averaging is the
    cause, and the original spatial-support choice was wrong for the
    instruments even if right for the treatment.
  Section A: if partial R2 climbs steadily from 2013-14 to 2018-19 as
    mean PM falls, the years explain it, which is a finding rather than
    a bug: a cleaner city has a larger meteorological share of variance.

  These are not exclusive. Both can be true, and the row-by-row numbers
  will say how much each contributes.
""")
