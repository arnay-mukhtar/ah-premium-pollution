# Source: paper2_meteorology_build.ipynb, code cell 76 (final version of step 36b as run).
# ============================================================
# Step 36b — what the screen actually shows
#
# Step 36 returned three things and one bug.
#
#   Geography is strongly predictive: latitude t -7.99, distance to
#   coast t +6.37, R2 0.36 over 200 cities.
#
#   Only 4% of cities invert. Receptor-like behaviour is RARE.
#
#   The fitted geography model does NOT predict Hong Kong's failure.
#   Its southern latitude pushes the coefficient weaker, its coastal
#   position pushes it stronger, and the two cancel at about the median
#   city. Section 6 claims receptor behaviour is a general, screenable
#   property; on this evidence Hong Kong is an outlier instead.
#
#   THE BUG: MIN_DAYS = 300 was applied to the seasonal subsets too, and
#   two winters is ~180 days, so every seasonal fit was silently
#   rejected and section 3 printed NaN. The winter test, which is the
#   sharpest check on the mechanism, never ran.
#
# This file settles four questions before Section 6 is rewritten.
#
#   1. The winter test, with a sane threshold.
#   2. WHICH cities invert, by name and position. If they cluster in the
#      southern coastal belt near Hong Kong, the receptor story survives
#      in a narrower form. If they are scattered, it does not.
#   3. Where Hong Kong actually sits, using EPD pollution and its own
#      meteorology, so it can be placed in the distribution rather than
#      discussed outside it.
#   4. Hong Kong's residual against the fitted geography model: is it an
#      outlier, and by how much?
#
# Whatever comes back is what Section 6 will say.
# ============================================================

import os
import time

import numpy as np
import pandas as pd
import requests

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DATA = '/content/drive/MyDrive/bilateral/data'
CACHE2 = os.path.join(DATA, 'city_meteo_2y')
WIN0, WIN1 = '2017-01-01', '2018-12-31'
MIN_DAYS = 300
MIN_DAYS_SEASON = 120          # two winters is ~180 days; 300 rejected all

INST = ['log_blh', 'u', 'v', 'surface_pressure']
CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']
FIELDS = ['boundary_layer_height', 'wind_speed_10m', 'wind_direction_10m',
          'surface_pressure', 'temperature_2m', 'precipitation',
          'cloud_cover', 'shortwave_radiation']

COAST = np.array([
    (39.9, 124.4), (39.0, 121.7), (38.9, 117.7), (37.8, 120.7),
    (36.1, 120.4), (34.6, 119.2), (32.0, 121.7), (31.2, 121.5),
    (30.0, 122.1), (28.0, 121.1), (26.1, 119.3), (24.5, 118.1),
    (23.4, 116.7), (22.5, 114.1), (21.6, 111.0), (21.2, 110.4),
    (21.5, 109.1)])


def km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = (np.sin(dp / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def coast_km(lat, lon):
    return float(np.min(km(lat, lon, COAST[:, 0], COAST[:, 1])))


def first_stage(d, min_days=MIN_DAYS):
    s = d.dropna(subset=INST + CONTROLS + ['log_pm25'])
    if len(s) < min_days:
        return None
    cols = [c for c in ['month', 'dow'] if s[c].nunique() > 1]
    X = [np.ones(len(s)), s[INST].values, s[CONTROLS].values]
    if cols:
        X.append(pd.get_dummies(s[cols].astype(str),
                                drop_first=True).astype(float).values)
    X = np.column_stack(X)
    y = s['log_pm25'].values
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    n, k = X.shape
    if n - k < 20:
        return None
    XtXi = np.linalg.pinv(X.T @ X)
    V = XtXi @ (X.T @ (X * (e ** 2)[:, None])) @ XtXi * (n / (n - k))
    se = float(np.sqrt(max(V[1, 1], 1e-300)))
    return float(b[1]), float(b[1] / se), int(n)


def prep(d):
    d = d[d['pm25'] > 0].copy()
    d['log_pm25'] = np.log(d['pm25'])
    d['log_blh'] = np.log(d['boundary_layer_height'].clip(lower=1))
    d['month'] = d['date'].dt.month
    d['dow'] = d['date'].dt.dayofweek
    return d


def safe(name):
    return ''.join('%04x' % ord(ch) for ch in name)


# ------------------------------------------------------------
# Rebuild the screen, now with seasons
# ------------------------------------------------------------
P = pd.read_csv(os.path.join(DATA, 'quotsoft_cities_daily.csv'),
                parse_dates=['date'])
P = P[(P['date'] >= WIN0) & (P['date'] <= WIN1)]
files = sorted(f for f in os.listdir(CACHE2) if f.endswith('.csv'))
print('cached cities: %d' % len(files))

rows = []
for f in files:
    m = pd.read_csv(os.path.join(CACHE2, f), parse_dates=['date'])
    if not len(m):
        continue
    city = str(m['city'].iloc[0])
    if city not in P.columns:
        continue
    lat, lon = float(m['lat'].iloc[0]), float(m['lon'].iloc[0])
    d = prep(m.merge(P[['date', city]].rename(columns={city: 'pm25'}),
                     on='date', how='inner'))
    full = first_stage(d)
    if full is None:
        continue
    w = first_stage(d[d['month'].isin([12, 1, 2])], MIN_DAYS_SEASON)
    s = first_stage(d[d['month'].isin([6, 7, 8])], MIN_DAYS_SEASON)
    rows.append({'city': city, 'lat': lat, 'lon': lon,
                 'coast_km': coast_km(lat, lon),
                 'b': full[0], 't': full[1], 'n': full[2],
                 'b_w': w[0] if w else np.nan,
                 't_w': w[1] if w else np.nan,
                 'b_s': s[0] if s else np.nan,
                 't_s': s[1] if s else np.nan})
R = pd.DataFrame(rows)
print('cities screened: %d' % len(R))

# ------------------------------------------------------------
# 1. The winter test, for real this time
# ------------------------------------------------------------
print('\n' + '=' * 72)
print('1. WINTER AGAINST SUMMER   (min %d days per season)'
      % MIN_DAYS_SEASON)
print('=' * 72)
s = R.dropna(subset=['b_w', 'b_s'])
print('   cities with both seasons: %d of %d' % (len(s), len(R)))
if len(s) >= 20:
    diff = s['b_w'] - s['b_s']
    print('   median winter  %+.4f' % s['b_w'].median())
    print('   median summer  %+.4f' % s['b_s'].median())
    print('   median(winter - summer) %+.4f' % diff.median())
    print('   winter WEAKER (less negative) in %.1f%% of cities'
          % (100 * (diff > 0).mean()))
    # Does the winter weakening grow as you go south, where northerly
    # transport matters more? That is the mechanism's own prediction.
    g = s.dropna(subset=['lat'])
    X = np.column_stack([np.ones(len(g)), g['lat'].values])
    yv = (g['b_w'] - g['b_s']).values
    bb, *_ = np.linalg.lstsq(X, yv, rcond=None)
    ee = yv - X @ bb
    Vv = (np.linalg.pinv(X.T @ X) @ (X.T @ (X * (ee ** 2)[:, None]))
          @ np.linalg.pinv(X.T @ X) * (len(g) / (len(g) - 2)))
    print('   (winter - summer) on latitude: %+.5f (t %+.2f)'
          % (bb[1], bb[1] / np.sqrt(Vv[1, 1])))
    print('''
   A POSITIVE latitude coefficient means the winter weakening is larger
   in the NORTH, which is not the transport story. A NEGATIVE one means
   it is larger in the SOUTH, where northerly transport arrives, which
   is what Section 6 claims for Shanghai.''')
else:
    print('   too few cities with both seasons to say anything')

# ------------------------------------------------------------
# 2. Which cities invert
# ------------------------------------------------------------
print('\n' + '=' * 72)
print('2. THE INVERTED CITIES')
print('=' * 72)
inv = R[(R['b'] > 0) & (R['t'] > 1.96)].sort_values('t', ascending=False)
print('   positive and significant: %d of %d (%.1f%%)'
      % (len(inv), len(R), 100.0 * len(inv) / max(len(R), 1)))
if len(inv):
    print('   %-12s %7s %8s %8s %9s %8s'
          % ('city', 'lat', 'lon', 'b', 't', 'coast km'))
    for r in inv.itertuples():
        print('   %-12s %7.2f %8.2f %8.3f %9.2f %8.0f'
              % (r.city, r.lat, r.lon, r.b, r.t, r.coast_km))
    print('\n   median latitude of inverted cities  %.1f' % inv['lat'].median())
    print('   median latitude of the rest         %.1f'
          % R[~R.index.isin(inv.index)]['lat'].median())
    print('   median coast km, inverted %.0f, rest %.0f'
          % (inv['coast_km'].median(),
             R[~R.index.isin(inv.index)]['coast_km'].median()))
    print('''
   Clustered in the southern coastal belt near Hong Kong -> the receptor
   story survives in a narrower, regional form. Scattered across the
   country -> it does not, and Section 6 must drop the general claim.''')

# ------------------------------------------------------------
# 3. Hong Kong, placed in the distribution
# ------------------------------------------------------------
print('\n' + '=' * 72)
print('3. HONG KONG')
print('=' * 72)
HK_LAT, HK_LON = 22.3193, 114.1694
cand = [f for f in os.listdir(DATA)
        if f.endswith('.csv') and any(k in f.lower()
                                      for k in ('hk', 'hong', 'epd'))]
print('   candidate HK pollution files: %s' % (cand or 'NONE FOUND'))

hk_pm = None
for f in cand:
    try:
        t = pd.read_csv(os.path.join(DATA, f))
        dc = next((c for c in t.columns
                   if 'date' in c.lower() or c.lower() == 'time'), None)
        vc = next((c for c in t.columns
                   if 'pm25' in c.lower().replace('.', '').replace('_', '')
                   or 'pm2' in c.lower()), None)
        if dc and vc:
            t = t[[dc, vc]].copy()
            t.columns = ['date', 'pm25']
            t['date'] = pd.to_datetime(t['date'], errors='coerce')
            t = t.dropna()
            t = t[(t['date'] >= WIN0) & (t['date'] <= WIN1)]
            if len(t) > MIN_DAYS:
                hk_pm = t.groupby('date', as_index=False)['pm25'].mean()
                print('   using %s: %d days in window' % (f, len(hk_pm)))
                break
    except Exception:
        continue

if hk_pm is None:
    print('   No usable HK pollution file found in %s.' % DATA)
    print('   Send me the filenames in that folder and I will point at it.')
else:
    hkf = os.path.join(CACHE2, safe('香港') + '.csv')
    if os.path.exists(hkf):
        met = pd.read_csv(hkf, parse_dates=['date'])
    else:
        print('   fetching HK meteorology (one call) ...')
        r = requests.get('https://archive-api.open-meteo.com/v1/archive',
                         params={'latitude': HK_LAT, 'longitude': HK_LON,
                                 'start_date': WIN0, 'end_date': WIN1,
                                 'hourly': ','.join(FIELDS),
                                 'timezone': 'Asia/Shanghai'}, timeout=600)
        r.raise_for_status()
        om = pd.DataFrame(r.json()['hourly']).drop_duplicates('time')
        om['time'] = pd.to_datetime(om['time'])
        rad = np.deg2rad(om['wind_direction_10m'])
        om['u'] = -om['wind_speed_10m'] * np.sin(rad)
        om['v'] = -om['wind_speed_10m'] * np.cos(rad)
        om['date'] = om['time'].dt.normalize()
        AGG = ['boundary_layer_height', 'wind_speed_10m', 'surface_pressure',
               'temperature_2m', 'precipitation', 'cloud_cover',
               'shortwave_radiation', 'u', 'v']
        grp = om.groupby('date')
        met = grp[AGG].mean().reset_index()
        met['n_hours'] = grp.size().values
        met = met[met['n_hours'] >= 8]
        met.insert(1, 'city', '香港')
        met.insert(2, 'lat', HK_LAT)
        met.insert(3, 'lon', HK_LON)
        met.to_csv(hkf, index=False)

    d = prep(met.merge(hk_pm, on='date', how='inner'))
    hk = first_stage(d)
    if hk is None:
        print('   too few overlapping days to fit')
    else:
        b_hk, t_hk, n_hk = hk
        pct = 100.0 * (R['b'] < b_hk).mean()
        print('   Hong Kong  b = %+.4f (t %+.2f), n = %d' % (b_hk, t_hk, n_hk))
        print('   percentile in the %d-city distribution: %.0f'
              % (len(R), pct))

        # Residual against the fitted geography model
        g = R.dropna(subset=['b', 'lat', 'coast_km'])
        Xg = np.column_stack([np.ones(len(g)), g['lat'].values,
                              np.log1p(g['coast_km'].values)])
        bg, *_ = np.linalg.lstsq(Xg, g['b'].values, rcond=None)
        pred = (bg[0] + bg[1] * HK_LAT
                + bg[2] * np.log1p(coast_km(HK_LAT, HK_LON)))
        resid = g['b'].values - Xg @ bg
        z = (b_hk - pred) / resid.std()
        print('   geography model predicts %+.4f; actual %+.4f'
              % (pred, b_hk))
        print('   residual %+.4f = %.1f standard deviations' % (b_hk - pred, z))
        print('''
   |z| under about 2 -> Hong Kong is an ordinary city for its position
   and Section 6's mechanism is not what makes it unusable; something
   else is, and the paper must say so.
   |z| above about 3 -> Hong Kong is a genuine outlier, which supports
   treating it as a distinct case but NOT the claim that receptor
   behaviour is a common, screenable property.''')

R.to_csv(os.path.join(DATA, 'city_screen_results.csv'), index=False)
print('\nwritten: %s' % os.path.join(DATA, 'city_screen_results.csv'))
