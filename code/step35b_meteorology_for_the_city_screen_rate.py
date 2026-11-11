# Source: paper2_meteorology_build.ipynb, code cell 74 (final version of step 35b as run).
# ============================================================
# Step 35b — meteorology for the city screen, rate-limit aware
#
# WHAT WENT WRONG IN STEP 35
#   Geocoding: Open-Meteo's geocoder throttled and the script logged a
#   throttle as "city not found". Shenzhen, Xiamen, Chongqing and Xi'an
#   were among the 44 "failures", which is how we know.
#   Downloads: Open-Meteo weights an archive call by variables x
#   timesteps, not by request count. Eight variables over six hourly
#   years is roughly 70 units; three calls a city over 141 cities is
#   about 30,000 units against a 10,000/day free allowance. It stopped
#   near city 80, which is what the arithmetic predicts.
#   The result was 80 of 185 cities selected by when the quota expired.
#
# THE FIXES
#   1. Geocode via OpenStreetMap Nominatim, 1 request/second as their
#      usage policy requires. Different service, no weighted quota.
#   2. Two-year window, 2017-2018, in ONE call per city. The screen is a
#      SIGN test, not an estimation: two years settles the sign of a
#      boundary-layer coefficient comfortably, and a third of the data
#      costs a third of the quota.
#   3. Reuse the 80 cities already cached. Those files carry the full
#      six years and are simply subset to the common window, so nothing
#      downloaded last night is wasted.
#
# Every city, Shanghai and Hong Kong included, runs on the same window,
# so the two cities the paper already discusses sit inside the same
# distribution as the rest instead of being special cases.
#
# Still resumable. Re-run until `remaining` reaches 0.
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
CACHE = os.path.join(DATA, 'city_meteo')          # step 35's 6-year files
CACHE2 = os.path.join(DATA, 'city_meteo_2y')      # this run's 2-year files
GEO = os.path.join(DATA, 'city_coords.csv')
os.makedirs(CACHE, exist_ok=True)
os.makedirs(CACHE2, exist_ok=True)

WIN0, WIN1 = '2017-01-01', '2018-12-31'
PANEL0, PANEL1 = '2014-05-13', '2019-12-31'
MIN_COVERAGE = 0.90
UA = {'User-Agent': 'ah-premium-city-screen/1.0 (academic research)'}

FIELDS = ['boundary_layer_height', 'wind_speed_10m', 'wind_direction_10m',
          'surface_pressure', 'temperature_2m', 'precipitation',
          'cloud_cover', 'shortwave_radiation']
AGG = ['boundary_layer_height', 'wind_speed_10m', 'surface_pressure',
       'temperature_2m', 'precipitation', 'cloud_cover',
       'shortwave_radiation', 'u', 'v']


def safe(name):
    return ''.join('%04x' % ord(ch) for ch in name)


def get(url, params, headers=None, tries=5, base=5.0):
    wait = base
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers or {},
                             timeout=600)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(wait)
            wait *= 2
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(wait)
            wait *= 2
            continue
        r.raise_for_status()
    raise RuntimeError('HTTP backoff exhausted')


# ------------------------------------------------------------
# 1. Cities: coverage measured on the SCREEN window
# ------------------------------------------------------------
D = pd.read_csv(os.path.join(DATA, 'quotsoft_cities_daily.csv'),
                parse_dates=['date'])
win = D[(D['date'] >= WIN0) & (D['date'] <= WIN1)]
cov = win.drop(columns=['date']).notna().mean()
cities = sorted(cov[cov >= MIN_COVERAGE].index.tolist())
print('window %s to %s' % (WIN0, WIN1))
print('cities with >=%.0f%% coverage in the window: %d'
      % (100 * MIN_COVERAGE, len(cities)))

# ------------------------------------------------------------
# 2. Coordinates via Nominatim
# ------------------------------------------------------------
coords = {}
if os.path.exists(GEO):
    g = pd.read_csv(GEO, dtype={'city': str}).dropna(subset=['lat', 'lon'])
    coords = {r.city: (r.lat, r.lon) for r in g.itertuples()}
    print('coordinates already cached: %d' % len(coords))

todo_geo = [c for c in cities if c not in coords]
if todo_geo:
    print('geocoding %d via Nominatim, 1/sec ...' % len(todo_geo))
    miss = []
    for i, c in enumerate(todo_geo, 1):
        hit = None
        for q in ('%s, China' % c, c):
            try:
                j = get('https://nominatim.openstreetmap.org/search',
                        {'q': q, 'format': 'json', 'limit': 1,
                         'countrycodes': 'cn,hk,mo'},
                        headers=UA, tries=3, base=3.0)
            except Exception:
                j = []
            if j:
                hit = (float(j[0]['lat']), float(j[0]['lon']))
                break
            time.sleep(1.1)
        if hit:
            coords[c] = hit
        else:
            miss.append(c)
        time.sleep(1.1)
        if i % 25 == 0:
            print('   %d/%d, %d unresolved' % (i, len(todo_geo), len(miss)))
            pd.DataFrame([{'city': k, 'lat': v[0], 'lon': v[1]}
                          for k, v in coords.items()]).to_csv(GEO,
                                                              index=False)
    pd.DataFrame([{'city': k, 'lat': v[0], 'lon': v[1]}
                  for k, v in coords.items()]).to_csv(GEO, index=False)
    print('resolved %d, unresolved %d' % (len(coords), len(miss)))
    if miss:
        print('   unresolved: %s' % miss)

cities = [c for c in cities if c in coords]
print('cities with coordinates: %d' % len(cities))

# ------------------------------------------------------------
# 3. Reuse what step 35 already downloaded
# ------------------------------------------------------------
reused = 0
for c in cities:
    dst = os.path.join(CACHE2, safe(c) + '.csv')
    src = os.path.join(CACHE, safe(c) + '.csv')
    if os.path.exists(dst) or not os.path.exists(src):
        continue
    try:
        m = pd.read_csv(src, parse_dates=['date'])
        m = m[(m['date'] >= WIN0) & (m['date'] <= WIN1)]
        if len(m) > 300:
            m.to_csv(dst, index=False)
            reused += 1
    except Exception:
        pass
print('reused from last night: %d' % reused)

# ------------------------------------------------------------
# 4. Download the rest, one call per city
# ------------------------------------------------------------
done = {f[:-4] for f in os.listdir(CACHE2) if f.endswith('.csv')}
todo = [c for c in cities if safe(c) not in done]
print('have %d, remaining %d' % (len(cities) - len(todo), len(todo)))

failed, streak, quota_hit, t0 = [], 0, False, time.time()
for i, c in enumerate(todo, 1):
    lat, lon = coords[c]
    try:
        j = get('https://archive-api.open-meteo.com/v1/archive',
                {'latitude': lat, 'longitude': lon,
                 'start_date': WIN0, 'end_date': WIN1,
                 'hourly': ','.join(FIELDS),
                 'timezone': 'Asia/Shanghai'})
        om = pd.DataFrame(j['hourly']).drop_duplicates('time')
        om['time'] = pd.to_datetime(om['time'])
        rad = np.deg2rad(om['wind_direction_10m'])
        om['u'] = -om['wind_speed_10m'] * np.sin(rad)
        om['v'] = -om['wind_speed_10m'] * np.cos(rad)
        om['date'] = om['time'].dt.normalize()
        grp = om.groupby('date')
        m = grp[AGG].mean().reset_index()
        m['n_hours'] = grp.size().values
        m = m[m['n_hours'] >= 8]
        m.insert(1, 'city', c)
        m.insert(2, 'lat', lat)
        m.insert(3, 'lon', lon)
        m.to_csv(os.path.join(CACHE2, safe(c) + '.csv'), index=False)
    except Exception as exc:
        failed.append(c)
        streak += 1
        # Five failures in a row means the daily allowance is gone, not
        # that these five cities are broken. Stop rather than spend an
        # hour proving it against a closed door.
        if streak >= 5:
            quota_hit = True
            print('\n%d consecutive failures at city %d of %d. The daily '
                  'quota is almost certainly exhausted.' % (streak, i,
                                                            len(todo)))
            print('Stopping. Re-run after 00:00 UTC; it resumes here.')
            break
        continue
    else:
        streak = 0
    time.sleep(1.2)
    if i % 10 == 0:
        el = time.time() - t0
        print('   %d/%d  %.0f min elapsed, ~%.0f min left'
              % (i, len(todo), el / 60,
                 el / 60 * (len(todo) - i) / max(i, 1)))

have = len([f for f in os.listdir(CACHE2) if f.endswith('.csv')])
print('\ncached: %d of %d cities (%.0f%%)'
      % (have, len(cities), 100.0 * have / max(len(cities), 1)))
if have < len(cities):
    print('Re-run this cell to continue; it skips everything on disk.')
    if quota_hit:
        print('If it stops immediately again, wait for 00:00 UTC.')
else:
    print('Complete. Run step 36.')
