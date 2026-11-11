# Source: paper2_meteorology_build.ipynb, code cell 69 (final version of step 38 as run).
# ============================================================
# Step 38 — validate coordinates against the pollution data itself
#
# WHY STEP 37 FAILED. It arbitrated between two geocoders assuming both
# were comparably reliable. Reading its output, they are not: Nominatim
# was right and Open-Meteo wrong in nearly every disagreement (Heihe,
# Ningde, Yulin, Siping, Tonghua, Penglai, Lijiang, Baoshan all placed
# in the wrong province by the second source). Dropping those 51 would
# have discarded mostly correct coordinates. Meanwhile 佛山, the one
# error known to be real, was never flagged, and 131 cities came back
# unverified because the second source throttled.
#
# A test that misses the known failure and flags good data instead is
# not a test.
#
# THE REPLACEMENT. Chinese air pollution is strongly spatially
# correlated: cities a few hundred kilometres apart share weather
# systems and airsheds, so their daily series move together. So:
#
#   for each city, correlate its daily log PM2.5 with that of its
#   nearest neighbours BY COORDINATE.
#
# A correctly placed city correlates highly with the cities actually
# near it. A mis-placed one is compared against the wrong neighbours and
# its correlation collapses. Foshan's Guangdong pollution sitting on a
# Yunnan coordinate is exactly the case this catches.
#
# This tests the thing that matters: whether the coordinate we
# downloaded weather for belongs to the pollution series paired with it.
# A wrong coordinate means a city's entire first stage is built from
# another place's weather, which is indistinguishable from a genuine
# inversion unless it is caught here.
#
# No network required.
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
GEO = os.path.join(DATA, 'city_coords.csv')
OUT = os.path.join(DATA, 'city_coords_sane.csv')
WIN0, WIN1 = '2017-01-01', '2018-12-31'
K_NEIGHBOURS = 5
MAX_NEIGHBOUR_KM = 600.0
MIN_OVERLAP = 300


def km_matrix(lat, lon):
    la = np.radians(lat)[:, None]
    lo = np.radians(lon)[:, None]
    dla = la - la.T
    dlo = lo - lo.T
    a = (np.sin(dla / 2) ** 2
         + np.cos(la) * np.cos(la.T) * np.sin(dlo / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


g = pd.read_csv(GEO, dtype={'city': str}).dropna(subset=['lat', 'lon'])
P = pd.read_csv(os.path.join(DATA, 'quotsoft_cities_daily.csv'),
                parse_dates=['date'])
P = P[(P['date'] >= WIN0) & (P['date'] <= WIN1)]

cities = [c for c in g['city'] if c in P.columns]
g = g[g['city'].isin(cities)].drop_duplicates('city').reset_index(drop=True)
print('cities with coordinates and pollution: %d' % len(g))

L = np.log(P[g['city'].tolist()].where(lambda x: x > 0))
L = L.loc[:, L.notna().sum() >= MIN_OVERLAP]
g = g[g['city'].isin(L.columns)].reset_index(drop=True)
C = L[g['city'].tolist()].corr(min_periods=MIN_OVERLAP).values
Dm = km_matrix(g['lat'].values, g['lon'].values)
np.fill_diagonal(Dm, np.inf)
print('usable after overlap filter: %d' % len(g))

rows = []
for i, city in enumerate(g['city']):
    order = np.argsort(Dm[i])
    near = [j for j in order[:K_NEIGHBOURS * 3]
            if Dm[i, j] <= MAX_NEIGHBOUR_KM][:K_NEIGHBOURS]
    if len(near) < 3:
        rows.append({'city': city, 'lat': g['lat'][i], 'lon': g['lon'][i],
                     'n_near': len(near), 'mean_r': np.nan,
                     'near_km': np.nan})
        continue
    rs = [C[i, j] for j in near if np.isfinite(C[i, j])]
    rows.append({'city': city, 'lat': g['lat'][i], 'lon': g['lon'][i],
                 'n_near': len(near),
                 'mean_r': float(np.mean(rs)) if rs else np.nan,
                 'near_km': float(np.mean([Dm[i, j] for j in near]))})

S = pd.DataFrame(rows)
ok = S.dropna(subset=['mean_r'])
print('\n' + '=' * 72)
print('NEIGHBOUR CORRELATION OF DAILY log PM2.5')
print('=' * 72)
print('   cities with >=3 neighbours within %.0f km: %d'
      % (MAX_NEIGHBOUR_KM, len(ok)))
print('   median mean_r %.3f   p10 %.3f   p05 %.3f   min %.3f'
      % (ok['mean_r'].median(), ok['mean_r'].quantile(.10),
         ok['mean_r'].quantile(.05), ok['mean_r'].min()))

# The correlation falls with distance, so judge each city against what
# its own neighbour distance predicts rather than one flat threshold.
X = np.column_stack([np.ones(len(ok)), ok['near_km'].values])
b, *_ = np.linalg.lstsq(X, ok['mean_r'].values, rcond=None)
pred = X @ b
resid = ok['mean_r'].values - pred
sd = resid.std()
ok = ok.assign(expected=pred, resid=resid, z=resid / sd)
print('   mean_r on neighbour distance: intercept %.3f, slope %+.5f /km'
      % (b[0], b[1]))
print('   residual sd %.3f' % sd)

flag = ok[ok['z'] < -2.5].sort_values('z')
print('\n   cities more than 2.5 sd BELOW what their neighbour distance')
print('   predicts (%d of %d):' % (len(flag), len(ok)))
print('   %-12s %7s %8s %7s %9s %7s'
      % ('city', 'lat', 'lon', 'mean_r', 'expected', 'z'))
for r in flag.itertuples():
    print('   %-12s %7.2f %8.2f %7.3f %9.3f %7.2f'
          % (r.city, r.lat, r.lon, r.mean_r, r.expected, r.z))

print('\n' + '=' * 72)
print('THE CITIES THE SCREEN CALLED INVERTED')
print('=' * 72)
INV = ['三亚', '安顺', '拉萨', '临沧', '梧州', '佛山', '梅州', '永州']
print('   %-12s %7s %8s %7s %9s %7s %s'
      % ('city', 'lat', 'lon', 'mean_r', 'expected', 'z', 'verdict'))
for c in INV:
    r = ok[ok['city'] == c]
    if not len(r):
        print('   %-12s  no neighbour test available' % c)
        continue
    r = r.iloc[0]
    verdict = 'SUSPECT' if r['z'] < -2.5 else 'coordinate looks fine'
    print('   %-12s %7.2f %8.2f %7.3f %9.3f %7.2f  %s'
          % (c, r['lat'], r['lon'], r['mean_r'], r['expected'], r['z'],
             verdict))
print('''
   An inverted city that is ALSO coordinate-suspect is a data error, not
   a receptor site, and must leave the inverted set. One that passes is
   a real inversion and stays.''')

keep = ok[ok['z'] >= -2.5][['city', 'lat', 'lon']]
keep.to_csv(OUT, index=False)
print('\ncoordinates passing the sanity test: %d of %d -> %s'
      % (len(keep), len(ok), OUT))
print('''
NOTE ON WHAT THIS CANNOT DO
  It catches a coordinate in the wrong REGION, which is the failure mode
  that corrupts a city's meteorology. It will not catch an error of
  twenty kilometres, which barely changes the weather anyway.

  It also cannot flag a city whose neighbours are themselves
  mis-placed. If a whole region is wrong the test goes quiet, so the
  count above is a floor on the number of good coordinates, not proof.
''')
