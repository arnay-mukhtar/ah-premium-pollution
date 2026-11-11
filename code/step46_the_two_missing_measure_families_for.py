# Source: paper2_meteorology_build.ipynb, code cell 89 (final version of step 46 as run).
# ============================================================
# Step 46 — the two missing measure families for the Hong Kong table
#
# WHY. Section 6.1 names six families of dispersion measures. The table
# reports four: ERA5 boundary layer height, wind components, surface
# pressure, and radiosonde temperature differences. The two it does not
# report are:
#
#     parcel-method mixing depth
#     bulk Richardson number, critical value 0.25
#
# Neither survives in sounding_daily.csv, which holds only the dT_*
# columns. So they are computed here from source and added as two rows.
#
# WHERE THEY COME FROM. The IGRA v2.2 DERIVED sounding files carry both
# the inputs and, for one of them, the answer:
#
#   MIXHGT   header field, columns 68-73, metres above surface,
#            "height of the top of the mixed layer as determined using
#            the parcel method" — this IS the first family, no
#            computation needed beyond parsing.
#
#   CALCGPH  data field, cols 17-23, m      ) the inputs for a bulk
#   VPTEMP   data field, cols 65-71, K x 10 ) Richardson profile, from
#   UWND     data field, cols 113-119, m/s x 10 ) which the critical-0.25
#   VWND     data field, cols 129-135, m/s x 10 ) height is derived here
#
#   Ri_b(z) = (g / th_v0) * (th_v(z) - th_v0) * (z - z0)
#                         / [ (u(z) - u0)^2 + (v(z) - v0)^2 ]
#
#   taking the surface level as the reference and the boundary layer top
#   as the lowest height where Ri_b crosses 0.25, linearly interpolated
#   between the bracketing levels. This is the standard bulk form; it
#   omits the surface-friction term, which is what "bulk Richardson
#   number" in the paper's sense means.
#
# SIGN CONVENTION — READ THIS BEFORE READING THE OUTPUT.
# Both new measures are DEPTHS, like ERA5 boundary layer height, not
# stability differences. Deeper mixing dilutes, so:
#
#     NEGATIVE coefficient = source site = the instrument works
#     POSITIVE coefficient = receptor site = the inversion
#
# That is the OPPOSITE of the dT rows, where positive means an inversion
# trapping pollution and so means source-site behaviour. Reading one with
# the other's convention inverts the conclusion. The script prints the
# expected sign on every row so this cannot be got wrong by eye.
#
# WHAT IT DECIDES, EITHER WAY.
#   Both new families wrong-signed in the cold season, like ERA5 BLH:
#   Section 6 is strengthened, the table becomes genuinely complete, and
#   6.1's claim of six families becomes true as written.
#   Either family correctly signed and strong in the cold season:
#   Section 6.2's claim that the failure is in the atmosphere rather than
#   the data product needs revisiting, because a correctly signed
#   OBSERVED depth measure in the failing months would point back at the
#   reanalysis after all. Report whatever comes out.
#
# Also runs the seasonal split, since 6.4 is the paper's central Hong
# Kong claim and these two measures span the full year.
# ============================================================

import io
import os
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import requests

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DATA = '/content/drive/MyDrive/bilateral/data'
DIR = '/content/drive/MyDrive/pollution-ah'
FULL0, FULL1 = '2014-05-13', '2019-12-31'
CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']
WINTER = [12, 1, 2]
RI_CRIT = 0.25
G = 9.80665

HK_LAT, HK_LON = 22.3193, 114.1694
FALLBACK_ID = 'CHM00045004'          # King's Park, used only if the
                                     # station list cannot be reached

STATION_URLS = [
    'https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt',
    'https://www.ncei.noaa.gov/pub/data/igra/igrav2.2-stations.txt',
    'https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/'
    'doc/igra2-station-list.txt',
]
DERIVED_DIRS = [
    'https://www.ncei.noaa.gov/pub/data/igra/derived/derived-por/',
    'https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/'
    'access/derived-por/',
    'https://www.ncei.noaa.gov/pub/data/igra/derived/derived-por2/',
]


def get(url, timeout=120):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r


# ------------------------------------------------------------
# 1. Find the Hong Kong station BY COORDINATES, not by name
# ------------------------------------------------------------
print('=' * 74)
print('1. LOCATING THE STATION')
print('=' * 74)

station_id, station_name = None, None
for u in STATION_URLS:
    try:
        txt = get(u).text
    except Exception as e:
        print('   %-62s %s' % (u.split('/')[-1], type(e).__name__))
        continue
    best, bestd = None, 1e9
    for ln in txt.splitlines():
        if len(ln) < 40:
            continue
        try:
            sid = ln[0:11].strip()
            lat = float(ln[12:20])
            lon = float(ln[21:30])
        except ValueError:
            continue
        if abs(lat) > 90 or abs(lon) > 180:
            continue
        d = ((lat - HK_LAT) ** 2 + (lon - HK_LON) ** 2) ** 0.5
        if d < bestd:
            best, bestd = (sid, ln[41:71].strip(), lat, lon), d
    if best and bestd < 1.0:
        station_id, station_name = best[0], best[1]
        print('   source: %s' % u)
        print('   nearest station to %.4f, %.4f: %s  %s  (%.4f, %.4f), '
              '%.0f km away'
              % (HK_LAT, HK_LON, station_id, station_name, best[2], best[3],
                 bestd * 111.0))
        break

if station_id is None:
    station_id, station_name = FALLBACK_ID, 'King\'s Park (fallback)'
    print('   station list unreachable; falling back to %s' % station_id)

# ------------------------------------------------------------
# 2. Download and parse the derived file
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('2. DOWNLOADING THE DERIVED SOUNDINGS')
print('=' * 74)

raw = None
for d in DERIVED_DIRS:
    u = d + station_id + '-drvd.txt.zip'
    try:
        r = get(u, timeout=300)
    except Exception as e:
        print('   %-70s %s' % (u[-58:], type(e).__name__))
        continue
    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
    except zipfile.BadZipFile:
        print('   %-70s not a zip' % u[-58:])
        continue
    raw = z.read(z.namelist()[0]).decode('latin-1')
    print('   got %s  (%.1f MB unzipped)' % (u, len(raw) / 1e6))
    break

if raw is None:
    raise SystemExit(
        'Could not reach any derived-file directory. Paste the list of '
        'URLs tried above and I will find the current path.')


def i(s):
    s = s.strip()
    if not s:
        return np.nan
    v = int(s)
    return np.nan if v == -99999 else float(v)


rows, prof = [], []
cur = None
nlev = 0
for ln in raw.splitlines():
    if ln.startswith('#'):
        if cur is not None:
            rows.append(cur)
        cur = {
            'year': i(ln[13:17]), 'month': i(ln[18:20]),
            'day': i(ln[21:23]), 'hour': i(ln[24:26]),
            'numlev': i(ln[31:36]),
            'mixhgt': i(ln[67:73]),           # cols 68-73, m above surface
            'levels': [],
        }
        nlev = 0
        continue
    if cur is None:
        continue
    cur['levels'].append((
        i(ln[16:23]),      # CALCGPH, m
        i(ln[64:71]),      # VPTEMP, K x 10
        i(ln[112:119]),    # UWND, m/s x 10
        i(ln[128:135]),    # VWND, m/s x 10
    ))
    nlev += 1
if cur is not None:
    rows.append(cur)

print('   soundings parsed: %s' % format(len(rows), ','))


def bulk_richardson_height(levels):
    """Lowest height where Ri_b crosses RI_CRIT, metres above surface."""
    L = [(z, th / 10.0, u / 10.0, v / 10.0)
         for (z, th, u, v) in levels
         if not (np.isnan(z) or np.isnan(th) or np.isnan(u) or np.isnan(v))]
    if len(L) < 4:
        return np.nan
    L.sort(key=lambda t: t[0])
    z0, th0, u0, v0 = L[0]
    prev_z, prev_ri = 0.0, 0.0
    for z, th, u, v in L[1:]:
        dz = z - z0
        if dz <= 0:
            continue
        if dz > 4000:                 # no boundary layer above 4 km
            break
        shear = (u - u0) ** 2 + (v - v0) ** 2
        if shear < 0.25:              # m^2/s^2; below this Ri_b explodes
            continue
        ri = (G / th0) * (th - th0) * dz / shear
        if ri >= RI_CRIT:
            if ri == prev_ri:
                return dz
            f = (RI_CRIT - prev_ri) / (ri - prev_ri)
            return prev_z + f * (dz - prev_z)
        prev_z, prev_ri = dz, ri
    return np.nan


recs = []
for r in rows:
    if np.isnan(r['year']) or np.isnan(r['month']) or np.isnan(r['day']):
        continue
    recs.append({
        'date': pd.Timestamp(int(r['year']), int(r['month']), int(r['day'])),
        'hour': r['hour'],
        'mix_depth': r['mixhgt'],
        'ri_depth': bulk_richardson_height(r['levels']),
    })
S = pd.DataFrame(recs)
print('   mixing depth present:    %s of %s soundings'
      % (format(int(S['mix_depth'].notna().sum()), ','), format(len(S), ',')))
print('   Richardson depth solved: %s of %s soundings'
      % (format(int(S['ri_depth'].notna().sum()), ','), format(len(S), ',')))
print('   sounding hours present:  %s'
      % dict(Counter(S['hour'].dropna().astype(int)).most_common()))

# The paper uses the morning sounding. 00Z is 08:00 Hong Kong time.
S00 = S[S['hour'] == 0].copy()
if len(S00) < 300:
    print('   00Z too sparse (%d); using the daily mean of all hours'
          % len(S00))
    S00 = S.groupby('date', as_index=False)[['mix_depth',
                                             'ri_depth']].mean()
else:
    S00 = S00.groupby('date', as_index=False)[['mix_depth',
                                               'ri_depth']].mean()
    print('   using the 00Z sounding (08:00 local), %d days' % len(S00))

os.makedirs(DATA, exist_ok=True)
S00.to_csv(os.path.join(DATA, 'hk_sounding_depths.csv'), index=False)
print('   saved -> hk_sounding_depths.csv')

# ------------------------------------------------------------
# 3. Rebuild the Hong Kong daily panel
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('3. BUILDING THE PANEL')
print('=' * 74)


def read_messy(path):
    for enc in ('utf-8-sig', 'utf-8', 'gb18030', 'latin-1'):
        try:
            with open(path, 'r', encoding=enc, errors='strict') as fh:
                head = [fh.readline() for _ in range(80)]
            break
        except (UnicodeDecodeError, LookupError):
            continue
    counts = [l.count(',') for l in head]
    body = [c for c in counts[3:] if c > 0]
    skip = 0
    if body:
        mode = Counter(body).most_common(1)[0][0]
        skip = next(k for k, c in enumerate(counts) if c == mode)
    return pd.read_csv(path, skiprows=skip, encoding=enc, low_memory=False)


parts = [read_messy(os.path.join(DIR, f))
         for f in ('hk_pm25_2013_2017.csv', 'hk_pm25_2018_2022.csv')
         if os.path.exists(os.path.join(DIR, f))]
P = pd.concat(parts, ignore_index=True)
STATION = [c for c in P.columns
           if c.upper() not in ('DATE', 'HOUR', 'POLLUTANT')]
P = P.assign(date=pd.to_datetime(P['DATE'], dayfirst=True,
                                 errors='coerce').dt.normalize())
lg = P.melt(id_vars=['date'], value_vars=STATION, value_name='pm25')
lg['pm25'] = pd.to_numeric(lg['pm25'], errors='coerce')
lg = lg.dropna(subset=['pm25', 'date'])
hk = lg[lg['pm25'] > 0].groupby('date', as_index=False)['pm25'].mean()

W = pd.read_csv(os.path.join(DATA, 'hongkong_weather_hourly.csv'))
W['time'] = pd.to_datetime(W['time'])
W['date'] = W['time'].dt.normalize()
grp = W.groupby('date')
met = grp[CONTROLS].mean().reset_index()
met['n_hours'] = grp.size().values
met = met[met['n_hours'] >= 8]

d = met.merge(hk, on='date', how='inner').merge(S00, on='date', how='inner')
d = d[(d['date'] >= FULL0) & (d['date'] <= FULL1)]
d = d[d['pm25'] > 0].copy()
d['log_pm25'] = np.log(d['pm25'])
d['month'] = d['date'].dt.month
d['year'] = d['date'].dt.year
d['dow'] = d['date'].dt.dayofweek
d['log_mix'] = np.log(d['mix_depth'].clip(lower=1))
d['log_ri'] = np.log(d['ri_depth'].clip(lower=1))
print('   panel: %d days, %s to %s'
      % (len(d), d['date'].min().date(), d['date'].max().date()))
for c in ('log_mix', 'log_ri'):
    s = d[c].dropna()
    print('   %-8s n %4d  mean %6.3f  sd %5.3f  (depth median %6.0f m)'
          % (c, len(s), s.mean(), s.std(),
             np.exp(np.median(s)) if len(s) else np.nan))


def fe(s):
    cols = [c for c in ('month', 'year', 'dow') if s[c].nunique() > 1]
    if not cols:
        return np.zeros((len(s), 0))
    return pd.get_dummies(s[cols].astype(str),
                          drop_first=True).astype(float).values


def one(y, x, W_):
    """Coefficient and HC1 t on a single regressor, given controls."""
    X = np.column_stack([np.ones(len(y)), x, W_])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    V = XtXi @ (X.T @ (X * (e ** 2)[:, None])) @ XtXi * (n / (n - k))
    se = float(np.sqrt(max(V[1, 1], 1e-300)))
    # partial R-squared of x alone
    Xr = np.column_stack([np.ones(len(y)), W_])
    br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    er = y - Xr @ br
    pr2 = 1.0 - (e @ e) / (er @ er)
    return float(b[1]), se, float(b[1]) / se, pr2, n


def eff_f(y, Z, W_):
    Wc = np.column_stack([np.ones(len(y)), W_])
    cf, *_ = np.linalg.lstsq(Wc, np.column_stack([y, Z]), rcond=None)
    R = np.column_stack([y, Z]) - Wc @ cf
    yt, Zt = R[:, 0], R[:, 1:]
    ZtZ = Zt.T @ Zt
    if np.linalg.cond(ZtZ) > 1e12:
        return np.nan
    pi = np.linalg.solve(ZtZ, Zt.T @ yt)
    e = yt - Zt @ pi
    n, k = Zt.shape
    br = np.linalg.inv(ZtZ)
    V = (br @ (Zt.T @ (Zt * (e ** 2)[:, None])) @ br
         * (n / max(n - k - Wc.shape[1], 1)))
    return float(pi @ ZtZ @ pi / np.trace(ZtZ @ V))


MEASURES = [('Parcel-method mixing depth', 'log_mix'),
            ('Bulk Richardson number, 0.25', 'log_ri')]

# ------------------------------------------------------------
# 4. The two new table rows
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('4. TWO NEW ROWS, THREE CONTROL SETS')
print('=' * 74)
print('   These are DEPTH measures. NEGATIVE = source site = works.')
print('   POSITIVE = receptor site = the inversion. This is the opposite')
print('   of the dT rows already in the table.\n')
print('   %-30s %-20s %-20s %-20s'
      % ('Measure', 'No controls', 'Weather controls', 'Weather + FE'))
for name, col in MEASURES:
    s = d.dropna(subset=[col, 'log_pm25'] + CONTROLS)
    cells = []
    for lab in ('none', 'weather', 'fe'):
        if lab == 'none':
            Wm = np.zeros((len(s), 0))
        elif lab == 'weather':
            Wm = s[CONTROLS].values
        else:
            Wm = np.column_stack([s[CONTROLS].values, fe(s)])
        b, se, t, pr2, n = one(s['log_pm25'].values, s[col].values, Wm)
        cells.append('%+.3f (t %+.2f)' % (b, t))
    print('   %-30s %-20s %-20s %-20s' % (name, cells[0], cells[1], cells[2]))
    print('   %-30s n = %d, partial R2 under weather+FE = %.4f'
          % ('', len(s), pr2))

# ------------------------------------------------------------
# 5. Seasonal split, which is the paper's central Hong Kong claim
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('5. SEASONAL SPLIT, weather controls only')
print('=' * 74)
print('   %-30s %-12s %8s %9s %11s'
      % ('Measure / season', 'days', 'coef', 'F_eff', 'partial R2'))
for name, col in MEASURES:
    base = d.dropna(subset=[col, 'log_pm25'] + CONTROLS)
    for lab, mask in [('full year', pd.Series(True, index=base.index)),
                      ('non-winter', ~base['month'].isin(WINTER)),
                      ('winter (DJF)', base['month'].isin(WINTER))]:
        s = base[mask]
        if len(s) < 120:
            print('   %-30s %-12d  too few days' % (name + ', ' + lab, len(s)))
            continue
        b, se, t, pr2, n = one(s['log_pm25'].values, s[col].values,
                               s[CONTROLS].values)
        F = eff_f(s['log_pm25'].values, s[[col]].values, s[CONTROLS].values)
        print('   %-30s %-12d %+8.4f %9.2f %11.4f'
              % (name + ', ' + lab, len(s), b, F, pr2))

print("""
======================================================================
HOW TO READ IT
  Expected if Hong Kong is a receptor site in the cold season, which is
  what Section 6 now argues: POSITIVE coefficients in winter on both new
  measures, turning negative or vanishing outside winter. That makes the
  table complete and 6.1's "six families" true as written.

  If instead either measure is strongly NEGATIVE in winter, an observed
  depth measure works in the months where the reanalysis fails, and
  Section 6.2's claim that the problem is the atmosphere rather than the
  data product has to be rewritten. Send the output either way.

  Note the partial R-squared column as well as the sign. Section 6.5
  turns on thinness, not sign, so a correctly signed measure with a
  partial R-squared near 0.01 changes nothing in the conclusion.
======================================================================
""")
