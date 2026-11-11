# Source: paper2_meteorology_build.ipynb, code cell 16 (final version of step 5 as run).
# ============================================================
# Step 5 — is ERA5's Hong Kong boundary layer real?
#
# The whole paper now branches on this. ERA5 says Hong Kong's winter
# boundary layer is DEEP on high-PM2.5 days, which inverts the standard
# dilution relationship. Either that is physics, in which case it is a
# finding, or ERA5 is wrong over coastal complex terrain, in which case
# Hong Kong drops out for a boring reason.
#
# Hong Kong Observatory launches radiosondes from King's Park at 00Z and
# 12Z, which are 08:00 and 20:00 local. ERA5 was pulled for all 24 hours,
# so every sounding has an exact ERA5 counterpart.
#
# Boundary layer height is computed from the raw profiles three ways:
#   parcel method        convective, virtual potential temperature excess
#   bulk Richardson      works in stable and convective conditions
#   theta gradient       height of maximum potential temperature gradient
#
# If ERA5 tracks all three, it is trustworthy. If it tracks none, it is not.
#
# Still pollution and meteorology only. Nothing here touches the A-H ratio.
# ============================================================

import io
import os
import zipfile

import numpy as np
import pandas as pd
import requests

DATA = '/content/drive/MyDrive/bilateral/data'
RAW = os.path.join(DATA, 'raw_igra')
os.makedirs(RAW, exist_ok=True)

IGRA = 'https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access'
STATION_LIST = 'https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt'

# ------------------------------------------------------------
# 1. Find King's Park in the station list
# ------------------------------------------------------------
sl_path = os.path.join(RAW, 'igra2-station-list.txt')
if not os.path.exists(sl_path):
    r = requests.get(STATION_LIST, timeout=180)
    r.raise_for_status()
    open(sl_path, 'wb').write(r.content)

lines = open(sl_path, encoding='latin-1').read().splitlines()
hits = [ln for ln in lines
        if '45004' in ln or 'KING' in ln.upper() or 'HONG KONG' in ln.upper()]
print('candidate stations:')
for h in hits:
    print('  ', h)

STATION = None
for h in hits:
    if '45004' in h:
        STATION = h.split()[0]
        break
if STATION is None:
    raise SystemExit('King\'s Park not found. Send me the lines above.')
print('\nusing station id: %s' % STATION)

# ------------------------------------------------------------
# 2. Download the raw soundings
# ------------------------------------------------------------
snd_path = os.path.join(RAW, '%s-data.txt' % STATION)
if not os.path.exists(snd_path):
    url = '%s/data-por/%s-data.txt.zip' % (IGRA, STATION)
    print('downloading %s ...' % url)
    r = requests.get(url, timeout=1800)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = z.namelist()[0]
    open(snd_path, 'wb').write(z.read(name))
print('soundings file: %s MB'
      % round(os.path.getsize(snd_path) / 1e6, 1))

# ------------------------------------------------------------
# 3. Parse. IGRA2 fixed-width. Offsets printed for verification.
# ------------------------------------------------------------
MISS = (-9999, -8888)


def fld(ln, a, b, scale=1.0):
    """One fixed-width field -> value, or NaN if blank or flagged missing.

    Parsed per field rather than per level: real IGRA levels often carry a
    temperature with blank wind, and dropping the whole level would throw
    away data the parcel and theta-gradient methods can still use.
    """
    s = ln[a:b]
    if not s.strip():
        return np.nan
    try:
        v = int(s)
    except ValueError:
        return np.nan
    return np.nan if v in MISS else v / scale


def parse(path, year_min=2013, year_max=2021):
    """Yield (year, month, day, hour, DataFrame of levels)."""
    cur = None
    rows = []
    shown = 0
    with open(path, encoding='latin-1') as f:
        for ln in f:
            if ln.startswith('#'):
                if cur is not None and rows:
                    yield cur, pd.DataFrame(rows)
                rows = []
                yr = int(ln[13:17])
                if not (year_min <= yr <= year_max):
                    cur = None
                    continue
                cur = (yr, int(ln[18:20]), int(ln[21:23]), int(ln[24:26]))
                continue
            if cur is None:
                continue
            row = {
                'p': fld(ln, 9, 15, 100.0),      # Pa -> hPa
                'z': fld(ln, 16, 21),            # m
                'T': fld(ln, 22, 27, 10.0),      # degC*10 -> degC
                'dpdp': fld(ln, 34, 39, 10.0),   # degC*10 -> degC
                'wd': fld(ln, 40, 45),           # deg
                'ws': fld(ln, 46, 51, 10.0),     # m/s*10 -> m/s
            }
            if shown < 3:
                print('  sample level: %s' % {k: (round(v, 1) if v == v else None)
                                              for k, v in row.items()})
                shown += 1
            rows.append(row)
    if cur is not None and rows:
        yield cur, pd.DataFrame(rows)


# ------------------------------------------------------------
# 4. Boundary layer height, three ways
# ------------------------------------------------------------
G = 9.80665
RD = 287.05
KAPPA = 0.286
EXCESS = 0.5      # K, parcel-method virtual temperature excess
RI_CRIT = 0.25
ZTOP = 4000.0     # m, search ceiling
DZ = 25.0         # m, analysis grid
GRID = np.arange(0.0, ZTOP + DZ, DZ)

_zcheck = []      # hypsometric vs reported height, mandatory levels


def thermo(d):
    """Profile -> thermodynamics on a regular height grid.

    IGRA reports geopotential height only at mandatory levels, so height is
    integrated from pressure and temperature with the hypsometric equation.
    Requiring a reported height instead would leave three or four points per
    profile, far too coarse to resolve a boundary layer.
    """
    d = d.dropna(subset=['p', 'T']).copy()
    d = d[(d['p'] > 100) & (d['p'] < 1100)]
    d = d[(d['T'] > -90) & (d['T'] < 60)]
    d = d.sort_values('p', ascending=False).drop_duplicates('p')
    if len(d) < 8:
        return None

    p = d['p'].values
    t = d['T'].values + 273.15
    td = d['T'].values - np.nan_to_num(d['dpdp'].values, nan=0.0)
    e = 6.112 * np.exp(17.67 * td / (td + 243.5))
    q = 0.622 * e / (p - 0.378 * e)
    tv = t * (1 + 0.608 * q)

    # Hypsometric integration upward from the surface level, z = 0 AGL.
    z = np.zeros_like(p)
    tvm = 0.5 * (tv[:-1] + tv[1:])
    z[1:] = np.cumsum((RD / G) * tvm * np.log(p[:-1] / p[1:]))

    rep = d['z'].values
    ok = ~np.isnan(rep)
    if ok.sum() >= 3 and len(_zcheck) < 400:
        # Reported heights are above sea level, computed ones above ground,
        # so compare after removing each profile's own offset.
        _zcheck.append((z[ok] - z[ok][0], rep[ok] - rep[ok][0]))

    theta = t * (1000.0 / p) ** KAPPA
    thetav = tv * (1000.0 / p) ** KAPPA

    keep = z <= ZTOP
    if keep.sum() < 8:
        return None
    z, theta, thetav = z[keep], theta[keep], thetav[keep]
    tc = d['T'].values[keep]

    g = pd.DataFrame({
        'zagl': GRID,
        'theta': np.interp(GRID, z, theta, left=np.nan, right=np.nan),
        'thetav': np.interp(GRID, z, thetav, left=np.nan, right=np.nan),
        'T': np.interp(GRID, z, tc, left=np.nan, right=np.nan),
    })

    # Wind is reported at a subset of levels; grid it separately.
    w = d.iloc[np.where(keep)[0]]
    has = ~np.isnan(w['ws'].values) & ~np.isnan(w['wd'].values)
    wz = z[has]
    wv = w.iloc[np.where(has)[0]]
    if len(wz) >= 5:
        rad = np.deg2rad(wv['wd'].values)
        uu = -wv['ws'].values * np.sin(rad)
        vv = -wv['ws'].values * np.cos(rad)
        g['u'] = np.interp(GRID, wz, uu, left=np.nan, right=np.nan)
        g['v'] = np.interp(GRID, wz, vv, left=np.nan, right=np.nan)
    else:
        g['u'] = np.nan
        g['v'] = np.nan

    g['n_src_2km'] = int((z <= 2000).sum())
    return g.dropna(subset=['theta', 'thetav'])


def blh_parcel(d):
    base = d['thetav'].iloc[0] + EXCESS
    hit = d[(d['zagl'] > 50) & (d['thetav'] >= base)]
    return float(hit['zagl'].iloc[0]) if len(hit) else np.nan


def blh_richardson(d):
    d = d.dropna(subset=['u', 'v', 'thetav'])
    if len(d) < 6:
        return np.nan
    tv0, u0, v0 = d['thetav'].iloc[0], d['u'].iloc[0], d['v'].iloc[0]
    z = d['zagl'].values
    shear = (d['u'].values - u0) ** 2 + (d['v'].values - v0) ** 2
    shear = np.where(shear < 0.1, 0.1, shear)
    ri = (G / tv0) * (d['thetav'].values - tv0) * z / shear
    ok = np.where((z > 50) & (ri >= RI_CRIT))[0]
    return float(z[ok[0]]) if len(ok) else np.nan


SPAN = 4            # grid steps, so gradients are over 100 m not 25 m


def blh_thetagrad(d):
    """Height of the maximum potential-temperature gradient.

    Differenced over 100 m rather than one grid step, because a 25 m
    gradient on an interpolated profile is mostly noise.
    """
    d = d[d['zagl'] > 50]
    if len(d) < 2 * SPAN + 2:
        return np.nan
    z, th = d['zagl'].values, d['theta'].values
    grad = (th[SPAN:] - th[:-SPAN]) / (z[SPAN:] - z[:-SPAN])
    zmid = 0.5 * (z[SPAN:] + z[:-SPAN])
    if not np.isfinite(grad).any():
        return np.nan
    return float(zmid[int(np.nanargmax(grad))])


MIN_INV = 50.0      # m, an inversion thinner than this is noise


def inversion(d):
    """Lowest inversion at least MIN_INV thick: base, top, strength."""
    d = d[d['zagl'] >= 0]
    if len(d) < 2 * SPAN + 2:
        return np.nan, np.nan, np.nan
    z, t = d['zagl'].values, d['T'].values
    dt = np.diff(t)
    i = 0
    while i < len(dt):
        if dt[i] > 0:
            j = i
            while j + 1 < len(dt) and dt[j + 1] > 0:
                j += 1
            if z[j + 1] - z[i] >= MIN_INV:
                return float(z[i]), float(z[j + 1]), float(t[j + 1] - t[i])
            i = j + 1
        else:
            i += 1
    return np.nan, np.nan, 0.0


print('\nparsing soundings (sample level values above are for offset checking):')
recs = []
for (yr, mo, dy, hr), lev in parse(snd_path):
    if hr not in (0, 12):
        continue
    d = thermo(lev)
    if d is None:
        continue
    ib, it, istr = inversion(d)
    recs.append({
        'date': pd.Timestamp(yr, mo, dy),
        'hour_local': 8 if hr == 0 else 20,
        'n_src_2km': int(d['n_src_2km'].iloc[0]),
        'blh_parcel': blh_parcel(d),
        'blh_ri': blh_richardson(d),
        'blh_thgrad': blh_thetagrad(d),
        'inv_base': ib, 'inv_top': it, 'inv_strength': istr,
        'T_sfc': d['T'].iloc[0],
    })

snd = pd.DataFrame(recs)
print('  %d usable soundings, %s to %s'
      % (len(snd), snd['date'].min().date(), snd['date'].max().date()))

# Does the hypsometric integration reproduce the reported heights?
if _zcheck:
    a = np.concatenate([x[0] for x in _zcheck])
    b = np.concatenate([x[1] for x in _zcheck])
    ok = np.isfinite(a) & np.isfinite(b)
    print('\nheight integration check, against reported heights at mandatory'
          ' levels (%d points from %d profiles):' % (ok.sum(), len(_zcheck)))
    print('   correlation      %.5f' % np.corrcoef(a[ok], b[ok])[0, 1])
    print('   mean abs error   %.1f m' % np.mean(np.abs(a[ok] - b[ok])))
    print('   median abs error %.1f m' % np.median(np.abs(a[ok] - b[ok])))
    print('   (want correlation above 0.999 and errors of tens of metres;'
          ' anything worse means the integration is wrong)')

print('\nsource levels below 2 km per sounding:')
print(snd['n_src_2km'].describe().round(1).to_string())

print('\nsounding-derived BLH, descriptive:')
print(snd[['blh_parcel', 'blh_ri', 'blh_thgrad',
           'inv_base', 'inv_top', 'inv_strength']]
      .describe().round(1).to_string())

# ------------------------------------------------------------
# 5. Merge with ERA5 and correlate. This is the answer.
# ------------------------------------------------------------
era = pd.read_csv(os.path.join(DATA, 'hongkong_blh_hourly.csv'),
                  parse_dates=['time', 'date'])
era = era.rename(columns={'hour': 'hour_local'})
ERA_COLS = [c for c in era.columns if c.startswith('blh_')]

m = snd.merge(era[['date', 'hour_local'] + ERA_COLS],
              on=['date', 'hour_local'], how='inner')
print('\nmatched sounding-ERA5 pairs: %d' % len(m))

SND_COLS = ['blh_parcel', 'blh_ri', 'blh_thgrad']
print('\n' + '=' * 64)
print('ERA5 against observed boundary layer height')
print('=' * 64)
for hr in (8, 20):
    sub = m[m['hour_local'] == hr]
    print('\n%02d:00 local  (n up to %d)' % (hr, len(sub)))
    print('   %-20s %12s %12s %12s'
          % ('ERA5 series', 'parcel', 'richardson', 'theta-grad'))
    for e in ERA_COLS:
        out = []
        for s in SND_COLS:
            p = sub[[e, s]].dropna()
            out.append(np.corrcoef(p[e], p[s])[0, 1] if len(p) > 30
                       else np.nan)
        print('   %-20s %12.3f %12.3f %12.3f' % (e, out[0], out[1], out[2]))

    print('   means (m):  ERA5 %s | parcel %.0f  ri %.0f  thgrad %.0f'
          % (', '.join('%s %.0f' % (e.replace('blh_', ''), sub[e].mean())
                       for e in ERA_COLS),
             sub['blh_parcel'].mean(), sub['blh_ri'].mean(),
             sub['blh_thgrad'].mean()))

out = os.path.join(DATA, 'hk_blh_validation.csv')
m.to_csv(out, index=False)
print('\nwritten: %s  (%d rows)' % (out, len(m)))

print("""
HOW TO READ THIS
  Correlations of 0.5 or better with two or three of the sounding methods:
  ERA5 is fine and the positive PM relationship is real physics.
  Correlations near zero or negative against all three: ERA5's Hong Kong
  boundary layer is a model artefact and Hong Kong drops out of the design.
  Something in between, around 0.2 to 0.4: ambiguous, and the observed
  inv_strength column becomes the instrument instead of ERA5 blh.
""")
