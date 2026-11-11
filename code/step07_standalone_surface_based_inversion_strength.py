# Source: paper2_meteorology_build.ipynb, code cell 19 (final version of step 7 as run).
# ============================================================
# Step 7 (STANDALONE) — surface-based inversion strength
#
# Self-contained: carries its own parser and thermodynamics, so it runs
# on a fresh runtime. The sounding files are already on your Drive from
# steps 5 and 6, so nothing is re-downloaded.
#
# Step 6's inversion measure failed in both cities because it searched
# for the lowest inversion anywhere up to 4 km and kept finding the
# elevated subsidence inversion near the boundary layer top: median base
# 825 m, median strength 0.8 K. That is not a trapping measure.
#
# The literature instrument (Chen, Oliva & Zhang; Arceo, Hanna & Oliva)
# is a temperature difference between FIXED LOW LEVELS, surface-based by
# construction. Five of those are built here, plus the low-level lapse
# rate.
#
# Expected sign on all of them: POSITIVE. Warm air over a colder surface
# caps vertical exchange and traps emissions.
#
# Pollution and meteorology only. Nothing touches the A-H ratio.
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
RAW = os.path.join(DATA, 'raw_igra')

# ------------------------------------------------------------
# Station lookup. WMO: King's Park 45004, Baoshan 58362.
# ------------------------------------------------------------
WANT = {'hongkong': '45004', 'shanghai': '58362'}

lines = open(os.path.join(RAW, 'igra2-station-list.txt'),
             encoding='latin-1').read().splitlines()

ids = {}
for city, wmo in WANT.items():
    hit = [ln for ln in lines if wmo in ln[:12]]
    if hit:
        ids[city] = hit[0].split()[0]
        print('%-9s -> %s' % (city, hit[0].strip()))
    else:
        print('%-9s -> WMO %s NOT FOUND' % (city, wmo))

for city, st in ids.items():
    p = os.path.join(RAW, '%s-data.txt' % st)
    print('  %s  %s' % (st, 'present, %.0f MB' % (os.path.getsize(p) / 1e6)
                        if os.path.exists(p) else 'MISSING, rerun step 6'))

# ------------------------------------------------------------
# Parser (IGRA2 fixed width). Verified against the documented layout.
# ------------------------------------------------------------
MISS = (-9999, -8888)


def fld(ln, a, b, scale=1.0):
    """One field -> value, or NaN if blank or flagged missing.

    Per field, not per level: real levels often carry a temperature with
    blank wind, and dropping the level would lose the temperature too.
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
    """Yield (year, month, day, hour), DataFrame of levels."""
    cur, rows = None, []
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
            rows.append({
                'p': fld(ln, 9, 15, 100.0),      # Pa -> hPa
                'z': fld(ln, 16, 21),            # m
                'T': fld(ln, 22, 27, 10.0),      # degC*10 -> degC
                'dpdp': fld(ln, 34, 39, 10.0),   # degC*10 -> degC
                'wd': fld(ln, 40, 45),           # deg
                'ws': fld(ln, 46, 51, 10.0),     # m/s*10 -> m/s
            })
    if cur is not None and rows:
        yield cur, pd.DataFrame(rows)


# ------------------------------------------------------------
# Thermodynamics on a regular height grid
# ------------------------------------------------------------
G = 9.80665
RD = 287.05
KAPPA = 0.286
ZTOP = 4000.0
DZ = 25.0
GRID = np.arange(0.0, ZTOP + DZ, DZ)

_zcheck = []


def thermo(d):
    """Profile -> thermodynamics on a 25 m grid.

    IGRA reports geopotential height only at mandatory levels, so height
    is integrated from pressure and temperature. Requiring a reported
    height leaves three or four points per profile, far too coarse.
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

    z = np.zeros_like(p)
    tvm = 0.5 * (tv[:-1] + tv[1:])
    z[1:] = np.cumsum((RD / G) * tvm * np.log(p[:-1] / p[1:]))

    rep = d['z'].values
    ok = ~np.isnan(rep)
    if ok.sum() >= 3 and len(_zcheck) < 400:
        _zcheck.append((z[ok] - z[ok][0], rep[ok] - rep[ok][0]))

    keep = z <= ZTOP
    if keep.sum() < 8:
        return None
    zk = z[keep]
    tck = d['T'].values[keep]

    return pd.DataFrame({
        'zagl': GRID,
        'T': np.interp(GRID, zk, tck, left=np.nan, right=np.nan),
    }).dropna(subset=['T'])


# ------------------------------------------------------------
# Surface-based measures
# ------------------------------------------------------------
LEVELS = [100.0, 200.0, 300.0, 500.0, 1000.0]


def at_height(g, h):
    s = g.loc[(g['zagl'] - h).abs() < 1e-6, 'T']
    return float(s.iloc[0]) if len(s) else np.nan


def inv_measures(g):
    t0 = at_height(g, 0.0)
    if t0 != t0:
        t0 = float(g['T'].iloc[0]) if len(g) else np.nan
    out = {}
    for h in LEVELS:
        out['dT_%d' % int(h)] = at_height(g, h) - t0
    t500 = at_height(g, 500.0)
    out['lapse_0_500'] = (t500 - t0) / 0.5 if t500 == t500 else np.nan
    return out


print('\nparsing morning profiles ...')
rows = []
for city, station in ids.items():
    path = os.path.join(RAW, '%s-data.txt' % station)
    n = 0
    for (yr, mo, dy, hr), lev in parse(path):
        if hr != 0:          # 00Z = 08:00 local, start of the window
            continue
        g = thermo(lev)
        if g is None:
            continue
        r = {'city': city, 'date': pd.Timestamp(yr, mo, dy)}
        r.update(inv_measures(g))
        rows.append(r)
        n += 1
    print('  %-9s %d morning profiles' % (city, n))

inv = pd.DataFrame(rows)

if _zcheck:
    a = np.concatenate([x[0] for x in _zcheck])
    b = np.concatenate([x[1] for x in _zcheck])
    ok = np.isfinite(a) & np.isfinite(b)
    print('\nheight integration check: corr %.5f, median abs error %.1f m'
          % (np.corrcoef(a[ok], b[ok])[0, 1],
             np.median(np.abs(a[ok] - b[ok]))))

COLS = ['dT_100', 'dT_200', 'dT_300', 'dT_500', 'dT_1000', 'lapse_0_500']
print('\nsurface-based measures by city (deg C; lapse in C/km):')
print(inv.groupby('city')[COLS].agg(['mean', 'std', 'min', 'max'])
      .T.round(2).to_string())

# ------------------------------------------------------------
# Merge and test
# ------------------------------------------------------------
d = pd.read_csv(os.path.join(DATA, 'observed_dispersion_panel.csv'),
                parse_dates=['date'])
d = d.merge(inv, on=['city', 'date'], how='left')
print('\nmerged: %s' % d.groupby('city').size().to_dict())

LABELS = [('dT_100', 'T(100m) - T(sfc)'), ('dT_200', 'T(200m) - T(sfc)'),
          ('dT_300', 'T(300m) - T(sfc)'), ('dT_500', 'T(500m) - T(sfc)'),
          ('dT_1000', 'T(1000m) - T(sfc)'),
          ('lapse_0_500', 'lapse rate 0-500 m')]

print('\n' + '=' * 72)
print('UNCONDITIONAL correlation with log PM2.5   (want POSITIVE)')
print('=' * 72)
print('   %-24s %10s %10s' % ('measure', 'shanghai', 'hongkong'))
for col, label in LABELS:
    row = []
    for city in ['shanghai', 'hongkong']:
        s = d[d['city'] == city][[col, 'log_pm25']].dropna()
        row.append(np.corrcoef(s[col], s['log_pm25'])[0, 1]
                   if len(s) > 50 else np.nan)
    print('   %-24s %+10.4f %+10.4f' % (label, row[0], row[1]))

FULL = ['temperature_2m', 'precipitation', 'wind_speed_10m',
        'cloud_cover', 'shortwave_radiation']
# Minimal set omits the physical generators of an inversion. That
# distinction is what flipped the boundary layer sign in Hong Kong, so a
# measure must hold its sign under BOTH sets to be usable.
SETS = [('minimal (T, precip)', ['temperature_2m', 'precipitation']),
        ('full', FULL)]


def first_stage(sub, col, controls):
    sub = sub.dropna(subset=[col, 'log_pm25'] + controls).copy()
    if len(sub) < 80:
        return None
    fe = pd.get_dummies(sub[['month', 'year', 'dow']].astype(str),
                        drop_first=True).astype(float)
    X = sm.add_constant(pd.concat(
        [sub[[col] + controls].reset_index(drop=True),
         fe.reset_index(drop=True)], axis=1))
    y = sub['log_pm25'].reset_index(drop=True)
    full = sm.OLS(y, X).fit(cov_type='HC1')
    red = sm.OLS(y, X.drop(columns=[col])).fit(cov_type='HC1')
    pr2 = (full.rsquared - red.rsquared) / (1 - red.rsquared)
    return full.params[col], full.tvalues[col], pr2, int(full.nobs)


for city in ['shanghai', 'hongkong']:
    print('\n' + '=' * 72)
    print('%s   FIRST STAGE   (want POSITIVE)' % city.upper())
    print('=' * 72)
    for sname, ctrl in SETS:
        print('\n  controls: %s' % sname)
        print('   %-24s %10s %8s %11s %7s %s'
              % ('measure', 'coef', 't', 'partial R2', 'N', 'verdict'))
        sub = d[d['city'] == city]
        for col, label in LABELS:
            r = first_stage(sub, col, ctrl)
            if r is None:
                print('   %-24s %10s' % (label, 'n/a'))
                continue
            b, t, pr2, n = r
            v = ('not significant' if abs(t) < 1.96
                 else 'CORRECT SIGN' if b > 0 else 'wrong sign')
            print('   %-24s %+10.4f %8.2f %11.4f %7d %s'
                  % (label, b, t, pr2, n, v))

out = os.path.join(DATA, 'surface_inversion_panel.csv')
d.to_csv(out, index=False)
print('\nwritten: %s  (%d rows)' % (out, len(d)))

print("""
THE TEST THAT MATTERS
  A measure is usable only if it holds a POSITIVE significant coefficient
  under BOTH control sets. The boundary layer measures failed precisely
  because their sign depended on which controls were included, so
  sign-stability across the two sets is the bar, not significance in one.
""")
