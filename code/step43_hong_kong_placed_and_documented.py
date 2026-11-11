# Source: paper2_meteorology_build.ipynb, code cell 82 (final version of step 43 as run).
# ============================================================
# Step 43 — Hong Kong, placed and documented
#
# TWO JOBS, both needed by Section 6.
#
# A. PLACE HONG KONG IN THE DISTRIBUTION. The screen covers 365
#    mainland cities on a common 2017-2018 window. Hong Kong has never
#    been in it, because its pollution comes from the EPD rather than
#    the mainland archive, so the paper has been discussing it from
#    outside the distribution it belongs in. Run the identical
#    specification and report its percentile.
#
# B. TABLE C, THE ATTEMPTS. Section 6 claims six dispersion measure
#    families crossed with three control sets all fail. Every other
#    claim in the paper carries its numbers; this one carries a
#    sentence, and it is the paper's second contribution. Build the
#    table.
#
# Data, now located:
#   pollution    pollution-ah/hk_pm25_2013_2017.csv, _2018_2022.csv
#   meteorology  data/hongkong_weather_hourly.csv
#   boundary lyr data/hongkong_blh_hourly.csv
#   soundings    data/sounding_daily.csv, city == 'hongkong'
#
# Column names are not assumed. The script prints what it finds and
# stops if it cannot identify a column, rather than guessing and
# producing a table that looks fine and means nothing.
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
DIR = '/content/drive/MyDrive/pollution-ah'
WIN0, WIN1 = '2017-01-01', '2018-12-31'
FULL0, FULL1 = '2014-05-13', '2019-12-31'

SCREEN_INST = ['log_blh', 'u', 'v', 'surface_pressure']
CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']


def pick(df, *keys):
    low = {c.lower().replace(' ', '').replace('_', ''): c
           for c in df.columns}
    for k in keys:
        kk = k.lower().replace(' ', '').replace('_', '')
        if kk in low:
            return low[kk]
    for k in keys:
        kk = k.lower().replace(' ', '').replace('_', '')
        for c in low:
            if kk in c:
                return low[c]
    return None


def read_messy(path):
    """Read a CSV that carries a preamble above its header.

    EPD exports put several lines of notes before the table, so
    pd.read_csv infers one column from line 1 and then fails when the
    real header arrives. This finds the first line whose field count
    matches the modal count of the body and starts there, rather than
    hardcoding a skip that differs between the three files.
    """
    from collections import Counter
    raw = None
    for enc in ('utf-8-sig', 'utf-8', 'gb18030', 'latin-1'):
        try:
            with open(path, 'r', encoding=enc, errors='strict') as fh:
                raw = [fh.readline() for _ in range(80)]
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if raw is None:
        enc = 'latin-1'
        with open(path, 'r', encoding=enc, errors='replace') as fh:
            raw = [fh.readline() for _ in range(80)]
    counts = [ln.count(',') for ln in raw]
    body = [c for c in counts[3:] if c > 0]
    skip = 0
    if body:
        mode = Counter(body).most_common(1)[0][0]
        for i, c in enumerate(counts):
            if c == mode:
                skip = i
                break
    print('      first lines: %r' % [ln.strip()[:60] for ln in raw[:3]])
    print('      modal field count %d, header on line %d'
          % (mode + 1 if body else 1, skip + 1))
    return pd.read_csv(path, skiprows=skip, encoding=enc,
                       low_memory=False)


def ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    n, k = X.shape
    if n - k < 10:
        return None
    XtXi = np.linalg.pinv(X.T @ X)
    V = XtXi @ (X.T @ (X * (e ** 2)[:, None])) @ XtXi * (n / (n - k))
    return b, np.sqrt(np.maximum(np.diag(V), 1e-300)), e, n


# ------------------------------------------------------------
# Pollution
# ------------------------------------------------------------
print('=' * 74)
print('HONG KONG POLLUTION')
print('=' * 74)
parts = []
for f in ('hk_pm25_2013_2017.csv', 'hk_pm25_2018_2022.csv'):
    p = os.path.join(DIR, f)
    if not os.path.exists(p):
        print('   MISSING %s' % p)
        continue
    print('   %s' % f)
    t = read_messy(p)
    print('      %d rows, columns %s' % (len(t), list(t.columns)[:12]))
    parts.append(t)
if not parts:
    raise SystemExit('No Hong Kong pollution files found in %s' % DIR)

P = pd.concat(parts, ignore_index=True)

# EPD layout: DATE, HOUR, POLLUTANT, then ONE COLUMN PER STATION, with
# "N.A." for missing. There is no single value column to find.
dc = pick(P, 'date')
if dc is None:
    raise SystemExit('No DATE column. Columns: %s' % list(P.columns))

if 'POLLUTANT' in P.columns:
    vals = P['POLLUTANT'].astype(str).str.upper().str.replace(' ', '')
    kinds = sorted(vals.unique())
    print('\n   POLLUTANT values: %s' % kinds[:8])
    if len(kinds) > 1:
        keep = vals.str.contains('PM2')
        if not keep.any():
            raise SystemExit('No PM2.5 rows among %s' % kinds)
        P = P[keep]
        print('   kept %d PM2.5 rows' % len(P))

STATION_COLS = [c for c in P.columns
                if c.upper() not in ('DATE', 'HOUR', 'POLLUTANT')]
print('   %d station columns: %s' % (len(STATION_COLS),
                                     STATION_COLS[:6]))

# Dates may be day-first or ISO. Try both and keep whichever parses
# without losses into a plausible window, rather than assuming.
best, best_name = None, None
for name, kw in [('day-first', {'dayfirst': True}),
                 ('month-first', {'dayfirst': False})]:
    try:
        cand = pd.to_datetime(P[dc], errors='coerce', **kw)
    except Exception:
        continue
    ok = cand.notna().mean()
    if ok < 0.95:
        continue
    span = (cand.min(), cand.max())
    plausible = (span[0] >= pd.Timestamp('2010-01-01')
                 and span[1] <= pd.Timestamp('2027-01-01'))
    print('   %s: %.1f%% parsed, %s to %s%s'
          % (name, 100 * ok, span[0].date(), span[1].date(),
             '' if plausible else '  IMPLAUSIBLE'))
    if plausible and best is None:
        best, best_name = cand, name
if best is None:
    raise SystemExit('Could not parse DATE. Sample: %s'
                     % P[dc].head(5).tolist())
print('   using %s parsing' % best_name)
P = P.assign(date=best.dt.normalize())

long = P.melt(id_vars=['date'], value_vars=STATION_COLS,
              var_name='station', value_name='pm25')
long['pm25'] = pd.to_numeric(long['pm25'], errors='coerce')
long = long.dropna(subset=['pm25'])
long = long[long['pm25'] > 0]
print('   station-hour observations: %s' % format(len(long), ','))

hk = long.groupby('date', as_index=False)['pm25'].mean()
print('   daily series: %d days, %s to %s, mean %.1f'
      % (len(hk), hk['date'].min().date(), hk['date'].max().date(),
         hk['pm25'].mean()))

# ------------------------------------------------------------
# Meteorology
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('HONG KONG METEOROLOGY')
print('=' * 74)
W = pd.read_csv(os.path.join(DATA, 'hongkong_weather_hourly.csv'))
print('   weather columns: %s' % list(W.columns)[:14])
tc = pick(W, 'time', 'date', 'datetime')
W[tc] = pd.to_datetime(W[tc], errors='coerce')
W = W.dropna(subset=[tc])

ws = pick(W, 'wind_speed_10m', 'windspeed10m', 'wind_speed')
wd = pick(W, 'wind_direction_10m', 'winddirection10m', 'wind_direction')
if ws and wd:
    rad = np.deg2rad(W[wd])
    W['u'] = -W[ws] * np.sin(rad)
    W['v'] = -W[ws] * np.cos(rad)

blh_col = pick(W, 'boundary_layer_height', 'blh')
if blh_col is None and os.path.exists(
        os.path.join(DATA, 'hongkong_blh_hourly.csv')):
    B = pd.read_csv(os.path.join(DATA, 'hongkong_blh_hourly.csv'))
    print('   blh columns: %s' % list(B.columns)[:10])
    bt = pick(B, 'time', 'date', 'datetime')
    bb = pick(B, 'boundary_layer_height', 'blh')
    if bt and bb:
        B[bt] = pd.to_datetime(B[bt], errors='coerce')
        W = W.merge(B[[bt, bb]].rename(columns={bt: tc, bb: 'blh'}),
                    on=tc, how='left')
        blh_col = 'blh'
if blh_col is None:
    raise SystemExit('No boundary layer height column found for Hong Kong.')

rename = {blh_col: 'blh'}
for want, found in [('temperature_2m', pick(W, 'temperature_2m', 'temp')),
                    ('precipitation', pick(W, 'precipitation', 'precip')),
                    ('wind_speed_10m', ws),
                    ('cloud_cover', pick(W, 'cloud_cover', 'cloudcover')),
                    ('shortwave_radiation',
                     pick(W, 'shortwave_radiation', 'shortwave')),
                    ('surface_pressure',
                     pick(W, 'surface_pressure', 'pressure'))]:
    if found and found != want:
        rename[found] = want
W = W.rename(columns=rename)
need = ['blh', 'u', 'v', 'surface_pressure'] + CONTROLS
missing = [c for c in need if c not in W.columns]
if missing:
    raise SystemExit('Missing Hong Kong meteorology columns: %s\n'
                     'Send me the column lists above.' % missing)

W['date'] = W[tc].dt.normalize()
grp = W.groupby('date')
met = grp[need].mean().reset_index()
met['n_hours'] = grp.size().values
met = met[met['n_hours'] >= 8]
print('   daily meteorology: %d days' % len(met))

d = met.merge(hk, on='date', how='inner')
d = d[d['pm25'] > 0].copy()
d['log_pm25'] = np.log(d['pm25'])
d['log_blh'] = np.log(d['blh'].clip(lower=1))
d['month'] = d['date'].dt.month
d['year'] = d['date'].dt.year
d['dow'] = d['date'].dt.dayofweek
print('   merged: %d days' % len(d))


def fe(s, cols):
    cols = [c for c in cols if s[c].nunique() > 1]
    if not cols:
        return np.zeros((len(s), 0))
    return pd.get_dummies(s[cols].astype(str),
                          drop_first=True).astype(float).values


# ------------------------------------------------------------
# A. Hong Kong in the distribution
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('A. HONG KONG IN THE 365-CITY DISTRIBUTION')
print('=' * 74)
w = d[(d['date'] >= WIN0) & (d['date'] <= WIN1)].dropna(
    subset=SCREEN_INST + CONTROLS + ['log_pm25'])
print('   window days: %d' % len(w))
if len(w) < 300:
    print('   too few days in the common window to place it')
else:
    X = np.column_stack([np.ones(len(w)), w[SCREEN_INST].values,
                         w[CONTROLS].values, fe(w, ['month', 'dow'])])
    out = ols(w['log_pm25'].values, X)
    b, se, e, n = out
    b_hk, t_hk = float(b[1]), float(b[1] / se[1])
    R = pd.read_csv(os.path.join(DATA, 'city_screen_results.csv'))
    pct = 100.0 * (R['b'] < b_hk).mean()
    print('   Hong Kong  log_blh = %+.4f  (t %+.2f), n = %d'
          % (b_hk, t_hk, n))
    print('   percentile among 365 mainland cities: %.0f' % pct)
    print('   inverted (positive and significant)? %s'
          % ('YES' if (b_hk > 0 and t_hk > 1.96) else 'no'))
    print('   national inversion rate 6.0%; rate below 28N 21.8%;')
    print('   predicted for Hong Kong by geography 15.6%')

# ------------------------------------------------------------
# B. Table C
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('B. TABLE C — EVERY DISPERSION MEASURE, THREE CONTROL SETS')
print('=' * 74)
snd = pd.read_csv(os.path.join(DATA, 'sounding_daily.csv'),
                  parse_dates=['date'])
if 'city' in snd.columns:
    snd = snd[snd['city'].astype(str).str.lower().str.contains('hong')]
scols = [c for c in snd.columns if c.startswith('dT_')]
print('   sounding measures available: %s (%d days)' % (scols, len(snd)))
full = d[(d['date'] >= FULL0) & (d['date'] <= FULL1)].merge(
    snd[['date'] + scols], on='date', how='left')

FAMILIES = [('ERA5 boundary layer height', ['log_blh']),
            ('wind components u, v', ['u', 'v']),
            ('surface pressure', ['surface_pressure']),
            ('all four reanalysis instruments', SCREEN_INST)]
for c in scols:
    FAMILIES.append(('radiosonde %s' % c, [c]))
FAMILIES.append(('radiosonde, all levels', scols))

SETS = [('none', []), ('weather controls', CONTROLS),
        ('weather + month/year/dow FE', CONTROLS)]

print('\n   %-32s %-26s %6s %9s %8s'
      % ('measure', 'controls', 'n', 'coef', 't'))
for fname, cols in FAMILIES:
    cols = [c for c in cols if c in full.columns]
    if not cols:
        continue
    for i, (sname, ctrl) in enumerate(SETS):
        s = full.dropna(subset=cols + ctrl + ['log_pm25'])
        if len(s) < 200:
            print('   %-32s %-26s %6d  too few days' % (fname, sname, len(s)))
            continue
        parts = [np.ones(len(s)), s[cols].values]
        if ctrl:
            parts.append(s[ctrl].values)
        if i == 2:
            parts.append(fe(s, ['month', 'year', 'dow']))
        X = np.column_stack(parts)
        out = ols(s['log_pm25'].values, X)
        if out is None:
            continue
        b, se, e, n = out
        print('   %-32s %-26s %6d %9.4f %8.2f'
              % (fname if i == 0 else '', sname, n, b[1], b[1] / se[1]))

print("""
HOW TO READ TABLE C
  The source-site logic requires a NEGATIVE coefficient on boundary
  layer height and on the sounding stability measures. A positive or
  insignificant coefficient across every family and every control set
  is the claim Section 6 makes, and this table is the evidence for it.

  Any family that IS correctly signed and strong must be reported, and
  Section 6 rewritten to say Hong Kong can be instrumented after all by
  that measure. Send me whatever comes back.
""")
