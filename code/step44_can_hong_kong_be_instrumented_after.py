# Source: paper2_meteorology_build.ipynb, code cell 84 (final version of step 44 as run).
# ============================================================
# Step 44 — can Hong Kong be instrumented after all?
#
# Table C, read with the correct sign convention, changed the picture.
# dT_200_100 is T(200m) - T(100m): POSITIVE means an inversion, which
# traps pollution, so the source-site prediction for a stability
# measure is a POSITIVE coefficient. My note on step 43 said negative,
# which is backwards, and under the right convention the radiosonde
# measures are correctly signed and significant in every control set.
#
# Section 6 claims nothing survives. That is no longer supportable
# without this check, because:
#
#   ERA5 boundary layer height  wrong sign under controls (+0.30, t 3.79)
#   radiosonde dT 200-100 m     RIGHT sign, t 3.37 with weather controls
#
# A single-instrument t of 3.37 is an F near 11. The question Section 6
# turns on is whether a usable first stage exists, and that needs the
# JOINT effective F, not the t on one coefficient, which is all step 43
# printed for the multi-measure rows.
#
# This computes, for each candidate instrument set:
#   Montiel Olea-Pflueger effective F, the same statistic the mainland
#   design reports, partial R-squared, and the individual coefficients
#   with their signs, on a common sample so the sets are comparable.
#
# WHAT IT DECIDES
#   F comfortably above 10 for a correctly signed set -> Hong Kong IS
#   instrumentable by observed soundings, Section 6 is rewritten around
#   retrieval failure rather than uninstrumentability, and the paper
#   must say whether the two-sided design is recoverable.
#   F below 10 everywhere -> the original claim stands in weakened form:
#   no usable instrument was found, though the sign evidence is mixed
#   rather than uniformly negative.
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
FULL0, FULL1 = '2014-05-13', '2019-12-31'
CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']
SND = ['dT_200_100', 'dT_300_100', 'dT_500_100', 'dT_1000_300']


def read_messy(path):
    from collections import Counter
    for enc in ('utf-8-sig', 'utf-8', 'gb18030', 'latin-1'):
        try:
            with open(path, 'r', encoding=enc, errors='strict') as fh:
                raw = [fh.readline() for _ in range(80)]
            break
        except (UnicodeDecodeError, LookupError):
            continue
    counts = [ln.count(',') for ln in raw]
    body = [c for c in counts[3:] if c > 0]
    skip = 0
    if body:
        mode = Counter(body).most_common(1)[0][0]
        skip = next(i for i, c in enumerate(counts) if c == mode)
    return pd.read_csv(path, skiprows=skip, encoding=enc, low_memory=False)


def effective_f(y, Z, W):
    """Montiel Olea-Pflueger effective F, plus partial R-squared."""
    Wc = np.column_stack([np.ones(len(y)), W]) if W.shape[1] else \
        np.ones((len(y), 1))
    cf, *_ = np.linalg.lstsq(Wc, np.column_stack([y, Z]), rcond=None)
    R = np.column_stack([y, Z]) - Wc @ cf
    yt, Zt = R[:, 0], R[:, 1:]
    ZtZ = Zt.T @ Zt
    pi = np.linalg.solve(ZtZ, Zt.T @ yt)
    e = yt - Zt @ pi
    n, k = Zt.shape
    br = np.linalg.inv(ZtZ)
    V = (br @ (Zt.T @ (Zt * (e ** 2)[:, None])) @ br
         * (n / max(n - k - Wc.shape[1], 1)))
    return (float(pi @ ZtZ @ pi / np.trace(ZtZ @ V)),
            float(1.0 - (e @ e) / (yt @ yt)), pi, n)


# ------------------------------------------------------------
# Rebuild the Hong Kong daily panel
# ------------------------------------------------------------
parts = [read_messy(os.path.join(DIR, f))
         for f in ('hk_pm25_2013_2017.csv', 'hk_pm25_2018_2022.csv')
         if os.path.exists(os.path.join(DIR, f))]
P = pd.concat(parts, ignore_index=True)
STATION = [c for c in P.columns
           if c.upper() not in ('DATE', 'HOUR', 'POLLUTANT')]
P = P.assign(date=pd.to_datetime(P['DATE'], dayfirst=True,
                                 errors='coerce').dt.normalize())
long = P.melt(id_vars=['date'], value_vars=STATION, value_name='pm25')
long['pm25'] = pd.to_numeric(long['pm25'], errors='coerce')
long = long.dropna(subset=['pm25', 'date'])
hk = long[long['pm25'] > 0].groupby('date', as_index=False)['pm25'].mean()

W = pd.read_csv(os.path.join(DATA, 'hongkong_weather_hourly.csv'))
W['time'] = pd.to_datetime(W['time'])
B = pd.read_csv(os.path.join(DATA, 'hongkong_blh_hourly.csv'))
B['time'] = pd.to_datetime(B['time'])
bcol = next((c for c in ('blh_station_mean', 'blh_land_mean', 'blh_point',
                         'blh_all_mean') if c in B.columns), None)
W = W.merge(B[['time', bcol]].rename(columns={bcol: 'blh'}), on='time',
            how='left')
W['date'] = W['time'].dt.normalize()
cols = ['blh', 'u', 'v', 'surface_pressure'] + CONTROLS
grp = W.groupby('date')
met = grp[cols].mean().reset_index()
met['n_hours'] = grp.size().values
met = met[met['n_hours'] >= 8]

snd = pd.read_csv(os.path.join(DATA, 'sounding_daily.csv'),
                  parse_dates=['date'])
if 'city' in snd.columns:
    snd = snd[snd['city'].astype(str).str.lower().str.contains('hong')]

d = met.merge(hk, on='date', how='inner').merge(
    snd[['date'] + [c for c in SND if c in snd.columns]], on='date',
    how='inner')
d = d[(d['date'] >= FULL0) & (d['date'] <= FULL1)]
d = d[d['pm25'] > 0].copy()
d['log_pm25'] = np.log(d['pm25'])
d['log_blh'] = np.log(d['blh'].clip(lower=1))
d['month'] = d['date'].dt.month
d['year'] = d['date'].dt.year
d['dow'] = d['date'].dt.dayofweek

have = [c for c in SND if c in d.columns]
d = d.dropna(subset=['log_pm25', 'log_blh'] + CONTROLS + have)
print('Hong Kong common sample: %d days, %s to %s'
      % (len(d), d['date'].min().date(), d['date'].max().date()))
print('(common to pollution, reanalysis AND soundings, so every set '
      'below\nruns on the same days and the F figures are comparable)')


def fe(s):
    return pd.get_dummies(s[['month', 'year', 'dow']].astype(str),
                          drop_first=True).astype(float).values


SETS = [
    ('ERA5 log BLH', ['log_blh'], 'negative'),
    ('ERA5 four instruments', ['log_blh', 'u', 'v', 'surface_pressure'],
     'mixed'),
    ('sounding dT 200-100', ['dT_200_100'], 'positive'),
    ('sounding dT 1000-300', ['dT_1000_300'], 'positive'),
    ('sounding, all four levels', have, 'positive'),
    ('registered secondary pair', ['dT_200_100', 'dT_1000_300'],
     'positive'),
    ('soundings + ERA5 BLH', have + ['log_blh'], 'mixed'),
]

for ctrl_name, use_fe in [('weather controls', False),
                          ('weather controls + month/year/dow FE', True)]:
    Wm = d[CONTROLS].values
    if use_fe:
        Wm = np.column_stack([Wm, fe(d)])
    print('\n' + '=' * 74)
    print('CONTROLS: %s' % ctrl_name)
    print('=' * 74)
    print('   %-30s %6s %9s %11s  %s'
          % ('instrument set', 'k', 'F_eff', 'partial R2', 'coefficients'))
    for name, cols_, expect in SETS:
        cols_ = [c for c in cols_ if c in d.columns]
        if not cols_:
            continue
        F, pr2, pi, n = effective_f(d['log_pm25'].values,
                                    d[cols_].values, Wm)
        sign_ok = {'positive': all(p > 0 for p in pi),
                   'negative': all(p < 0 for p in pi),
                   'mixed': None}[expect]
        flag = '' if sign_ok is None else ('  sign OK' if sign_ok
                                           else '  SIGN WRONG')
        print('   %-30s %6d %9.2f %11.4f  %s%s'
              % (name, len(cols_), F, pr2,
                 ' '.join('%+.4f' % p for p in pi), flag))

print("""
READING IT
  Compare against the Shanghai figures the paper already reports:
  effective F 106.89 with partial R-squared 0.2490 on the four
  reanalysis instruments.

  Any Hong Kong set with the right sign and F above 10 means the
  "uninstrumentable" claim must go. The honest replacement is that
  boundary-layer RETRIEVAL fails here while direct thermodynamic
  stability works, which is a sharper and more useful finding.

  Even then the exclusion restriction is a separate question. Low-level
  stability aloft is not directly experienced, which is the argument for
  it, but it still correlates with the clear, calm synoptic regime that
  carries the mood channel. Strength would make a Hong Kong leg
  possible; it would not make it clean.
""")
