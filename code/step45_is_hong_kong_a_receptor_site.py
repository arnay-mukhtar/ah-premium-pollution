# Source: paper2_meteorology_build.ipynb, code cell 86 (final version of step 45 as run).
# ============================================================
# Step 45 — is Hong Kong a receptor site only in WINTER?
#
# THE PUZZLE. Two tables on the same specification disagree:
#
#   step 43, soundings on 1,921 days:  dT_200_100 +0.153, t  3.37
#   step 44, soundings on   811 days:  dT_200_100 -0.020, F  0.16
#
# Same measure, same controls, opposite sign. The difference is the
# sample: step 44 forces the intersection with ERA5 boundary layer
# height, and the Copernicus boundary-layer data was downloaded for
# WINTER ONLY.
#
# THE HYPOTHESIS. Hong Kong is a receptor site in winter and a source
# site the rest of the year. Northeast monsoon days bring continental
# air on clear, dry, strongly mixing days, so dispersion and pollution
# move together and the first-stage sign inverts. In summer the
# monsoon reverses to maritime air and local accumulation governs,
# which is ordinary source-site behaviour.
#
# If true it explains everything at once: why the soundings work across
# the year and fail on the winter-heavy subsample, why ERA5 BLH looked
# uniformly broken (it was only ever available in winter), why Shanghai
# is weakest in winter, and why inversion concentrates below 28N in the
# 365-city screen.
#
# THE TEST. Run the same instrument sets separately on winter and
# non-winter days, on the SOUNDING sample, which does not require the
# winter-only reanalysis field. First print the month composition of
# each sample, so the premise is checked rather than assumed.
#
# WHAT IT DECIDES
#   Correct sign and usable F outside winter, inverted or nil inside
#   it -> Section 6's mechanism is demonstrated, not conjectured, and
#   the claim becomes seasonal rather than absolute.
#   No seasonal split -> the two tables disagree for some other reason
#   and neither can be used until that is found.
# ============================================================

import os
from collections import Counter

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
WINTER = [12, 1, 2]


def read_messy(path):
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
    Wc = np.column_stack([np.ones(len(y)), W])
    cf, *_ = np.linalg.lstsq(Wc, np.column_stack([y, Z]), rcond=None)
    R = np.column_stack([y, Z]) - Wc @ cf
    yt, Zt = R[:, 0], R[:, 1:]
    ZtZ = Zt.T @ Zt
    if np.linalg.cond(ZtZ) > 1e12:
        return None
    pi = np.linalg.solve(ZtZ, Zt.T @ yt)
    e = yt - Zt @ pi
    n, k = Zt.shape
    br = np.linalg.inv(ZtZ)
    V = (br @ (Zt.T @ (Zt * (e ** 2)[:, None])) @ br
         * (n / max(n - k - Wc.shape[1], 1)))
    return (float(pi @ ZtZ @ pi / np.trace(ZtZ @ V)),
            float(1.0 - (e @ e) / (yt @ yt)), pi, n)


# ------------------------------------------------------------
# Build
# ------------------------------------------------------------
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
wcols = ['u', 'v', 'surface_pressure'] + CONTROLS
grp = W.groupby('date')
met = grp[wcols].mean().reset_index()
met['n_hours'] = grp.size().values
met = met[met['n_hours'] >= 8]

B = pd.read_csv(os.path.join(DATA, 'hongkong_blh_hourly.csv'))
B['time'] = pd.to_datetime(B['time'])
bcol = next((c for c in ('blh_station_mean', 'blh_land_mean', 'blh_point',
                         'blh_all_mean') if c in B.columns), None)
B['date'] = B['time'].dt.normalize()
blh = B.groupby('date', as_index=False)[bcol].mean().rename(
    columns={bcol: 'blh'})

snd = pd.read_csv(os.path.join(DATA, 'sounding_daily.csv'),
                  parse_dates=['date'])
if 'city' in snd.columns:
    snd = snd[snd['city'].astype(str).str.lower().str.contains('hong')]
have = [c for c in SND if c in snd.columns]

base = met.merge(hk, on='date', how='inner')
base = base[(base['date'] >= FULL0) & (base['date'] <= FULL1)]
base = base[base['pm25'] > 0].copy()
base['log_pm25'] = np.log(base['pm25'])
base['month'] = base['date'].dt.month
base['year'] = base['date'].dt.year
base['dow'] = base['date'].dt.dayofweek

S = base.merge(snd[['date'] + have], on='date', how='inner').dropna(
    subset=have + CONTROLS + ['log_pm25'])
Bm = base.merge(blh, on='date', how='inner').dropna(
    subset=['blh'] + CONTROLS + ['log_pm25'])
Bm['log_blh'] = np.log(Bm['blh'].clip(lower=1))
C = S.merge(blh, on='date', how='inner').dropna(subset=['blh'])
C['log_blh'] = np.log(C['blh'].clip(lower=1))

# ------------------------------------------------------------
# 1. Month composition — is the premise true?
# ------------------------------------------------------------
print('=' * 74)
print('1. MONTH COMPOSITION OF EACH SAMPLE')
print('=' * 74)
print('   %-26s %6s  %s' % ('sample', 'days',
                            ''.join('%4d' % m for m in range(1, 13))))
for nm, fr in [('soundings', S), ('ERA5 boundary layer', Bm),
               ('both (step 44 sample)', C)]:
    cnt = fr['month'].value_counts()
    print('   %-26s %6d  %s'
          % (nm, len(fr), ''.join('%4d' % cnt.get(m, 0)
                                  for m in range(1, 13))))
wshare = 100.0 * Bm['month'].isin(WINTER).mean()
print('\n   winter (Dec-Feb) share of the ERA5 sample: %.0f%%' % wshare)
print('   winter share of the sounding sample:       %.0f%%'
      % (100.0 * S['month'].isin(WINTER).mean()))
if wshare > 60:
    print('   -> the reanalysis field IS winter-weighted, as suspected')
else:
    print('   -> NOT winter-only; the two tables disagree for another '
          'reason')

# ------------------------------------------------------------
# 2. Winter against the rest, on the sounding sample
# ------------------------------------------------------------
SETS = [('dT 200-100', ['dT_200_100']),
        ('dT 1000-300', ['dT_1000_300']),
        ('registered pair', ['dT_200_100', 'dT_1000_300']),
        ('all four levels', have)]

print('\n' + '=' * 74)
print('2. SOUNDING INSTRUMENTS, WINTER AGAINST THE REST')
print('=' * 74)
print('   positive coefficient = inversion traps pollution = SOURCE site')
print('   negative coefficient = the receptor pattern')
for season, mask in [('full year', pd.Series(True, index=S.index)),
                     ('winter (DJF)', S['month'].isin(WINTER)),
                     ('non-winter', ~S['month'].isin(WINTER))]:
    s = S[mask]
    print('\n   %s, %d days' % (season, len(s)))
    if len(s) < 120:
        print('      too few days')
        continue
    print('      %-20s %6s %9s %11s  %s'
          % ('set', 'k', 'F_eff', 'partial R2', 'coefficients'))
    for nm, cols in SETS:
        cols = [c for c in cols if c in s.columns]
        out = effective_f(s['log_pm25'].values, s[cols].values,
                          s[CONTROLS].values)
        if out is None:
            continue
        F, pr2, pi, n = out
        print('      %-20s %6d %9.2f %11.4f  %s'
              % (nm, len(cols), F, pr2,
                 ' '.join('%+.4f' % p for p in pi)))

# ------------------------------------------------------------
# 3. The same for ERA5 BLH, where the season allows
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('3. ERA5 BOUNDARY LAYER HEIGHT, BY SEASON')
print('=' * 74)
print('   negative coefficient = deeper mixing dilutes = SOURCE site')
for season, mask in [('full ERA5 sample', pd.Series(True, index=Bm.index)),
                     ('winter (DJF)', Bm['month'].isin(WINTER)),
                     ('non-winter', ~Bm['month'].isin(WINTER))]:
    b = Bm[mask]
    if len(b) < 120:
        print('   %-20s %6d days  too few' % (season, len(b)))
        continue
    out = effective_f(b['log_pm25'].values, b[['log_blh']].values,
                      b[CONTROLS].values)
    if out is None:
        continue
    F, pr2, pi, n = out
    print('   %-20s %6d days  F %8.2f  partial R2 %.4f  coef %+.4f'
          % (season, len(b), F, pr2, pi[0]))

print("""
WHAT THIS SETTLES
  Soundings correctly signed with usable F outside winter and inverted
  or nil inside it: Section 6's mechanism is measured rather than
  argued. Hong Kong is a SEASONAL receptor site, the claim becomes
  "uninstrumentable in winter, and the annual instrument is
  contaminated by those months", and the Shanghai winter weakness and
  the sub-28N screen result are the same phenomenon at three scales.

  No seasonal split: the 1,921-day and 811-day results disagree for a
  reason not yet found, and neither can appear in the paper until it is.
""")
