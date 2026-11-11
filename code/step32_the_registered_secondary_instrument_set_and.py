# Source: paper2_meteorology_build.ipynb, code cell 60 (final version of step 32 as run).
# ============================================================
# Step 32 — the registered secondary instrument set, and Table 1
#
# TWO JOBS.
#
# 1. THE SECONDARY INSTRUMENT SET. The registration commits to
#    re-estimating with radiosonde low-level stability in place of
#    reanalysis dispersion:
#
#        dT_200_100    temperature difference between 200m and 100m
#        dT_1000_300   temperature difference between 1000m and 300m
#
#    Both are measured BETWEEN LEVELS ALOFT, so no surface temperature
#    enters the regressor and nothing is collinear with the 2m
#    temperature control. This is the observed atmosphere rather than a
#    model of it, with an entirely different error structure, so
#    agreement between the two instrument sets is evidence the result is
#    not an artefact of ERA5. It is a registered analysis and must appear
#    in the paper whatever it shows.
#
#    Two instruments rather than four, from an 08:00 sounding rather than
#    a 24-hour mean, so expect a weaker first stage and a wider AR set.
#    That is why it is secondary. If the first stage is weak, the AR set
#    is still valid and the Wald interval is not; the script says so.
#
# 2. TABLE 1. Descriptive statistics for the paper.
#
# Both run on the registered sample. Neither changes the primary result.
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
START, END = '2014-05-13', '2019-12-31'
LAGS = 7
SESOI_BP = 25.0

INST = ['log_blh', 'u', 'v', 'surface_pressure']
CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']

import sys
sys.path.insert(0, DATA)
sys.path.insert(0, '/content')
try:
    import ar_inference as ari
except ImportError:
    raise SystemExit('ar_inference.py not found. Put it in %s.' % DATA)


def _dk(S, lags):
    om = S.T @ S
    for j in range(1, int(lags) + 1):
        if j >= len(S):
            break
        w = 1.0 - j / (lags + 1.0)
        G = S[j:].T @ S[:-j]
        om += w * (G + G.T)
    return om


def tsls(y, x, Z, W, codes, G, lags=LAGS):
    n = len(y)
    Zf = np.column_stack([np.ones(n), Z, W])
    Xf = np.column_stack([x, np.ones(n), W])
    P, *_ = np.linalg.lstsq(Zf, Xf, rcond=None)
    Xh = Zf @ P
    A = Xh.T @ Xf
    b = np.linalg.solve(A, Xh.T @ y)
    e = y - Xf @ b
    S = np.zeros((G, Xh.shape[1]))
    np.add.at(S, codes, Xh * e[:, None])
    Ai = np.linalg.inv(A)
    V = (Ai @ _dk(S, lags) @ Ai.T
         * (G / (G - 1.0)) * ((n - 1.0) / (n - Xf.shape[1])))
    return float(b[0]), float(np.sqrt(V[0, 0]))


def effective_f(y, Z, W):
    Wc = np.column_stack([np.ones(len(y)), W])
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
    pr2 = 1.0 - (e @ e) / (yt @ yt)
    return float(pi @ ZtZ @ pi / np.trace(ZtZ @ V)), float(pr2)


def design(frame, inst_cols, outcome):
    need = list(inst_cols) + CONTROLS + [outcome, 'log_pm25']
    s = frame.dropna(subset=need).copy()
    fe = pd.get_dummies(s[['month', 'year', 'dow']].astype(str),
                        drop_first=True).astype(float)
    W = np.column_stack([s[CONTROLS].values, fe.values])
    _, codes = np.unique(s['date'].values, return_inverse=True)
    return (s[outcome].values, s['log_pm25'].values,
            s[inst_cols].values, W, codes, codes.max() + 1, s)


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------
prem = pd.read_csv(os.path.join(DATA, 'ah_premium_clean.csv'),
                   parse_dates=['date'], dtype={'h_code': str},
                   usecols=['date', 'h_code', 'd_prem_raw', 'd_logA',
                            'd_logH'])
fs = pd.read_csv(os.path.join(DATA, 'shanghai_first_stage_fullyear.csv'),
                 parse_dates=['date'])

d = prem.merge(fs, on='date', how='inner')
d = d[(d['date'] >= START) & (d['date'] <= END)].copy()
sd_pm = d.groupby('date')['log_pm25'].first().std()
BP = 10000.0 * sd_pm
print('sample: %s firm-days, %d firms, %d days'
      % (format(len(d), ','), d['h_code'].nunique(), d['date'].nunique()))

# ------------------------------------------------------------
# 1. Secondary instruments
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('1. SECONDARY INSTRUMENT SET   radiosonde low-level stability')
print('=' * 74)

snd_path = os.path.join(DATA, 'sounding_daily.csv')
if not os.path.exists(snd_path):
    print('sounding_daily.csv not found in %s.' % DATA)
    print('Send me the filenames in that folder and I will point at the')
    print('right one. H7 and the primary result are unaffected.')
    snd = None
else:
    snd = pd.read_csv(snd_path, parse_dates=['date'])
    print('columns: %s' % list(snd.columns))
    # The file keys on `city` and holds both soundings. Only the Shanghai
    # one may be used: a Hong Kong sounding cannot instrument Shanghai
    # pollution, and Section 6 is the reason there is no Hong Kong leg.
    key = next((c for c in ('city', 'station', 'site') if c in snd.columns),
               None)
    if key:
        vals = sorted(str(v) for v in snd[key].dropna().unique())
        print('%s values: %s' % (key, vals))
        pick = [v for v in vals
                if any(t in v.upper() for t in
                       ('SHANGHAI', 'BAOSHAN', '58362')) or '上海' in v]
        if not pick:
            raise SystemExit('No Shanghai sounding among %s. Send me this '
                             'list and I will pick the right one.' % vals)
        snd = snd[snd[key].astype(str) == pick[0]]
        print('using %s = %r, %d rows' % (key, pick[0], len(snd)))

SEC = ['dT_200_100', 'dT_1000_300']
if snd is not None:
    have = [c for c in SEC if c in snd.columns]
    if len(have) < 2:
        print('\nexpected %s; found %s' % (SEC, have))
        print('Send me the column list above and I will adjust.')
    else:
        # shanghai_first_stage_fullyear.csv already carries these columns,
        # so merging without dropping them first yields _x/_y suffixes and
        # the names below stop existing. Drop, then take the sounding
        # file's version, which is the one filtered to Shanghai above.
        dup = [c for c in have if c in d.columns]
        if dup:
            print('dropping %s from the panel; taking them from the '
                  'sounding file' % dup)
        base = d.drop(columns=dup) if dup else d
        ds = base.merge(snd[['date'] + have], on='date', how='inner')
        ds = ds.dropna(subset=have)
        print('\nmerged: %s firm-days, %d days with a sounding'
              % (format(len(ds), ','), ds['date'].nunique()))

        y, x, Z, W, codes, G, s = design(ds, have, 'd_prem_raw')
        F, pr2 = effective_f(x, Z, W)
        print('\n   first stage   effective F %.2f   partial R2 %.4f'
              % (F, pr2))
        if F < 10:
            print('   WEAK, as anticipated for two instruments from a')
            print('   single 08:00 sounding. The AR set below is valid;')
            print('   the Wald interval is not and is shown for reference')
            print('   only.')

        b, se = tsls(y, x, Z, W, codes, G, LAGS)
        print('\n   %-30s %12s %12s %8s' % ('', 'beta', 'se', 't'))
        print('   %-30s %12.6f %12.6f %8.2f'
              % ('2SLS, Driscoll-Kraay(7)', b, se, b / se))
        print('   effect of a 1 sd rise in PM2.5: %+.2f bp' % (b * BP))
        print('   95%% Wald (reference only): [%+.2f, %+.2f] bp'
              % ((b - 1.96 * se) * BP, (b + 1.96 * se) * BP))

        print('\n   Anderson-Rubin, Driscoll-Kraay(7):')
        for lv, a in [('95%', 0.05), ('90%', 0.10)]:
            r = ari.ar_confidence_set(y, x, Z, W, codes, alpha=a, lags=LAGS)
            if r['unbounded'] or r.get('empty'):
                print('      %s  %s' % (lv, ari.report(r)))
            else:
                print('      %s  [%+.2f, %+.2f] bp%s'
                      % (lv, r['lo'] * BP, r['hi'] * BP,
                         '   DISJOINT' if r['disjoint'] else ''))
            if a == 0.10:
                ar90 = r

        print('\n   Compare with the primary: +0.73 bp, AR95 [-15.52,')
        print('   +18.07]. Agreement between a modelled and an observed')
        print('   atmosphere is the point of this test. A wider set here')
        print('   is expected and is not a disagreement.')
        if not (ar90['unbounded'] or ar90.get('empty')):
            inside = (abs(ar90['lo'] * BP) < SESOI_BP
                      and abs(ar90['hi'] * BP) < SESOI_BP)
            print('   equivalence against +/- %.0f bp: %s'
                  % (SESOI_BP, 'EQUIVALENT' if inside else 'not equivalent'))

# ------------------------------------------------------------
# 2. Table 1
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('2. TABLE 1   descriptive statistics, registered sample')
print('=' * 74)

daily = d.drop_duplicates('date').set_index('date')
panel = d.set_index(['h_code', 'date'])

rows = [
    ('Outcome (firm-day)', None, None),
    ('  d_prem_raw, bp', d['d_prem_raw'] * 10000, 'firm-day'),
    ('  d_logA, bp', d['d_logA'] * 10000, 'firm-day'),
    ('  d_logH, bp', d['d_logH'] * 10000, 'firm-day'),
    ('Treatment (daily)', None, None),
    ('  PM2.5, ug/m3', np.exp(daily['log_pm25']), 'day'),
    ('  log PM2.5', daily['log_pm25'], 'day'),
    ('Instruments (daily)', None, None),
    ('  log BLH', daily['log_blh'], 'day'),
    ('  u, m/s', daily['u'], 'day'),
    ('  v, m/s', daily['v'], 'day'),
    ('  surface pressure, hPa', daily['surface_pressure'], 'day'),
    ('Controls (daily)', None, None),
    ('  temperature 2m, C', daily['temperature_2m'], 'day'),
    ('  precipitation, mm', daily['precipitation'], 'day'),
    ('  wind speed 10m, m/s', daily['wind_speed_10m'], 'day'),
    ('  cloud cover, %', daily['cloud_cover'], 'day'),
    ('  shortwave rad, W/m2', daily['shortwave_radiation'], 'day'),
]

print('| Variable | N | Mean | SD | p10 | Median | p90 |')
print('| --- | --- | --- | --- | --- | --- | --- |')
for lab, v, unit in rows:
    if v is None:
        print('| **%s** | | | | | | |' % lab.strip())
        continue
    v = pd.Series(v).dropna()
    print('| %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f |'
          % (lab.strip(), format(len(v), ','), v.mean(), v.std(),
             v.quantile(.10), v.median(), v.quantile(.90)))

print('\nfirms %d   days %d   firm-days %s   %s to %s'
      % (d['h_code'].nunique(), d['date'].nunique(), format(len(d), ','),
         d['date'].min().date(), d['date'].max().date()))
print('\nPaste the table above straight into the paper; it is already in')
print('markdown. Send me this whole output and I will fill both gaps.')
