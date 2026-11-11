# Source: paper2_meteorology_build.ipynb, code cell 51 (final version of step 27 as run).
# ============================================================
# Step 27 — THE REGISTERED ESTIMATION
#
# DO NOT RUN THIS UNTIL THE OSF REGISTRATION IS TIMESTAMPED.
# It is the first thing that joins pollution to the premium. Running it
# beforehand makes the registration worthless, and the paper would have
# to say so.
#
# Everything here implements the pre-registration exactly. No choices
# are made in this file.
#
#   Sample      2014-05-13 to 2019-12-31, Shanghai-listed pairs
#   Treatment   log daily PM2.5, Shanghai, MEP published, 24h mean
#   Instruments log BLH, u, v, surface pressure (24h means)
#   Controls    temperature, precipitation, wind speed, cloud, radiation
#   FE          month, year, day of week
#   Inference   Driscoll-Kraay, Bartlett bandwidth 7
#               Anderson-Rubin sets on the same covariance
#
# Outputs, in registered order:
#   1. Primary coefficient, with AR set and the equivalence test
#   2. Two-leg decomposition (says WHICH market moved)
#   3. Falsification: future pollution, three cities, persistence
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
LAGS = 7                 # Newey-West (1994) automatic value, registered
SESOI_BP = 25.0          # smallest effect of interest, basis points per sd

INST = ['log_blh', 'u', 'v', 'surface_pressure']
CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']

import sys
sys.path.insert(0, DATA)
sys.path.insert(0, '/content')
try:
    import ar_inference as ari
except ImportError:
    raise SystemExit('Put ar_inference.py beside this notebook (or in %s) '
                     'and rerun.' % DATA)


# ------------------------------------------------------------
# Estimator: 2SLS with Driscoll-Kraay covariance
# ------------------------------------------------------------
def _dk(S, lags):
    om = S.T @ S
    for j in range(1, int(lags) + 1):
        if j >= len(S):
            break
        w = 1.0 - j / (lags + 1.0)
        G = S[j:].T @ S[:-j]
        om += w * (G + G.T)
    return om


def tsls(y, x, Z, W, codes, n_clusters, lags=LAGS):
    """Returns (beta, se) for the coefficient on x. W carries no constant."""
    n = len(y)
    Zf = np.column_stack([np.ones(n), Z, W])
    Xf = np.column_stack([x, np.ones(n), W])
    P, *_ = np.linalg.lstsq(Zf, Xf, rcond=None)
    Xh = Zf @ P
    A = Xh.T @ Xf
    b = np.linalg.solve(A, Xh.T @ y)
    e = y - Xf @ b
    S = np.zeros((n_clusters, Xh.shape[1]))
    np.add.at(S, codes, Xh * e[:, None])
    Ai = np.linalg.inv(A)
    V = (Ai @ _dk(S, lags) @ Ai.T
         * (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - Xf.shape[1])))
    return float(b[0]), float(np.sqrt(V[0, 0]))


def design(d, inst_cols, outcome):
    """Build y, x, Z, W, cluster codes from a merged frame."""
    need = inst_cols + CONTROLS + [outcome, 'log_pm25']
    s = d.dropna(subset=need).copy()
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
                   parse_dates=['date'],
                   usecols=['date', 'h_code', 'd_prem_raw', 'd_logA',
                            'd_logH', 'hkd_cny'])
fs = pd.read_csv(os.path.join(DATA, 'shanghai_first_stage_fullyear.csv'),
                 parse_dates=['date'])
three = pd.read_csv(os.path.join(DATA, 'three_city_panel.csv'),
                    parse_dates=['date'])

d = prem.merge(fs, on='date', how='inner')
d = d[(d['date'] >= START) & (d['date'] <= END)].copy()
d = d.sort_values(['h_code', 'date'])
print('estimation sample: %s firm-days, %d firms, %d days'
      % (format(len(d), ','), d['h_code'].nunique(), d['date'].nunique()))

sd_pm = d.groupby('date')['log_pm25'].first().std()
BP = 10000.0 * sd_pm        # coefficient -> basis points per 1 sd
print('sd(log PM2.5) = %.4f, so 1 unit of beta = %.0f bp per sd'
      % (sd_pm, BP))

# ------------------------------------------------------------
# 1. PRIMARY
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('1. PRIMARY   outcome: daily change in the log A-H premium')
print('=' * 74)

y, x, Z, W, codes, G, s = design(d, INST, 'd_prem_raw')
b, se = tsls(y, x, Z, W, codes, G, LAGS)
b_cl, se_cl = tsls(y, x, Z, W, codes, G, 0)

print('   %-26s %12s %12s %8s' % ('', 'beta', 'se', 't'))
print('   %-26s %12.6f %12.6f %8.2f'
      % ('Driscoll-Kraay(7)  REGISTERED', b, se, b / se))
print('   %-26s %12.6f %12.6f %8.2f'
      % ('date clustering', b_cl, se_cl, b_cl / se_cl))
print('\n   effect of a 1 sd rise in PM2.5: %+.2f basis points' % (b * BP))
print('   95%% Wald interval:              [%+.2f, %+.2f] bp'
      % ((b - 1.96 * se) * BP, (b + 1.96 * se) * BP))

print('\n   Anderson-Rubin, same covariance:')
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

print('\n   EQUIVALENCE TEST against +/- %.0f bp (TOST at 5%%):' % SESOI_BP)
if ar90['unbounded'] or ar90.get('empty'):
    print('      inconclusive: the 90%% set is not bounded')
elif (abs(ar90['lo'] * BP) < SESOI_BP) and (abs(ar90['hi'] * BP) < SESOI_BP):
    print('      EQUIVALENT: the 90%% set lies inside the bound, so an effect')
    print('      of practical size is rejected.')
else:
    print('      NOT equivalent: the 90%% set reaches outside the bound, so a')
    print('      practically significant effect cannot be excluded.')

print('\n   READING THE SIGN')
print('      beta < 0  mood         beta > 0  impairment   beta = 0  fundamentals')

# ------------------------------------------------------------
# 2. TWO-LEG DECOMPOSITION
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('2. TWO-LEG DECOMPOSITION   which market actually moved')
print('=' * 74)
print('   %-24s %12s %12s %8s %12s'
      % ('leg', 'beta', 'se', 't', 'bp per sd'))
legs = {}
for col, lab in [('d_logA', 'A leg (Shanghai)'),
                 ('d_logH', 'H leg (Hong Kong)')]:
    yy, xx, ZZ, WW, cc, GG, _ = design(d, INST, col)
    bb, ss = tsls(yy, xx, ZZ, WW, cc, GG, LAGS)
    legs[col] = bb
    print('   %-24s %12.6f %12.6f %8.2f %12.2f'
          % (lab, bb, ss, bb / ss, bb * BP))
print('   %-24s %12.6f %12s %8s %12.2f'
      % ('A minus H', legs['d_logA'] - legs['d_logH'], '', '',
         (legs['d_logA'] - legs['d_logH']) * BP))
print('   %-24s %12.6f %12s %8s %12.2f'
      % ('premium (check)', b, '', '', b * BP))
print("""
   A moves and H does not        -> local investors
   both move together            -> fundamentals
   H moves and A does not        -> the premise is wrong""")

# FX as a third leg: Shanghai air cannot move the HKD/CNY parity.
fxd = d.drop_duplicates('date')[['date', 'hkd_cny']].sort_values('date')
fxd['d_fx'] = np.log(fxd['hkd_cny']).diff()
dfx = fxd.merge(fs, on='date', how='inner').dropna(subset=['d_fx'])
yy, xx, ZZ, WW, cc, GG, _ = design(dfx, INST, 'd_fx')
bb, ss = tsls(yy, xx, ZZ, WW, cc, GG, LAGS)
print('   FX placebo (HKD/CNY): beta %+.6f  t %+.2f   <- must be ~0'
      % (bb, bb / ss))

# ------------------------------------------------------------
# 3. FALSIFICATION
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('3. FALSIFICATION')
print('=' * 74)
print('   %-34s %12s %8s %12s' % ('test', 'beta', 't', 'bp per sd'))

# future pollution: day t+1 treatment on day t outcome
nxt = fs[['date', 'log_pm25'] + INST].copy()
nxt['date'] = nxt['date'] - pd.Timedelta(days=1)
nxt = nxt.rename(columns={c: 'f_' + c for c in ['log_pm25'] + INST})
dfut = (d[['date', 'h_code', 'd_prem_raw'] + CONTROLS + ['month', 'year',
                                                         'dow']]
        .merge(nxt, left_on='date', right_on='date', how='inner'))
dfut = dfut.rename(columns={'f_log_pm25': 'log_pm25'})
F_INST = ['f_' + c for c in INST]
yy, xx, ZZ, WW, cc, GG, _ = design(dfut, F_INST, 'd_prem_raw')
bb, ss = tsls(yy, xx, ZZ, WW, cc, GG, LAGS)
print('   %-34s %12.6f %8.2f %12.2f   <- must be ~0'
      % ('future pollution (t+1)', bb, bb / ss, bb * BP))

# three cities, entered jointly
tc = d[['date', 'h_code', 'd_prem_raw']].merge(three, on='date', how='inner')
for tag in ('beijing', 'urumqi'):
    cols = ['%s_%s' % (tag, c) for c in INST]
    sub = tc.dropna(subset=cols + ['%s_log_pm25' % tag]).copy()
    sub = sub.rename(columns={'%s_log_pm25' % tag: 'log_pm25'})
    for c in CONTROLS:
        sub[c] = sub['%s_%s' % (tag, c)]
    sub['month'] = sub['date'].dt.month
    sub['year'] = sub['date'].dt.year
    sub['dow'] = sub['date'].dt.dayofweek
    yy, xx, ZZ, WW, cc, GG, _ = design(sub, cols, 'd_prem_raw')
    bb, ss = tsls(yy, xx, ZZ, WW, cc, GG, LAGS)
    print('   %-34s %12.6f %8.2f %12.2f'
          % ('%s pollution (own instruments)' % tag, bb, bb / ss, bb * BP))

# persistence
for k in (1, 2):
    lead = d[['date', 'h_code', 'd_prem_raw']].copy()
    lead['date'] = lead['date'] - pd.Timedelta(days=k)
    lead = lead.rename(columns={'d_prem_raw': 'lead_prem'})
    dl = (d.drop(columns=['d_prem_raw'])
          .merge(lead, on=['date', 'h_code'], how='inner'))
    yy, xx, ZZ, WW, cc, GG, _ = design(dl, INST, 'lead_prem')
    bb, ss = tsls(yy, xx, ZZ, WW, cc, GG, LAGS)
    print('   %-34s %12.6f %8.2f %12.2f'
          % ('premium change at t+%d' % k, bb, bb / ss, bb * BP))

print("""
   Future pollution and the two placebo cities must all be near zero.
   Persistence: a mood channel decays at once and may reverse; a
   fundamentals channel persists.

   Romano-Wolf stepdown p-values across this family are computed
   separately, as registered. These are the unadjusted figures.
""")
