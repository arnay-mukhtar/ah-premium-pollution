# Source: paper2_meteorology_build.ipynb, code cell 55 (final version of step 30 as run).
# ============================================================
# Step 30 — H7, the registered within-day announcement interaction
#
# The primary test is identified from time variation alone: one treatment
# value per day, 1,311 days, and every confound is a day-level confound
# that has to be argued away with controls and placebo cities.
#
# H7 is identified from WITHIN-DAY variation. Two firms on the same day
# differ in whether they have just reported earnings, so the model can
# carry DAY fixed effects, which absorb the market, the macro news, the
# weather, the sentiment and everything else common to a trading day. The
# pollution main effect is absorbed along with them, by construction, and
# what remains is the differential response of announcement-window firms.
#
#   d_prem(i,t) = delta * [ log_pm25(t) x D(i,t) ]
#                 + theta * D(i,t) + day_FE(t) + firm_FE(i) + e(i,t)
#
#   D(i,t) = 1 on firm i's announcement day and the following trading day
#   instruments: Z(t) x D(i,t), four of them, for the one interaction
#
# D is exogenous because reporting dates are scheduled and disclosed in
# advance and cannot respond to the weather, so the exclusion restriction
# is inherited from the primary specification rather than added to it.
# Seasonal clustering of Chinese filings would be a confound in a time
# series and is absorbed entirely by day_FE here.
#
# WHAT THE SIGN MEANS, given the contemporaneous null:
#   delta indistinguishable from zero  -> H7b / H7c. The headline null
#       holds uniformly, including where processing load is high. This is
#       the strongest version of the fundamentals reading.
#   delta materially non-zero          -> H7a. Announcement-window firms
#       respond while others do not, which is cognitive impairment
#       concentrated where judgement is actually being exercised, and it
#       QUALIFIES the headline null rather than contradicting it.
#
# About 3.5% of firm-days are treated, so the interval will be wide. A
# failure to reject is reported as an interval, never as support for the
# null, exactly as registered.
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


# ------------------------------------------------------------
# Covariance helpers
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


def _meat(Xh, e, codes, G, lags=0):
    S = np.zeros((G, Xh.shape[1]))
    np.add.at(S, codes, Xh * e[:, None])
    return _dk(S, lags) if lags else S.T @ S


def tsls(y, x, Z, W, groups, kind='dk'):
    """2SLS of y on [x, W] using [Z, W]. Returns beta on x and its se.

    groups is a dict of cluster-code arrays. kind picks the covariance:
      'dk'   Driscoll-Kraay(7) over days, as registered
      'firm' clustered by firm
      'two'  two-way, Cameron-Gelbach-Miller
    No constant: the day fixed effects are removed by the within
    transformation before anything reaches this function.
    """
    n = len(y)
    Zf = np.column_stack([Z, W])
    Xf = np.column_stack([x, W])
    P, *_ = np.linalg.lstsq(Zf, Xf, rcond=None)
    Xh = Zf @ P
    A = Xh.T @ Xf
    b = np.linalg.solve(A, Xh.T @ y)
    e = y - Xf @ b
    Ai = np.linalg.inv(A)
    k = Xf.shape[1]

    def sandwich(codes, G, lags):
        adj = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
        return Ai @ _meat(Xh, e, codes, G, lags) @ Ai.T * adj

    day, Gd = groups['day']
    firm, Gf = groups['firm']
    if kind == 'dk':
        V = sandwich(day, Gd, LAGS)
    elif kind == 'firm':
        V = sandwich(firm, Gf, 0)
    else:
        both, Gb = groups['both']
        V = (sandwich(day, Gd, 0) + sandwich(firm, Gf, 0)
             - sandwich(both, Gb, 0))
    return float(b[0]), float(np.sqrt(max(V[0, 0], 1e-300)))


def demean_by(M, codes, G):
    """Within transformation: subtract the group mean of every column."""
    M = np.asarray(M, float)
    one = M.ndim == 1
    if one:
        M = M[:, None]
    sums = np.zeros((G, M.shape[1]))
    np.add.at(sums, codes, M)
    cnt = np.bincount(codes, minlength=G).astype(float)[:, None]
    out = M - (sums / cnt)[codes]
    return out[:, 0] if one else out


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------
prem = pd.read_csv(os.path.join(DATA, 'ah_premium_clean.csv'),
                   parse_dates=['date'],
                   dtype={'h_code': str, 'a_code': str},
                   usecols=['date', 'h_code', 'a_code', 'd_prem_raw'])
fs = pd.read_csv(os.path.join(DATA, 'shanghai_first_stage_fullyear.csv'),
                 parse_dates=['date'])
ann = pd.read_csv(os.path.join(DATA, 'announcement_dates.csv'),
                  parse_dates=['ann_date'], dtype={'a_code': str,
                                                   'h_code': str})

d = prem.merge(fs, on='date', how='inner')
d = d[(d['date'] >= START) & (d['date'] <= END)]
d = d.dropna(subset=['d_prem_raw', 'log_pm25'] + INST + CONTROLS).copy()
d = d.sort_values(['date', 'h_code']).reset_index(drop=True)
d['a_code'] = d['a_code'].str.zfill(6)
ann['a_code'] = ann['a_code'].astype(str).str.zfill(6)

sd_pm = d.groupby('date')['log_pm25'].first().std()
BP = 10000.0 * sd_pm
print('panel: %s firm-days, %d firms, %d days'
      % (format(len(d), ','), d['h_code'].nunique(), d['date'].nunique()))

# ------------------------------------------------------------
# Build D(i,t)
#
# An announcement filed on a non-trading day takes effect on the next
# trading day, so the window is anchored to the first trading day on or
# after the filing date, then extended one trading day. This is the
# registered "announcement day and the following trading day", made
# explicit for weekend filings.
# ------------------------------------------------------------
# Keep the calendar in datetime64[ns]. np.array(sorted(...)) over a
# pandas unique() yields an OBJECT array of Timestamps, and searchsorted
# then compares a Timestamp against a raw integer and raises.
cal = pd.DatetimeIndex(sorted(pd.unique(d['date']))).values.astype(
    'datetime64[ns]')
ann_vals = pd.DatetimeIndex(ann['ann_date']).values.astype('datetime64[ns]')

idx = np.searchsorted(cal, ann_vals, side='left')
ok = idx < len(cal)
ev = ann.loc[ok, ['a_code']].copy()
ev['i0'] = idx[ok]
rows = []
for off in (0, 1):
    e = ev.copy()
    e['i'] = e['i0'] + off
    rows.append(e[e['i'] < len(cal)][['a_code', 'i']])
win = pd.concat(rows).drop_duplicates()
win['date'] = cal[win['i'].values]
win['D'] = 1.0

d = d.merge(win[['a_code', 'date', 'D']], on=['a_code', 'date'], how='left')
d['D'] = d['D'].fillna(0.0)

n_treat = int(d['D'].sum())
print('announcement window: %s firm-days treated (%.2f%% of the panel), '
      '%d firms'
      % (format(n_treat, ','), 100.0 * n_treat / len(d),
         d.loc[d['D'] > 0, 'h_code'].nunique()))

# Days that contain BOTH a treated and an untreated firm are the only days
# that contribute anything once day fixed effects are in. Reported because
# it is the real sample size of this test.
by_day = d.groupby('date')['D'].agg(['sum', 'size'])
usable = by_day[(by_day['sum'] > 0) & (by_day['sum'] < by_day['size'])]
print('days with within-day contrast: %d of %d'
      % (len(usable), d['date'].nunique()))
if len(usable) < 50:
    raise SystemExit('Too few days carry a within-day contrast. H7 is not '
                     'identified on this sample; report it as NOT RUN.')

# ------------------------------------------------------------
# Within transformation and design
# ------------------------------------------------------------
_, day = np.unique(d['date'].values, return_inverse=True)
_, firm = np.unique(d['h_code'].values, return_inverse=True)
Gd, Gf = day.max() + 1, firm.max() + 1
both = day * Gf + firm
_, both = np.unique(both, return_inverse=True)
groups = {'day': (day, Gd), 'firm': (firm, Gf),
          'both': (both, both.max() + 1)}

D = d['D'].values
x_raw = d['log_pm25'].values * D                 # endogenous interaction
Z_raw = d[INST].values * D[:, None]              # interacted instruments
firm_d = pd.get_dummies(d['h_code'], drop_first=True).values.astype(float)

# Day fixed effects are removed exactly by day-demeaning; the pollution
# main effect and every daily control vanish with them, which is the
# design. Firm dummies stay in explicitly after demeaning.
y = demean_by(d['d_prem_raw'].values, day, Gd)
x = demean_by(x_raw, day, Gd)
Z = demean_by(Z_raw, day, Gd)
W = demean_by(np.column_stack([D, firm_d]), day, Gd)

# ------------------------------------------------------------
# First stage for the interaction
# ------------------------------------------------------------
Wc = W
cf, *_ = np.linalg.lstsq(Wc, np.column_stack([x, Z]), rcond=None)
R = np.column_stack([x, Z]) - Wc @ cf
xt, Zt = R[:, 0], R[:, 1:]
ZtZ = Zt.T @ Zt
pi = np.linalg.solve(ZtZ, Zt.T @ xt)
res = xt - Zt @ pi
nn, kk = Zt.shape
bread = np.linalg.inv(ZtZ)
Vp = (bread @ (Zt.T @ (Zt * (res ** 2)[:, None])) @ bread
      * (nn / max(nn - kk - Wc.shape[1], 1)))
F_eff = float(pi @ ZtZ @ pi / np.trace(ZtZ @ Vp))
pr2 = 1.0 - (res @ res) / (xt @ xt)

print('\n' + '=' * 74)
print('H7 FIRST STAGE   Z x D predicting log_pm25 x D, within day')
print('=' * 74)
print('   effective F  %.2f        partial R2  %.4f' % (F_eff, pr2))
if F_eff < 10:
    print('   WEAK. The AR set below is still valid; the Wald interval is')
    print('   not, and should not be the headline.')

# ------------------------------------------------------------
# H7
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('H7   delta on [ log PM2.5 x announcement window ]')
print('=' * 74)
print('   %-34s %12s %12s %8s' % ('covariance', 'delta', 'se', 't'))
for kind, lab in [('dk', 'Driscoll-Kraay(7)  REGISTERED'),
                  ('firm', 'clustered by firm'),
                  ('two', 'two-way, firm and day')]:
    bb, ss = tsls(y, x, Z, W, groups, kind)
    print('   %-34s %12.6f %12.6f %8.2f' % (lab, bb, ss, bb / ss))
    if kind == 'dk':
        delta, se = bb, ss

print('\n   differential effect of a 1 sd rise in PM2.5 for a firm in its')
print('   announcement window: %+.2f basis points' % (delta * BP))
print('   95%% Wald interval: [%+.2f, %+.2f] bp'
      % ((delta - 1.96 * se) * BP, (delta + 1.96 * se) * BP))

print('\n   Anderson-Rubin, Driscoll-Kraay(7):')
for lv, a in [('95%', 0.05), ('90%', 0.10)]:
    r = ari.ar_confidence_set(y, x, Z, W, day, alpha=a, lags=LAGS)
    if r['unbounded'] or r.get('empty'):
        print('      %s  %s' % (lv, ari.report(r)))
    else:
        print('      %s  [%+.2f, %+.2f] bp%s'
              % (lv, r['lo'] * BP, r['hi'] * BP,
                 '   DISJOINT' if r['disjoint'] else ''))
    if a == 0.10:
        ar90 = r

print('\n   EQUIVALENCE against +/- %.0f bp (TOST at 5%%):' % SESOI_BP)
if ar90['unbounded'] or ar90.get('empty'):
    print('      INCONCLUSIVE. The 90% set is not bounded, so this test is')
    print('      uninformative and is reported as such, not as a null.')
elif abs(ar90['lo'] * BP) < SESOI_BP and abs(ar90['hi'] * BP) < SESOI_BP:
    print('      EQUIVALENT. No differential of practical size, so the')
    print('      headline null holds even where processing load is high.')
    print('      This is H7b / H7c and it strengthens the main result.')
else:
    print('      NOT equivalent. A differential of practical size cannot be')
    print('      excluded, so H7 is inconclusive at this power and must be')
    print('      reported as an interval rather than as evidence for zero.')

print("""
READING IT
  A delta near zero with a BOUNDED interval inside the SESOI is the
  strongest available version of the fundamentals result: no response
  even among the firms whose investors were demonstrably processing
  information that day.

  A delta near zero with a WIDE interval says only that 3.5% of the panel
  cannot resolve this. Report the interval and say so. Do not let a wide
  interval be written up as agreement with the main result.

  A materially non-zero delta is the interesting outcome and must be
  reported as such even though it complicates the headline.
""")
