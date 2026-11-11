# Source: paper2_meteorology_build.ipynb, code cell 38 (final version of step 23 as run).
# ============================================================
# Step 23 — excise the split-mismatch windows
#
# Step 22's worst-10 table shows every extreme move comes in PAIRS:
# one leg splits, then the other splits days or weeks later.
#
#   洛阳钼业  H -1.0697 (2015-11-03)  A -1.0582 (2015-11-13)   10 days
#   长城汽车  H -0.9186 (2015-09-30)  A -1.0387 (2015-10-13)   13 days
#   辽港股份  H -0.8019 (2016-07-25)  A -0.8711 (2016-08-01)    7 days
#   金隅集团  H -0.7079 (2016-05-24)  A -0.6536 (2016-07-06)   43 days
#
# exp(-1.07) = 0.343, almost exactly one third: 3-for-1 bonus issues
# with different ex-dates in Shanghai and Hong Kong.
#
# A daily-move filter removes the two ex-dates and leaves every day in
# between, where the premium is wrong by the whole split factor and
# looks perfectly ordinary. Ten contaminated days for 洛阳钼业,
# forty-three for 金隅集团.
#
# This pairs the events and removes the span between them.
#
# DISCIPLINE: outcome only.
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

A_LIMIT = 0.11      # A-share +/-10% limit, asymmetric in logs
H_EVENT = 0.25      # H-side candidate event; deliberately loose, because
                    # pairing (not the threshold) is what identifies a split
MAX_GAP = 5
PAIR_WINDOW = 120   # days within which the second leg must follow
PAIR_TOL = 0.30     # how close the two magnitudes must be, proportionally

P = pd.read_csv(os.path.join(DATA, 'ah_premium_panel.csv'),
                parse_dates=['date'], dtype={'h_code': str, 'a_code': str})
P['h_code'] = P['h_code'].str.zfill(5)
P = P.sort_values(['h_code', 'date']).reset_index(drop=True)
P['d_logA'] = np.log(P['a_raw']).groupby(P['h_code']).diff()
P['d_logH'] = np.log(P['h_raw']).groupby(P['h_code']).diff()
P['gap'] = P.groupby('h_code')['date'].diff().dt.days
print('input: %s firm-days, %d firms' % (format(len(P), ','),
                                         P['h_code'].nunique()))

# ------------------------------------------------------------
# 1. Candidate events on each leg
# ------------------------------------------------------------
ev = []
for hc, g in P.groupby('h_code'):
    for _, r in g.iterrows():
        if pd.notna(r['d_logA']) and abs(r['d_logA']) > A_LIMIT:
            ev.append((hc, r['name'], r['date'], 'A', r['d_logA']))
        if pd.notna(r['d_logH']) and abs(r['d_logH']) > H_EVENT:
            ev.append((hc, r['name'], r['date'], 'H', r['d_logH']))
E = pd.DataFrame(ev, columns=['h_code', 'name', 'date', 'leg', 'dlog'])
print('candidate events: %d on A, %d on H'
      % ((E['leg'] == 'A').sum(), (E['leg'] == 'H').sum()))

# ------------------------------------------------------------
# 2. Pair them: same firm, opposite legs, similar size, same sign
# ------------------------------------------------------------
windows = []
for hc, g in E.groupby('h_code'):
    g = g.sort_values('date').reset_index(drop=True)
    used = set()
    for i in range(len(g)):
        if i in used:
            continue
        a = g.loc[i]
        for j in range(i + 1, len(g)):
            if j in used:
                continue
            b = g.loc[j]
            if (b['date'] - a['date']).days > PAIR_WINDOW:
                break
            # BOTH must be price-REDUCING. A split, bonus issue or rights
            # issue lowers the quoted price; two same-signed POSITIVE moves
            # are two big up-days that happen to match, which is what
            # paired 凯盛新能 across 82 days of the 2015 bubble.
            both_down = (a['dlog'] < 0) and (b['dlog'] < 0)
            close = (abs(abs(a['dlog']) - abs(b['dlog']))
                     / max(abs(a['dlog']), abs(b['dlog'])) < PAIR_TOL)
            if a['leg'] != b['leg'] and both_down and close:
                windows.append({'h_code': hc, 'name': a['name'],
                                'start': a['date'], 'end': b['date'],
                                'days': (b['date'] - a['date']).days,
                                'leg1': a['leg'], 'dlog1': a['dlog'],
                                'leg2': b['leg'], 'dlog2': b['dlog'],
                                'factor': float(np.exp(a['dlog']))})
                used.update({i, j})
                break

W = pd.DataFrame(windows)
print('\n' + '=' * 76)
print('SPLIT-MISMATCH WINDOWS FOUND: %d' % len(W))
print('=' * 76)
if len(W):
    print(W.sort_values('days', ascending=False)
          .to_string(index=False, float_format=lambda x: '%.4f' % x))
    print('\n   total firm-days inside these windows: %s'
          % format(int(W['days'].sum()), ','))
    print('   firms affected: %d of %d' % (W['h_code'].nunique(),
                                           P['h_code'].nunique()))
else:
    print('   none paired. If the worst-10 table still shows pairs, loosen')
    print('   PAIR_TOL or PAIR_WINDOW and rerun.')

# ------------------------------------------------------------
# 3. Excise
# ------------------------------------------------------------
P['in_window'] = False
for _, w in W.iterrows():
    m = ((P['h_code'] == w['h_code']) & (P['date'] >= w['start'])
         & (P['date'] <= w['end']))
    P.loc[m, 'in_window'] = True
print('\nrows inside a split window: %s (%.2f%%)'
      % (format(int(P['in_window'].sum()), ','),
         100 * P['in_window'].mean()))

P['f_A'] = P['d_logA'].abs() > A_LIMIT
P['f_H'] = P['d_logH'].abs() > H_EVENT
P['f_gap'] = (P['gap'] > MAX_GAP) | P['gap'].isna()
P['clean'] = (~(P['f_A'] | P['f_H'] | P['f_gap'] | P['in_window'])
              & P['d_prem_raw'].notna())

C = P[P['clean']].copy()
print('usable: %s (%.1f%%)' % (format(len(C), ','), 100 * P['clean'].mean()))

print('\n   %-22s %8s %9s %10s %10s'
      % ('', 'n', 'sd', 'p99', 'max abs'))
stages = [('raw', P['d_prem_raw'].dropna()),
          ('events removed',
           P.loc[~(P['f_A'] | P['f_H'] | P['f_gap'])
                 & P['d_prem_raw'].notna(), 'd_prem_raw']),
          ('+ windows removed', C['d_prem_raw'])]
for lab, s in stages:
    print('   %-22s %8s %9.5f %10.4f %10.4f'
          % (lab, format(len(s), ','), s.std(), s.quantile(0.99),
             s.abs().max()))

# ------------------------------------------------------------
# 4. Per-firm sanity: no premium should swing by a split factor
# ------------------------------------------------------------
rng_ = C.groupby(['h_code', 'name'])['prem_raw'].agg(['min', 'max'])
rng_['swing'] = np.exp(rng_['max'] - rng_['min'])
print('\n   widest premium swings remaining (max/min within firm):')
print(rng_.sort_values('swing', ascending=False).head(8)
      .round(3).to_string())
print('   a swing near 2, 3 or 10 suggests an unpaired split still present')

# ------------------------------------------------------------
# 5. Registered sample
# ------------------------------------------------------------
R = C[(C['date'] >= START) & (C['date'] <= END)]
day = R.groupby('date')['d_prem_raw'].agg(['mean', 'var', 'size'])
day = day[day['size'] >= 10]
v_c, v_i = day['mean'].var(), day['var'].mean()
print('\n' + '=' * 76)
print('REGISTERED SAMPLE  %s to %s' % (START, END))
print('=' * 76)
print('   firm-days %s   firms %d   days %d'
      % (format(len(R), ','), R['h_code'].nunique(), len(day)))
print('   common share %.3f   sd daily mean %.5f'
      % (v_c / (v_c + v_i), np.sqrt(v_c)))

try:
    fs = pd.read_csv(os.path.join(DATA,
                                  'shanghai_first_stage_fullyear.csv'),
                     usecols=['date', 'log_pm25'], parse_dates=['date'])
    fs = fs[(fs['date'] >= START) & (fs['date'] <= END)]
    n = len(set(day.index) & set(fs['date']))
    sd_pm = fs['log_pm25'].std()
    se = np.sqrt(v_c) / (sd_pm * np.sqrt(n) * np.sqrt(0.252))
    print('   days overlapping the first stage %d' % n)
    print('   95%% interval per 1 sd of PM2.5   +/- %.1f basis points'
          % (10000 * 1.96 * se * sd_pm))
except Exception as e:
    print('   power estimate skipped: %s' % type(e).__name__)

C.to_csv(os.path.join(DATA, 'ah_premium_clean.csv'), index=False)
if len(W):
    W.to_csv(os.path.join(DATA, 'split_windows.csv'), index=False)
    print('\nwritten: ah_premium_clean.csv and split_windows.csv')
print('\nIf max abs is now under about 0.25 the outcome is clean and the')
print('pre-registration can be written.')
