# Source: paper2_meteorology_build.ipynb, code cell 36 (final version of step 22 as run).
# ============================================================
# Step 22 — the filter step 21 should have had
#
# Step 21 flagged A-side corporate actions using the +/-10% daily limit
# and removed suspension gaps, and the maximum premium change was STILL
# 1.0835, a 108% move. H-shares have no daily price limit, so a split,
# rights issue or bad print on the Hong Kong leg passes a filter that
# only watches Shanghai.
#
# Rather than assert a threshold, this prints the distribution of daily
# moves on BOTH legs and the actual worst cases with their components,
# so the cut is chosen from what the data shows.
#
# Also compares a mechanical corporate-action filter against plain
# winsorisation, since both are defensible and the pre-registration has
# to name one as primary.
#
# DISCIPLINE: outcome only. No pollution or meteorological variable.
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

P = pd.read_csv(os.path.join(DATA, 'ah_premium_panel.csv'),
                parse_dates=['date'], dtype={'h_code': str, 'a_code': str})
P['h_code'] = P['h_code'].str.zfill(5)
P = P.sort_values(['h_code', 'date'])

P['d_logA'] = np.log(P['a_raw']).groupby(P['h_code']).diff()
P['d_logH'] = np.log(P['h_raw']).groupby(P['h_code']).diff()
P['gap'] = P.groupby('h_code')['date'].diff().dt.days
print('input: %s firm-days, %d firms' % (format(len(P), ','),
                                         P['h_code'].nunique()))

# ------------------------------------------------------------
# 1. What do daily moves on each leg actually look like?
# ------------------------------------------------------------
print('\n' + '=' * 72)
print('DAILY LOG MOVES BY LEG  (A has a +/-10% limit; H has none)')
print('=' * 72)
qs = [0.50, 0.90, 0.99, 0.999, 0.9999, 1.0]
print('   %-10s %s' % ('quantile', ''.join('%10.4g' % q for q in qs)))
for c, lab in [('d_logA', 'A side'), ('d_logH', 'H side'),
               ('d_prem_raw', 'premium')]:
    s = P[c].abs().dropna()
    print('   %-10s %s' % (lab, ''.join('%10.4f' % s.quantile(q)
                                        for q in qs)))

print('\n   worst 10 premium changes, with their components:')
w = P.dropna(subset=['d_prem_raw']).copy()
w = w.reindex(w['d_prem_raw'].abs().sort_values(ascending=False).index)
print(w.head(10)[['date', 'h_code', 'name', 'a_raw', 'h_raw',
                  'd_logA', 'd_logH', 'd_prem_raw', 'gap']]
      .to_string(index=False, float_format=lambda x: '%.4f' % x))

# ------------------------------------------------------------
# 2. Both-leg filter
# ------------------------------------------------------------
# A: +/-10% price limit is asymmetric in logs (up 0.0953, down -0.10536);
#    0.11 keeps both directions and catches a 15% rights issue at 0.163.
# H: no limit exists, so the cut comes from the distribution above. A
#    35% log move (42% in price) is far beyond anything an H-share does
#    on news and is comfortably below a 2-for-1 split at 0.693.
A_LIMIT, H_LIMIT, MAX_GAP = 0.11, 0.35, 5

# The H threshold is a judgement call, so show the trade-off instead of
# imposing one: a genuine 30% fall on a small-cap H-share is dlog 0.357,
# which 0.35 would wrongly flag as a corporate action.
print('\n   H threshold sensitivity:')
print('      %-10s %9s %12s %12s'
      % ('threshold', 'flagged', '% of rows', 'max abs prem'))
for h in (0.25, 0.30, 0.35, 0.50, 0.70, 1.00):
    fH = P['d_logH'].abs() > h
    keep = ~(P['d_logA'].abs() > A_LIMIT) & ~fH \
        & ~((P['gap'] > MAX_GAP) | P['gap'].isna()) \
        & P['d_prem_raw'].notna()
    mx = P.loc[keep, 'd_prem_raw'].abs().max()
    print('      %-10.2f %9s %11.2f%% %12.4f'
          % (h, format(int(fH.sum()), ','), 100 * fH.mean(), mx))
print('      pick the loosest threshold that still removes the extreme')
print('      tail: that is the one doing corporate-action work only.')

P['f_A'] = P['d_logA'].abs() > A_LIMIT
P['f_H'] = P['d_logH'].abs() > H_LIMIT
P['f_gap'] = (P['gap'] > MAX_GAP) | P['gap'].isna()
P['clean'] = ~(P['f_A'] | P['f_H'] | P['f_gap']) & P['d_prem_raw'].notna()

print('\n' + '=' * 72)
print('FILTERS')
print('=' * 72)
for c, lab in [('f_A', 'A move beyond the 10%% limit (>%.2f)' % A_LIMIT),
               ('f_H', 'H move beyond %.2f in logs' % H_LIMIT),
               ('f_gap', 'gap over %d days or series start' % MAX_GAP)]:
    print('   %-42s %7s  (%.2f%%)'
          % (lab, format(int(P[c].sum()), ','), 100 * P[c].mean()))
print('   %-42s %7s  (%.1f%%)'
      % ('usable', format(int(P['clean'].sum()), ','),
         100 * P['clean'].mean()))

C = P[P['clean']].copy()
print('\n   %-12s %8s %9s %10s %10s %10s'
      % ('', 'n', 'sd', 'p1', 'p99', 'max abs'))
for lab, s in [('before', P['d_prem_raw'].dropna()),
               ('after', C['d_prem_raw'])]:
    print('   %-12s %8s %9.5f %10.4f %10.4f %10.4f'
          % (lab, format(len(s), ','), s.std(), s.quantile(0.01),
             s.quantile(0.99), s.abs().max()))

# ------------------------------------------------------------
# 3. Filter versus winsorisation
# ------------------------------------------------------------
print('\n' + '=' * 72)
print('ALTERNATIVE: winsorise instead of filter')
print('=' * 72)
raw = P.dropna(subset=['d_prem_raw']).copy()
for p in (0.005, 0.01):
    lo, hi = raw['d_prem_raw'].quantile([p, 1 - p])
    wz = raw['d_prem_raw'].clip(lo, hi)
    print('   winsorise at %.1f%%/%.1f%%: sd %.5f  bounds %+.4f %+.4f'
          % (100 * p, 100 * (1 - p), wz.std(), lo, hi))
print('   filter (above):            sd %.5f  max abs %.4f'
      % (C['d_prem_raw'].std(), C['d_prem_raw'].abs().max()))
print("""
   The filter removes observations for a NAMED mechanical reason, which
   is defensible to a referee. Winsorisation keeps every day but
   distorts the tail, and the tail is where a pollution effect would be
   largest if it exists. Register the filter as primary and winsorised
   results as robustness.""")

# ------------------------------------------------------------
# 4. The registered sample, final
# ------------------------------------------------------------
R = C[(C['date'] >= START) & (C['date'] <= END)]
day = R.groupby('date')['d_prem_raw'].agg(['mean', 'var', 'size'])
day = day[day['size'] >= 10]
v_c, v_i = day['mean'].var(), day['var'].mean()

print('\n' + '=' * 72)
print('REGISTERED SAMPLE  %s to %s' % (START, END))
print('=' * 72)
print('   firm-days %s   firms %d   days %d'
      % (format(len(R), ','), R['h_code'].nunique(), len(day)))
print('   common share %.3f   sd one firm-day %.5f   sd daily mean %.5f'
      % (v_c / (v_c + v_i), np.sqrt(v_c + v_i), np.sqrt(v_c)))

try:
    fs = pd.read_csv(os.path.join(DATA,
                                  'shanghai_first_stage_fullyear.csv'),
                     usecols=['date', 'log_pm25'], parse_dates=['date'])
    fs = fs[(fs['date'] >= START) & (fs['date'] <= END)]
    n = len(set(day.index) & set(fs['date']))
    sd_pm, pr2 = fs['log_pm25'].std(), 0.252
    se = np.sqrt(v_c) / (sd_pm * np.sqrt(n) * np.sqrt(pr2))
    print('\n   days overlapping the first stage %d' % n)
    print('   95%% interval per 1 sd of PM2.5   +/- %.1f basis points'
          % (10000 * 1.96 * se * sd_pm))
except Exception as e:
    print('   power estimate skipped: %s' % type(e).__name__)

out = os.path.join(DATA, 'ah_premium_clean.csv')
C.to_csv(out, index=False)
print('\nwritten: %s  (%s usable firm-days)' % (out, format(len(C), ',')))
print('\nIf max abs is now in the 0.2-0.3 range the panel is clean and the')
print('pre-registration can be written. If something above 0.5 survives,')
print('send me the worst-10 table and I will find what it is.')
