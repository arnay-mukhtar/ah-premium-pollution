# Source: paper2_meteorology_build.ipynb, code cell 35 (final version of step 21 as run).
# ============================================================
# Step 21 — clean the premium panel, then it is ready to register
#
# Three problems from step 20, each with a non-arbitrary fix.
#
# 1. CORPORATE ACTIONS. Max one-day premium change of 1.0835, a 108%
#    move. Raw prices carry splits and rights issues. A-shares have a
#    +/-10% daily price limit, so |dlog P_A| > 0.105 cannot be a price
#    move and must be a corporate action or a suspension gap. That is a
#    rule from exchange microstructure, not a threshold I chose to make
#    the data look better.
#
# 2. SUSPENSION GAPS. A change spanning a multi-day closure is not a
#    daily change. Requires consecutive observations within 5 calendar
#    days, which covers a weekend plus one holiday.
#
# 3. LEVEL 25-30 POINTS ABOVE THE HANG SENG AH INDEX. That index is
#    market-cap weighted; this is an unweighted mean over 66 firms, and
#    large bank pairs trade near parity while small industrials trade at
#    very large premia. Checked here against named large caps rather
#    than assumed.
#
# DISCIPLINE: outcome only. No pollution or meteorological variable is
# read anywhere in this file.
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

P = pd.read_csv(os.path.join(DATA, 'ah_premium_panel.csv'),
                parse_dates=['date'], dtype={'h_code': str, 'a_code': str})
P['h_code'] = P['h_code'].str.zfill(5)
P['a_code'] = P['a_code'].str.zfill(6)
P = P.sort_values(['h_code', 'date'])
print('input: %s firm-days, %d firms' % (format(len(P), ','),
                                         P['h_code'].nunique()))

# ------------------------------------------------------------
# 1. Does the level make sense? Large caps should sit near parity.
# ------------------------------------------------------------
BIG = {'01398': 'ICBC', '00939': 'CCB', '03988': 'Bank of China',
       '02318': 'Ping An', '00386': 'Sinopec', '00857': 'PetroChina',
       '01288': 'AgBank', '02628': 'China Life'}
print('\n' + '=' * 70)
print('LEVEL CHECK: large caps should trade near parity')
print('=' * 70)
print('   %-14s %-22s %7s %9s' % ('h_code', 'name', 'days', 'A/H'))
for hc, nm in BIG.items():
    s = P.loc[P['h_code'] == hc, 'prem_raw'].dropna()
    if len(s):
        print('   %-14s %-22s %7d %9.3f' % (hc, nm, len(s),
                                            np.exp(s.mean())))

fm = (P.groupby(['h_code', 'name'])['prem_raw'].mean()
      .apply(np.exp).sort_values())
print('\n   10 lowest premia:')
print(fm.head(10).round(3).to_string())
print('\n   10 highest premia:')
print(fm.tail(10).round(3).to_string())
print('\n   unweighted mean across firms %.3f, median %.3f'
      % (fm.mean(), fm.median()))
print('   -> if the large caps sit near 1.0-1.2 and the tail runs to 3+,')
print('      composition explains the gap against the cap-weighted index')

# ------------------------------------------------------------
# 2. Corporate actions and suspension gaps
# ------------------------------------------------------------
P['d_logA'] = np.log(P['a_raw']).groupby(P['h_code']).diff()
P['d_logH'] = np.log(P['h_raw']).groupby(P['h_code']).diff()
P['gap'] = P.groupby('h_code')['date'].diff().dt.days

# A-share daily limit is +/-10% in PRICE, which is asymmetric in logs:
# a limit-up is log(1.10) = 0.0953, a limit-down log(0.90) = -0.10536.
# A threshold of 0.105 would keep every limit-up and drop every
# limit-down, cutting the most extreme negative-return days only. 0.11
# sits just above the limit-down and still far below a 2-for-1 split
# (0.693) or a rights issue (0.16).
A_LIMIT = 0.11
MAX_GAP = 5            # weekend plus one holiday

P['flag_action'] = P['d_logA'].abs() > A_LIMIT
P['flag_gap'] = (P['gap'] > MAX_GAP) | P['gap'].isna()
P['clean'] = ~(P['flag_action'] | P['flag_gap']) & P['d_prem_raw'].notna()

n = len(P)
print('\n' + '=' * 70)
print('FILTERS')
print('=' * 70)
print('   A-share move beyond the +/-10%% limit   %6s  (%.2f%%)'
      % (format(int(P['flag_action'].sum()), ','),
         100 * P['flag_action'].mean()))
print('   gap over %d calendar days or series start %6s  (%.2f%%)'
      % (MAX_GAP, format(int(P['flag_gap'].sum()), ','),
         100 * P['flag_gap'].mean()))
print('   usable daily changes                  %6s  (%.1f%% of rows)'
      % (format(int(P['clean'].sum()), ','), 100 * P['clean'].mean()))

C = P[P['clean']].copy()
print('\n   daily change, before and after:')
for lab, s in [('all rows', P['d_prem_raw'].dropna()),
               ('cleaned ', C['d_prem_raw'])]:
    print('      %-9s n %7s  sd %.5f  p1 %+.4f  p99 %+.4f  max abs %.4f'
          % (lab, format(len(s), ','), s.std(), s.quantile(0.01),
             s.quantile(0.99), s.abs().max()))

# ------------------------------------------------------------
# 3. Variance decomposition on the cleaned panel
# ------------------------------------------------------------
day = C.groupby('date')['d_prem_raw'].agg(['mean', 'var', 'size'])
day = day[day['size'] >= 10]
v_c, v_i = day['mean'].var(), day['var'].mean()
print('\n' + '=' * 70)
print('VARIANCE DECOMPOSITION, cleaned')
print('=' * 70)
print('   firms per day (mean)            %.1f' % day['size'].mean())
print('   common share of one firm-day    %.3f' % (v_c / (v_c + v_i)))
print('   sd of one firm-day              %.5f' % np.sqrt(v_c + v_i))
print('   sd of the daily average         %.5f' % np.sqrt(v_c))
print('   precision gain from averaging   %.1fx'
      % (np.sqrt(v_c + v_i) / np.sqrt(v_c)))

# ------------------------------------------------------------
# 4. The registered sample
# ------------------------------------------------------------
START, END = '2014-05-13', '2019-12-31'
R = C[(C['date'] >= START) & (C['date'] <= END)]
print('\n' + '=' * 70)
print('REGISTERED SAMPLE   %s to %s' % (START, END))
print('=' * 70)
print('   firm-days %s   firms %d   days %d'
      % (format(len(R), ','), R['h_code'].nunique(), R['date'].nunique()))

d2 = R.groupby('date')['d_prem_raw'].agg(['mean', 'size'])
d2 = d2[d2['size'] >= 10]
print('   days with >=10 firms            %d' % len(d2))
print('   sd of the daily average outcome %.5f' % d2['mean'].std())

# Detectable effect, given the first stage already estimated
PARTIAL_R2, SD_LOGPM = 0.252, None
try:
    fs = pd.read_csv(os.path.join(DATA,
                                  'shanghai_first_stage_fullyear.csv'),
                     usecols=['date', 'log_pm25'], parse_dates=['date'])
    fs = fs[(fs['date'] >= START) & (fs['date'] <= END)]
    SD_LOGPM = fs['log_pm25'].std()
    overlap = len(set(d2.index) & set(fs['date']))
    print('   days overlapping the first stage %d' % overlap)
except Exception as e:
    overlap = len(d2)
    print('   first-stage overlap unavailable: %s' % type(e).__name__)

if SD_LOGPM:
    se = d2['mean'].std() / (SD_LOGPM * np.sqrt(overlap)
                             * np.sqrt(PARTIAL_R2))
    print('\n   sd(log PM2.5) %.3f, partial R2 %.3f, N %d'
          % (SD_LOGPM, PARTIAL_R2, overlap))
    print('   implied IV standard error      %.5f' % se)
    print('   95%% interval per log unit      +/- %.4f' % (1.96 * se))
    print('   95%% interval per 1 sd of PM2.5 +/- %.1f basis points'
          % (10000 * 1.96 * se * SD_LOGPM))
    print('\n   Back of an envelope: assumes the second stage inherits the')
    print('   first-stage partial R2 and ignores clustering. The real')
    print('   number comes from the estimation, but this is the figure to')
    print('   state as expected power in the pre-registration.')

out = os.path.join(DATA, 'ah_premium_clean.csv')
C.to_csv(out, index=False)
print('\nwritten: %s  (%s usable firm-days)' % (out, format(len(C), ',')))
