# Source: paper2_meteorology_build.ipynb, code cell 34 (final version of step 20 as run).
# ============================================================
# Step 20 — rebuild the A-H premium correctly
#
# TWO PROBLEMS with ah_panel.csv as built.
#
# 1. CURRENCY. It computes h_close / a_close with no conversion: a Hong
#    Kong dollar price divided by a renminbi price. HKD/CNY ranged from
#    0.786 to 0.920 over the sample, a 17% swing sitting inside the
#    outcome as spurious price movement.
#
# 2. PRICE ADJUSTMENT. Both sides use adjust="hfq", back-adjusted. The
#    A-H premium is a RELATIVE PRICE: what an investor pays in Shanghai
#    against what they pay in Hong Kong for the same claim. That is raw
#    prices in a common currency, which is how the Hang Seng AH Premium
#    Index is defined and what this literature reports. Back-adjusted
#    series fold each line's own dividend history into the ratio.
#    You already have raw H prices (h_prices.csv). Raw A prices were
#    never pulled, so this pulls them.
#
#    Both versions are built and compared, because the hfq version has
#    one genuine advantage: it removes ex-dividend jumps from the daily
#    change. The comparison decides which is primary.
#
# Also reports the variance decomposition of the daily premium change
# into common and idiosyncratic parts. That determines how much the 60
# firms per day buy in precision, which the pre-registration needs in
# order to state detectable effect sizes.
#
# DISCIPLINE: builds and describes the OUTCOME only. No pollution,
# meteorological or instrument variable is loaded anywhere in this file.
# ============================================================

import os
import time

import numpy as np
import pandas as pd

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DIR = '/content/drive/MyDrive/pollution-ah'
DATA = '/content/drive/MyDrive/bilateral/data'
RAW_A = os.path.join(DATA, 'a_prices_raw.csv')

try:
    import akshare as ak
except ImportError:
    os.system('pip install -q akshare')
    import akshare as ak

pairs = pd.read_csv(os.path.join(DIR, 'ah_pairs.csv'),
                    dtype={'h_code': str, 'a_code': str})
pairs['a_code'] = pairs['a_code'].str.zfill(6)
pairs['h_code'] = pairs['h_code'].str.zfill(5)
sh = pairs[pairs['a_code'].str.startswith('6')].copy()
print('Shanghai-listed pairs: %d (of %d)' % (len(sh), len(pairs)))

# ------------------------------------------------------------
# 1. Raw A prices
# ------------------------------------------------------------
if os.path.exists(RAW_A):
    A = pd.read_csv(RAW_A, parse_dates=['date'], dtype={'a_code': str})
    A['a_code'] = A['a_code'].str.zfill(6)
    print('raw A prices cached: %d series' % A['a_code'].nunique())
else:
    rows, failed = [], []
    for i, ac in enumerate(sh['a_code'], 1):
        try:
            x = ak.stock_zh_a_daily(symbol='sh' + ac,
                                    start_date='20130101',
                                    end_date='20201231', adjust='')
            x = x[['date', 'close']].copy()
            x['date'] = pd.to_datetime(x['date'])
            x['a_code'] = ac
            rows.append(x.drop_duplicates('date'))
        except Exception as e:
            failed.append((ac, type(e).__name__))
        if i % 15 == 0:
            print('  %d/%d  ok %d  failed %d' % (i, len(sh), len(rows),
                                                 len(failed)))
        time.sleep(0.8)
    A = (pd.concat(rows, ignore_index=True)
         .drop_duplicates(['a_code', 'date'])
         .rename(columns={'close': 'a_raw'}))
    A.to_csv(RAW_A, index=False)
    print('\nraw A: %d series, %s rows' % (A['a_code'].nunique(),
                                           format(len(A), ',')))
    if failed:
        print('failed (%d): %s' % (len(failed), failed[:8]))

A = A.rename(columns={'close': 'a_raw'}) if 'close' in A.columns else A

# ------------------------------------------------------------
# 2. The other three price series
# ------------------------------------------------------------
Ah = pd.read_csv(os.path.join(DIR, 'a_prices.csv'), parse_dates=['date'],
                 dtype={'a_code': str}, usecols=['date', 'a_code', 'close'])
Ah['a_code'] = Ah['a_code'].str.zfill(6)
Ah = Ah.rename(columns={'close': 'a_hfq'}).drop_duplicates(['a_code', 'date'])

Hr = pd.read_csv(os.path.join(DIR, 'h_prices.csv'), parse_dates=['date'],
                 dtype={'h_code': str}, usecols=['date', 'h_code', 'close'])
Hr['h_code'] = Hr['h_code'].str.zfill(5)
Hr = Hr.rename(columns={'close': 'h_raw'}).drop_duplicates(['h_code', 'date'])

Hh = pd.read_csv(os.path.join(DIR, 'h_prices_hfq.csv'), parse_dates=['date'],
                 dtype={'h_code': str}, usecols=['date', 'h_code', 'close'])
Hh['h_code'] = Hh['h_code'].str.zfill(5)
Hh = Hh.rename(columns={'close': 'h_hfq'}).drop_duplicates(['h_code', 'date'])

fx = pd.read_csv(os.path.join(DATA, 'hkd_cny_daily.csv'),
                 parse_dates=['date'])
print('\nFX: %d days, %.4f to %.4f' % (len(fx), fx['hkd_cny'].min(),
                                       fx['hkd_cny'].max()))

# ------------------------------------------------------------
# 3. Assemble
# ------------------------------------------------------------
P = (sh[['h_code', 'a_code', 'name']]
     .merge(A, on='a_code').merge(Ah, on=['a_code', 'date'], how='left')
     .merge(Hr, on=['h_code', 'date'], how='inner')
     .merge(Hh, on=['h_code', 'date'], how='left')
     .merge(fx, on='date', how='left'))
P = P[(P['a_raw'] > 0) & (P['h_raw'] > 0) & P['hkd_cny'].notna()]

# premium > 0 means the A line trades above the H line, the usual case
P['prem_raw'] = np.log(P['a_raw']) - np.log(P['h_raw'] * P['hkd_cny'])
m = P['a_hfq'].notna() & P['h_hfq'].notna() & (P['a_hfq'] > 0) & (P['h_hfq'] > 0)
P.loc[m, 'prem_hfq'] = (np.log(P.loc[m, 'a_hfq'])
                        - np.log(P.loc[m, 'h_hfq'] * P.loc[m, 'hkd_cny']))
# what the old panel used: no currency conversion at all
P['prem_old'] = np.log(P['h_raw']) - np.log(P['a_raw'])

P = P.sort_values(['h_code', 'date'])
for c in ('prem_raw', 'prem_hfq', 'prem_old'):
    P['d_' + c] = P.groupby('h_code')[c].diff()

print('\npanel: %s firm-days, %d firms, %d days (%s to %s)'
      % (format(len(P), ','), P['h_code'].nunique(), P['date'].nunique(),
         P['date'].min().date(), P['date'].max().date()))

# ------------------------------------------------------------
# 4. Is the premium sane?
# ------------------------------------------------------------
print('\n' + '=' * 72)
print('PREMIUM LEVEL   (positive = A trades above H, the usual case)')
print('=' * 72)
for c, lab in [('prem_raw', 'raw prices + FX'),
               ('prem_hfq', 'hfq prices + FX')]:
    s = P[c].dropna()
    if not len(s):
        continue
    print('   %-18s mean %+.4f  median %+.4f  sd %.4f  -> A/H ratio %.3f'
          % (lab, s.mean(), s.median(), s.std(), np.exp(s.mean())))
print('   %-18s mean %+.4f   (this is what ah_panel.csv carried)'
      % ('uncorrected', P['prem_old'].dropna().mean()))
print('\n   The Hang Seng AH Premium Index ran roughly 100-150 over this')
print('   period, i.e. A trading 0-50%% above H. exp(mean) should sit in')
print('   that range; far outside it means a code or currency error.')

yr = P.assign(y=P['date'].dt.year).groupby('y')['prem_raw'].agg(
    ['size', 'mean', 'std'])
yr['A_over_H'] = np.exp(yr['mean'])
print('\n   by year:')
print(yr.round(4).to_string())

# ------------------------------------------------------------
# 5. Daily change, and what the fixes do to it
# ------------------------------------------------------------
print('\n' + '=' * 72)
print('DAILY CHANGE IN THE PREMIUM')
print('=' * 72)
for c, lab in [('d_prem_raw', 'raw + FX'), ('d_prem_hfq', 'hfq + FX'),
               ('d_prem_old', 'uncorrected')]:
    s = P[c].dropna()
    if not len(s):
        continue
    print('   %-14s sd %.5f   p1 %+.4f   p99 %+.4f   max abs %.4f'
          % (lab, s.std(), s.quantile(0.01), s.quantile(0.99), s.abs().max()))

ok = P[['d_prem_raw', 'd_prem_hfq']].dropna()
if len(ok):
    print('\n   corr(raw, hfq) on daily changes: %.4f'
          % ok['d_prem_raw'].corr(ok['d_prem_hfq']))
    print('   days where they differ by >0.5%%: %d of %s'
          % (int(((ok['d_prem_raw'] - ok['d_prem_hfq']).abs() > 0.005).sum()),
             format(len(ok), ',')))
    print('   (large divergence = ex-dividend days, which is the one real')
    print('    advantage of the hfq version)')

# ------------------------------------------------------------
# 6. Common vs idiosyncratic: how much do 60 firms buy?
# ------------------------------------------------------------
print('\n' + '=' * 72)
print('VARIANCE DECOMPOSITION of the daily change')
print('=' * 72)
s = P.dropna(subset=['d_prem_raw'])
day = s.groupby('date')['d_prem_raw'].agg(['mean', 'var', 'size'])
day = day[day['size'] >= 10]
v_common = day['mean'].var()
v_idio = day['var'].mean()
n_bar = day['size'].mean()
print('   variance of the daily cross-sectional MEAN   %.6e' % v_common)
print('   mean within-day variance across firms        %.6e' % v_idio)
print('   average firms per day                        %.1f' % n_bar)
print('   common share of a single firm-day            %.3f'
      % (v_common / (v_common + v_idio)))
print('\n   sd of one firm-day        %.5f' % np.sqrt(v_common + v_idio))
print('   sd of the daily average   %.5f' % np.sqrt(day['mean'].var()))
print('   precision gain from averaging %.1fx'
      % (np.sqrt(v_common + v_idio) / np.sqrt(day['mean'].var())))
print("""
   This is the number that sets detectable effect sizes. Identification
   runs off days, so the daily average is the effective outcome; if the
   common share is high, the 60 firms buy far less than sqrt(60).""")

out = os.path.join(DATA, 'ah_premium_panel.csv')
P.to_csv(out, index=False)
print('\nwritten: %s  (%s firm-days)' % (out, format(len(P), ',')))
