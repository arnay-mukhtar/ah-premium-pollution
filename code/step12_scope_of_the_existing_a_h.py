# Source: paper2_meteorology_build.ipynb, code cell 24 (final version of step 12 as run).
# ============================================================
# Step 12 — scope of the existing A-H panel
#
# Instant. Reads only the pair and panel files already on your Drive.
#
# Answers three questions the design now depends on:
#   1. How many pairs are Shenzhen-listed? They currently have no
#      instrument, since meteorology exists for Shanghai and Hong Kong
#      only. Few -> drop them. Many -> Shenzhen needs its own ERA5 pull.
#   2. Does the panel carry any currency conversion? (It does not; this
#      confirms the shape of the fix.)
#   3. What is the realistic firm-day count once the sample is limited
#      to days the Shanghai instrument actually covers?
#
# DISCIPLINE: counts sample composition only. No pollution variable is
# joined to any price variable, and no ratio is regressed on anything.
# ============================================================

import os
import numpy as np
import pandas as pd

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DIR = '/content/drive/MyDrive/pollution-ah'
DATA = '/content/drive/MyDrive/bilateral/data'

pairs = pd.read_csv(os.path.join(DIR, 'ah_pairs.csv'),
                    dtype={'h_code': str, 'a_code': str})
pairs['a_code'] = pairs['a_code'].str.zfill(6)
pairs['mkt'] = np.where(pairs['a_code'].str.startswith('6'),
                        'shanghai', 'shenzhen')

print('=' * 64)
print('1. PAIR COMPOSITION')
print('=' * 64)
print(pairs['mkt'].value_counts().to_string())
print('\nA-code prefixes:')
print(pairs['a_code'].str[:3].value_counts().to_string())

panel = pd.read_csv(os.path.join(DIR, 'ah_panel.csv'),
                    parse_dates=['date'], dtype={'h_code': str,
                                                 'a_code': str})
panel['a_code'] = panel['a_code'].str.zfill(6)
panel['mkt'] = np.where(panel['a_code'].str.startswith('6'),
                        'shanghai', 'shenzhen')

print('\n' + '=' * 64)
print('2. PANEL AS BUILT')
print('=' * 64)
print('columns: %s' % list(panel.columns))
fx = [c for c in panel.columns
      if any(k in c.lower() for k in ('fx', 'rate', 'hkd', 'cny', 'rmb'))]
print('currency columns present: %s'
      % (fx if fx else 'NONE  <- confirms the ratio mixes HKD and RMB'))
print('\nfirm-days  %s' % format(len(panel), ','))
print('firms      %d' % panel['h_code'].nunique())
print('days       %d   (%s to %s)'
      % (panel['date'].nunique(), panel['date'].min().date(),
         panel['date'].max().date()))
print('\nby market:')
g = panel.groupby('mkt').agg(firm_days=('date', 'size'),
                             firms=('h_code', 'nunique'))
g['pct_firm_days'] = (100 * g['firm_days'] / len(panel)).round(1)
print(g.to_string())

print('\n' + '=' * 64)
print('3. WHAT THE SHANGHAI INSTRUMENT ACTUALLY COVERS')
print('=' * 64)

cn = pd.read_csv(os.path.join(DATA, 'cn_pm25_daily_24h.csv'),
                 index_col='date', parse_dates=True)
inst_days = set(cn.index[cn['Shanghai'].notna()])
print('days with Shanghai PM2.5: %d' % len(inst_days))

snd = pd.read_csv(os.path.join(DATA, 'sounding_daily.csv'),
                  parse_dates=['date'])
snd_days = set(snd.loc[snd['city'] == 'shanghai', 'date'])
print('days with a Shanghai morning sounding: %d' % len(snd_days))
print('days with both: %d' % len(inst_days & snd_days))

for name, keep in [('all pairs', panel),
                   ('Shanghai-listed only',
                    panel[panel['mkt'] == 'shanghai'])]:
    s = keep[keep['date'].isin(inst_days)]
    s2 = keep[keep['date'].isin(inst_days & snd_days)]
    print('\n%s' % name)
    print('   firm-days on instrument days          %s  (%d firms, %d days)'
          % (format(len(s), ','), s['h_code'].nunique(), s['date'].nunique()))
    print('   firm-days also with a sounding        %s  (%d firms, %d days)'
          % (format(len(s2), ','), s2['h_code'].nunique(),
             s2['date'].nunique()))

print("""
WHAT THE NUMBERS DECIDE
  Shenzhen under ~15% of firm-days: drop them, restrict to Shanghai-listed
  A-shares, and the design stays clean at almost no cost.
  Shenzhen a third or more: it needs its own ERA5 pull, and its nearest
  radiosonde is King's Park at ~30 km, which would make the Shenzhen
  instrument mechanically correlated with the Hong Kong one. Say so in
  the paper rather than letting a referee find it.

  Note also that identification runs off DAYS, not firm-days: the
  instrument varies by city-day, so the day count is the effective
  sample for the first stage no matter how many firms sit on each day.
""")
