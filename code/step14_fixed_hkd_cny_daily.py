# Source: paper2_meteorology_build.ipynb, code cell 30 (final version of step 14 as run).
# ============================================================
# Step 14 (FIXED) — HKD/CNY daily
#
# ah_panel.csv divides a Hong Kong dollar price by a renminbi price with
# no conversion, so `dratio` currently carries the currency move mixed
# into the price signal. The premium needs
#     premium = log(P_A) - log(P_H * HKD_to_CNY)
#
# Two bugs in the first version, both fixed here:
#   1. It picked 中行汇买价 (BOC buying rate). The right series is
#      央行中间价, the PBOC central parity rate: the official reference.
#   2. reset_index() created a column named 'index', and the date
#      detector matched that before '日期', so row numbers became dates,
#      parsed as 1970, and every row fell outside the sample filter.
#
# The format is known now, so the column is named rather than guessed.
#
# currency_boc_sina quotes CNY per 100 units, hence the 0.01 scaling.
# ============================================================

import os
import numpy as np
import pandas as pd

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

OUT = '/content/drive/MyDrive/bilateral/data'
os.makedirs(OUT, exist_ok=True)
START, END = '2013-01-01', '2020-12-31'

try:
    import akshare as ak
except ImportError:
    os.system('pip install -q akshare')
    import akshare as ak

raw = ak.currency_boc_sina(symbol='港币',
                           start_date=START.replace('-', ''),
                           end_date=END.replace('-', ''))
print('currency_boc_sina(港币): %s' % (raw.shape,))
print(raw.head(3).to_string())

DATE_COL = '日期'
RATE_COL = '央行中间价'        # PBOC central parity, CNY per 100 HKD
FALLBACK = '中行折算价'        # BOC conversion rate, same units

col = RATE_COL if RATE_COL in raw.columns else FALLBACK
if col not in raw.columns:
    raise SystemExit('Neither rate column present. Columns: %s'
                     % list(raw.columns))
print('\nusing %r (median %.2f, quoted per 100 HKD)'
      % (col, raw[col].median()))

fx = raw[[DATE_COL, col]].copy()
fx.columns = ['date', 'rate']
fx['date'] = pd.to_datetime(fx['date'])
fx['hkd_cny'] = pd.to_numeric(fx['rate'], errors='coerce') / 100.0
fx = fx[['date', 'hkd_cny']].dropna().sort_values('date')
fx = fx[(fx['date'] >= START) & (fx['date'] <= END)]
print('quoted days in range: %d  (%s to %s)'
      % (len(fx), fx['date'].min().date(), fx['date'].max().date()))

# Forward fill onto every calendar day: FX and equity holidays differ.
full = pd.DataFrame({'date': pd.date_range(START, END, freq='D')})
fx = full.merge(fx, on='date', how='left')
fx['hkd_cny'] = fx['hkd_cny'].ffill().bfill()

v = fx['hkd_cny']
print('\nHKD/CNY')
print('   start %.4f   end %.4f   min %.4f   max %.4f'
      % (v.iloc[0], v.iloc[-1], v.min(), v.max()))
chg = np.log(v).diff()
print('   daily log change: sd %.5f   largest %.5f'
      % (chg.std(), chg.abs().max()))
print('   total drift over the sample: %+.2f%%'
      % (100 * (v.iloc[-1] / v.iloc[0] - 1)))

ok = (0.80 <= v.min()) and (v.max() <= 0.92)
print('\n   level inside the expected 0.80-0.92 band: %s' % ok)
if not ok:
    print('   NOT in band. Send me the medians and I will pick again.')

p = os.path.join(OUT, 'hkd_cny_daily.csv')
fx.to_csv(p, index=False)
print('\nwritten: %s  (%d rows)' % (p, len(fx)))

print("""
WHAT THIS CHANGES
  The premium becomes  log(P_A) - log(P_H) - log(HKD_CNY).
  Over 2013-2020 the drift printed above is the amount currently sitting
  inside `dratio` as a spurious trend. The daily log change sd is what
  it adds to each day's observation.

  Onshore versus offshore: this is the onshore CNY parity. Mainland
  investors face capital controls, so some of the A-H literature uses
  offshore CNH. Name whichever you use in the pre-registration and keep
  the other as a robustness check.
""")
