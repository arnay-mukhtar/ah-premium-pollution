# Source: paper2_meteorology_build.ipynb, code cell 32 (final version of step 18 as run).
# ============================================================
# Step 18 — the full-year mainland pollution build
#
# SCOPE CHANGED from the original plan. Not November and March from
# Zenodo, because (a) those months are not in that deposit at all, and
# (b) step 15 showed the instrument works in every season, so the winter
# restriction was costing days for nothing, and (c) step 17 showed the
# Zenodo and quotsoft series are different spatial aggregates, and
# quotsoft is MEP's published city figure, which is what this literature
# uses and what the public actually sees.
#
# Range starts 2014-05-13, the first day of quotsoft's national
# coverage. Full year to 2020-12-31, which covers the price panel and
# leaves the COVID months available as a robustness check rather than
# deciding now to drop them.
#
# Expected: roughly 2,400 days, about 1,650 of them trading days, against
# the 390 the winter-only Zenodo build delivered.
#
# Writes two things:
#   quotsoft_cities_hourly.csv  Shanghai and Shenzhen, hourly
#   quotsoft_cities_daily.csv   every city, daily 24h mean, >=8 hours
#
# The second one is for the methodological paper: a first-stage audit
# across hundreds of cities is the natural extension of the Hong Kong
# instrument-failure result, and these files carry it for free.
#
# RESUMABLE. Saves every 200 days. Rerun after a disconnect and it
# continues from where it stopped.
#
# Pollution only.
# ============================================================

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DATA = '/content/drive/MyDrive/bilateral/data'
os.makedirs(DATA, exist_ok=True)
H_PATH = os.path.join(DATA, 'quotsoft_cities_hourly.csv')
D_PATH = os.path.join(DATA, 'quotsoft_cities_daily.csv')

BASE = 'https://quotsoft.net/air/data'
START, END = '2014-05-13', '2020-12-31'
KEEP_HOURLY = ['上海', '深圳']        # Shanghai, Shenzhen
MIN_HOURS = 8

days = pd.date_range(START, END, freq='D')
print('range: %s to %s  (%d days)' % (START, END, len(days)))

# ------------------------------------------------------------
# Resume
# ------------------------------------------------------------
done = set()
h_parts, d_parts = [], []
if os.path.exists(D_PATH):
    prev_d = pd.read_csv(D_PATH, parse_dates=['date'])
    done = set(prev_d['date'])
    d_parts.append(prev_d)
    if os.path.exists(H_PATH):
        h_parts.append(pd.read_csv(H_PATH, parse_dates=['ts']))
    print('resuming: %d days already stored' % len(done))

todo = [d for d in days if d not in done]
print('to fetch: %d days\n' % len(todo))


def flush():
    if h_parts:
        (pd.concat(h_parts, ignore_index=True)
           .drop_duplicates('ts').sort_values('ts').to_csv(H_PATH,
                                                           index=False))
    if d_parts:
        (pd.concat(d_parts, ignore_index=True)
           .drop_duplicates('date').sort_values('date').to_csv(D_PATH,
                                                               index=False))


WORKERS = 8          # polite concurrency; 2,400 files in ~5 min not ~40
SESSION = requests.Session()


def fetch_day(day):
    """Download and reduce one day. Returns (day, daily_row, hourly, status)."""
    try:
        r = SESSION.get('%s/china_cities_%s.csv'
                        % (BASE, day.strftime('%Y%m%d')), timeout=30)
        if r.status_code != 200:
            return day, None, None, 'missing'
        t = pd.read_csv(io.BytesIO(r.content))
        t = t[t['type'] == 'PM2.5']
        if not len(t):
            return day, None, None, 'missing'

        cities = [c for c in t.columns if c not in ('date', 'hour', 'type')]
        vals = t[cities].apply(pd.to_numeric, errors='coerce')

        # Daily, every city, with the minimum-hours rule
        row = vals.mean().where(vals.notna().sum() >= MIN_HOURS).to_frame().T
        row.insert(0, 'date', day)

        # Hourly, the two cities the A-H design needs
        hh = None
        have = [c for c in KEEP_HOURLY if c in t.columns]
        if have:
            hh = t[['hour'] + have].copy()
            hh['ts'] = day + pd.to_timedelta(hh['hour'].astype(int),
                                             unit='h')
            hh = hh[['ts'] + have]
        return day, row, hh, 'ok'
    except Exception as e:
        return day, None, None, type(e).__name__


missing, errors, ok = 0, 0, 0
t0 = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    for i, (day, row, hh, status) in enumerate(
            pool.map(fetch_day, todo), 1):
        if status == 'ok':
            d_parts.append(row)
            if hh is not None:
                h_parts.append(hh)
            ok += 1
        elif status == 'missing':
            missing += 1
        else:
            errors += 1

        if i % 200 == 0:
            flush()
            el = time.time() - t0
            print('  %d/%d  ok %d  missing %d  errors %d'
                  '  %.1f min elapsed, ~%.1f left'
                  % (i, len(todo), ok, missing, errors, el / 60,
                     (el / i) * (len(todo) - i) / 60))

flush()
print('\nfetch done in %.1f min. ok %d, missing %d, errors %d'
      % ((time.time() - t0) / 60, ok, missing, errors))
if errors > len(todo) * 0.02:
    print('WARNING: %.1f%% errored. If that is a rate limit, lower WORKERS'
          ' to 4 and rerun; it resumes.' % (100 * errors / len(todo)))

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------
D = pd.read_csv(D_PATH, parse_dates=['date'])
H = pd.read_csv(H_PATH, parse_dates=['ts'])
print('\ndaily:  %d days x %d cities' % (len(D), D.shape[1] - 1))
print('hourly: %d rows, %s to %s'
      % (len(H), H['ts'].min().date(), H['ts'].max().date()))

for c in KEEP_HOURLY:
    if c not in D.columns:
        continue
    s = D[c].dropna()
    print('\n%s: %d days, mean %.1f, sd %.1f, max %.1f'
          % (c, len(s), s.mean(), s.std(), s.max()))
    yr = D.assign(y=D['date'].dt.year).groupby('y')[c].agg(['size', 'mean'])
    print(yr.round(1).to_string())

cov = D.drop(columns='date').notna().mean().sort_values(ascending=False)
print('\ncity coverage: %d at >=90%%, %d at >=50%%, %d below 10%%'
      % ((cov >= 0.9).sum(), (cov >= 0.5).sum(), (cov < 0.1).sum()))

# Overlap with the existing panel's trading days
try:
    cal = pd.read_csv('/content/drive/MyDrive/pollution-ah/ah_panel.csv',
                      usecols=['date', 'a_code'], parse_dates=['date'],
                      dtype={'a_code': str})
    cal['a_code'] = cal['a_code'].str.zfill(6)
    trading = set(cal.loc[cal['a_code'].str.startswith('6'), 'date'])
    have = set(D.loc[D['上海'].notna(), 'date']) if '上海' in D.columns else set()
    print('\nShanghai days overlapping panel trading days: %d'
          % len(have & trading))
    print('   (against 390 in the winter-only Zenodo build)')
except Exception as e:
    print('\npanel overlap check skipped: %s' % type(e).__name__)

print("""
NEXT
  Send the Shanghai day count and the trading-day overlap. Those two
  numbers set the first stage, and the first stage sets whether the
  design is viable. After that the panel gets rebuilt with the currency
  fix and the pre-registration can be written.
""")
