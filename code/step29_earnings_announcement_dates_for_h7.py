# Source: paper2_meteorology_build.ipynb, code cell 54 (final version of step 29 as run).
# ============================================================
# Step 29 — earnings announcement dates for H7
#
# H7 is the registered within-day test: does the pollution response differ
# for firms in an earnings announcement window? It is identified from the
# difference between two firms ON THE SAME DAY, so it carries day fixed
# effects and is immune to every day-level confound the primary test has
# to argue away. That is why it is worth the trouble.
#
# The registration names the source and the fallback, and commits to
# reporting H7 as NOT RUN if fewer than 50 of the 61 firms can be covered.
# This file collects the dates and reports that coverage. It estimates
# nothing.
#
#   primary    EastMoney performance-report endpoint via akshare, which
#              carries the latest announcement date for every A-share by
#              reporting period. No token required.
#   fallback   CNINFO official disclosure archive, also via akshare.
#   last       CSMAR via institutional access, done by hand.
#
# Announcement dates are a treatment-side variable in the H7 design and
# are collected without touching the price panel.
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

DATA = '/content/drive/MyDrive/bilateral/data'
DIR = '/content/drive/MyDrive/pollution-ah'
OUT = os.path.join(DATA, 'announcement_dates.csv')
START, END = '2014-05-13', '2019-12-31'

# Reporting periods to sweep. Chinese listed firms file quarterly, so this
# is every quarter end from 2014Q1 through 2019Q4, plus 2013Q4 so that an
# annual report announced in spring 2014 is not missed.
PERIODS = [d.strftime('%Y%m%d') for d in
           pd.date_range('2013-12-31', '2019-12-31', freq='QE')]

# ------------------------------------------------------------
# 0. What are the join keys? Printed before anything else, because the
#    H7 estimation has to merge on whatever the panel actually uses.
# ------------------------------------------------------------
print('=' * 74)
print('PANEL COLUMNS')
print('=' * 74)
for name, path in [('ah_premium_clean.csv',
                    os.path.join(DATA, 'ah_premium_clean.csv')),
                   ('ah_panel.csv', os.path.join(DIR, 'ah_panel.csv'))]:
    try:
        head = pd.read_csv(path, nrows=3, dtype=str)
        print('\n%s' % name)
        print('   columns: %s' % list(head.columns))
        print(head.to_string(index=False, max_colwidth=14))
    except Exception as exc:
        print('\n%s  COULD NOT READ: %s' % (name, exc))

# The set of firms the registered sample actually uses.
prem = pd.read_csv(os.path.join(DATA, 'ah_premium_clean.csv'),
                   parse_dates=['date'], dtype={'h_code': str})
prem = prem[(prem['date'] >= START) & (prem['date'] <= END)]
firms_h = sorted(prem['h_code'].dropna().unique())
print('\nfirms in the registered sample: %d' % len(firms_h))

acode = None
try:
    pan = pd.read_csv(os.path.join(DIR, 'ah_panel.csv'),
                      usecols=['a_code', 'h_code'],
                      dtype={'a_code': str, 'h_code': str}).drop_duplicates()
    pan['a_code'] = pan['a_code'].str.zfill(6)
    acode = pan[pan['h_code'].isin(firms_h)].drop_duplicates('h_code')
    print('a_code mapping available for %d of them' % len(acode))
except Exception as exc:
    print('could not build the a_code mapping: %s' % exc)

# ------------------------------------------------------------
# 1. Fetch
# ------------------------------------------------------------
try:
    import akshare as ak
except ImportError:
    os.system('pip -q install akshare')
    import akshare as ak
print('\nakshare %s' % getattr(ak, '__version__', 'unknown'))

CODE_KEYS = ['股票代码', '证券代码', 'code', 'symbol']
DATE_KEYS = ['最新公告日期', '公告日期', '首次公告日期', 'ann_date',
             '披露日期']


def pick(df, keys):
    for k in keys:
        if k in df.columns:
            return k
    return None


print('\n' + '=' * 74)
print('FETCHING, %d reporting periods' % len(PERIODS))
print('=' * 74)
rows, failed = [], []
for p in PERIODS:
    got = None
    for fn, kw in [('stock_yjbb_em', {'date': p}),
                   ('stock_yjkb_em', {'date': p}),
                   ('stock_yysj_em', {'date': p})]:
        f = getattr(ak, fn, None)
        if f is None:
            continue
        try:
            df = f(**kw)
            if df is None or not len(df):
                continue
            ck, dk = pick(df, CODE_KEYS), pick(df, DATE_KEYS)
            if ck is None or dk is None:
                continue
            sub = df[[ck, dk]].copy()
            sub.columns = ['a_code', 'ann_date']
            sub['period'] = p
            sub['source'] = fn
            rows.append(sub)
            got = '%s  %d rows' % (fn, len(sub))
            break
        except Exception as exc:
            last = '%s: %s' % (fn, str(exc)[:70])
    if got:
        print('   %s  %s' % (p, got))
    else:
        failed.append(p)
        print('   %s  FAILED  %s' % (p, locals().get('last', '')))
    time.sleep(1.0)

if not rows:
    raise SystemExit(
        '\nNo reporting period returned usable data. The akshare endpoints '
        'have changed or the network is blocked.\nFall back to CNINFO or '
        'CSMAR, or send me this output and I will rewrite the fetch.')

ann = pd.concat(rows, ignore_index=True)
ann['a_code'] = (ann['a_code'].astype(str).str.extract(r'(\d{6})')[0])
ann['ann_date'] = pd.to_datetime(ann['ann_date'], errors='coerce')
ann = ann.dropna(subset=['a_code', 'ann_date']).drop_duplicates(
    ['a_code', 'ann_date'])
ann = ann[(ann['ann_date'] >= '2014-01-01') & (ann['ann_date'] <= END)]
print('\n%s announcement records, %d distinct A-share codes'
      % (format(len(ann), ','), ann['a_code'].nunique()))

# ------------------------------------------------------------
# 2. Coverage against the registered 50-of-61 threshold
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('COVERAGE')
print('=' * 74)
if acode is None or not len(acode):
    print('No a_code mapping, so coverage cannot be checked here. The full')
    print('archive is written anyway and the merge is done in step 30.')
    keep = ann
else:
    mine = acode.merge(ann, on='a_code', how='left')
    per = (mine.dropna(subset=['ann_date'])
           .groupby('h_code')['ann_date'].nunique())
    covered = int((per >= 4).sum())          # at least one year of filings
    print('firms with at least 4 announcement dates: %d of %d'
          % (covered, len(acode)))
    print('median announcements per covered firm: %.0f'
          % (per.median() if len(per) else 0))
    print('firms with none at all: %s'
          % sorted(set(acode['h_code']) - set(per.index))[:20])
    if covered >= 50:
        print('\nTHRESHOLD MET. H7 runs on %d firms.' % covered)
    else:
        print('\nTHRESHOLD NOT MET (%d < 50). As registered, H7 is reported'
              % covered)
        print('as NOT RUN unless a fallback source closes the gap. It is')
        print('not replaced by a substitute interaction.')
    keep = mine.dropna(subset=['ann_date'])[['h_code', 'a_code', 'ann_date']]

keep.to_csv(OUT, index=False)
print('\nwritten: %s  (%s rows)' % (OUT, format(len(keep), ',')))
if failed:
    print('periods that failed: %s' % failed)

print("""
WHAT HAPPENS NEXT
  Step 30 builds D(i,t), equal to 1 on the announcement day and the
  following trading day, and runs the registered H7 interaction with day
  and firm fixed effects, instrumented by Z(t) x D(i,t).

  Send me this output before running it. The announcement count per firm
  decides whether the test has any power at all, and the column names
  printed at the top decide how the merge is written.
""")
