# Source: paper2_meteorology_build.ipynb, code cell 57 (final version of step 31 as run).
# ============================================================
# Step 31 — EXPLORATORY follow-ups to H7
#
# H7 returned +16.6 bp with t near 1.6 and an AR set of about
# [-10, +49]. Suggestive in the cognitive-impairment direction, nowhere
# near significant, and at 211 identifying days it is not going to
# become significant. Nothing here is an attempt to make it so.
#
# These are the two questions the estimate invites, and both can
# embarrass it rather than support it, which is why they are worth
# running.
#
# 1. WHICH LEG. H7a says the differential belongs to the Shanghai
#    investors who breathe the air. Here the decomposition is EXACT: the
#    FX term is common to every firm on a day, so the day fixed effects
#    absorb it and delta_A - delta_H = delta with no residual. A
#    differential coming from the Hong Kong leg is noise.
#
# 2. IS IT JUST THE SEASON. Chinese reporting season is also a weather
#    season. The same interaction is run with Beijing and Urumqi
#    pollution, each on its own instruments. If a distant city shows the
#    same differential, the finding is seasonality and must be dropped.
#
# EVERYTHING HERE IS EXPLORATORY AND IS LABELLED AS SUCH IN THE PAPER.
# It was written after seeing the H7 estimate. None of it is registered
# and none of it may be reported as confirmatory.
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

INST = ['log_blh', 'u', 'v', 'surface_pressure']
CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']


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
        bo, Gb = groups['both']
        V = (sandwich(day, Gd, 0) + sandwich(firm, Gf, 0)
             - sandwich(bo, Gb, 0))
    return float(b[0]), float(np.sqrt(max(V[0, 0], 1e-300)))


def demean_by(M, codes, G):
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
# Rebuild exactly the step 30 sample
# ------------------------------------------------------------
prem = pd.read_csv(os.path.join(DATA, 'ah_premium_clean.csv'),
                   parse_dates=['date'],
                   dtype={'h_code': str, 'a_code': str},
                   usecols=['date', 'h_code', 'a_code', 'd_prem_raw',
                            'd_logA', 'd_logH'])
fs = pd.read_csv(os.path.join(DATA, 'shanghai_first_stage_fullyear.csv'),
                 parse_dates=['date'])
ann = pd.read_csv(os.path.join(DATA, 'announcement_dates.csv'),
                  parse_dates=['ann_date'],
                  dtype={'a_code': str, 'h_code': str})
three = pd.read_csv(os.path.join(DATA, 'three_city_panel.csv'),
                    parse_dates=['date'])

d = prem.merge(fs, on='date', how='inner')
d = d[(d['date'] >= START) & (d['date'] <= END)]
d = d.dropna(subset=['d_prem_raw', 'd_logA', 'd_logH', 'log_pm25']
             + INST + CONTROLS).copy()
d = d.sort_values(['date', 'h_code']).reset_index(drop=True)
d['a_code'] = d['a_code'].str.zfill(6)
ann['a_code'] = ann['a_code'].astype(str).str.zfill(6)

sd_pm = d.groupby('date')['log_pm25'].first().std()
BP = 10000.0 * sd_pm

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

_, day = np.unique(d['date'].values, return_inverse=True)
_, firm = np.unique(d['h_code'].values, return_inverse=True)
Gd, Gf = day.max() + 1, firm.max() + 1
_, both = np.unique(day * Gf + firm, return_inverse=True)
groups = {'day': (day, Gd), 'firm': (firm, Gf),
          'both': (both, both.max() + 1)}

D = d['D'].values
firm_d = pd.get_dummies(d['h_code'], drop_first=True).values.astype(float)
W = demean_by(np.column_stack([D, firm_d]), day, Gd)

print('step 31: %s firm-days, %s treated, %d firms'
      % (format(len(d), ','), format(int(D.sum()), ','),
         d['h_code'].nunique()))


def h7(frame, outcome_col, pm_col, inst_cols):
    y = demean_by(frame[outcome_col].values, day, Gd)
    x = demean_by(frame[pm_col].values * D, day, Gd)
    Z = demean_by(frame[inst_cols].values * D[:, None], day, Gd)
    return {k: tsls(y, x, Z, W, groups, k) for k in ('dk', 'firm', 'two')}


# ------------------------------------------------------------
# 1. Which leg carries the differential
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('1. LEG DECOMPOSITION OF delta                       EXPLORATORY')
print('=' * 74)
print('   %-26s %12s %12s %8s %11s'
      % ('outcome', 'delta', 'se (DK)', 't', 'bp per sd'))
got = {}
for col, lab in [('d_prem_raw', 'premium'),
                 ('d_logA', 'A leg (Shanghai)'),
                 ('d_logH', 'H leg (Hong Kong)')]:
    bb, ss = h7(d, col, 'log_pm25', INST)['dk']
    got[col] = bb
    print('   %-26s %12.6f %12.6f %8.2f %11.2f'
          % (lab, bb, ss, bb / ss, bb * BP))

diff = got['d_logA'] - got['d_logH']
print('   %-26s %12.6f %12s %8s %11.2f'
      % ('A minus H', diff, '', '', diff * BP))
print('   identity residual %.2e   (must be ~0: the FX term is common to'
      % (got['d_prem_raw'] - diff))
print('   all firms on a day, so the day fixed effects absorb it)')
print('')
print('   A carries it, H near zero -> coherent with H7a')
print('   H carries it              -> the differential is noise')
print('   both move together        -> a common shock the day fixed')
print('                                effects failed to absorb')

# ------------------------------------------------------------
# 2. Distant-city placebo on the same interaction
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('2. IS IT JUST REPORTING SEASON                      EXPLORATORY')
print('=' * 74)
dd = d.merge(three, on='date', how='left')
print('   %-26s %12s %12s %8s %11s'
      % ('pollution of', 'delta', 'se (DK)', 't', 'bp per sd'))
bb, ss = h7(d, 'd_prem_raw', 'log_pm25', INST)['dk']
print('   %-26s %12.6f %12.6f %8.2f %11.2f'
      % ('Shanghai (the result)', bb, ss, bb / ss, bb * BP))
for tag, lab in [('beijing', 'Beijing'), ('urumqi', 'Urumqi')]:
    cols = ['%s_%s' % (tag, c) for c in INST]
    pm = '%s_log_pm25' % tag
    sub = dd.copy()
    miss = sub[[pm] + cols].isna().any(axis=1)
    if miss.any():
        # Zero out the interaction on days the city is missing rather than
        # dropping rows, which would change the fixed-effect structure and
        # make the placebo a different sample from the result.
        sub.loc[miss, [pm] + cols] = 0.0
        print('   (%s missing on %d firm-days, interaction set to zero '
              'there)' % (lab, int(miss.sum())))
    bb, ss = h7(sub, 'd_prem_raw', pm, cols)['dk']
    print('   %-26s %12.6f %12.6f %8.2f %11.2f'
          % (lab + ' (placebo)', bb, ss, bb / ss, bb * BP))

print('')
print('   A distant city showing a similar differential means the result')
print('   is reporting season interacting with weather season, not')
print('   Shanghai air, and H7 must be dropped rather than caveated.')

# ------------------------------------------------------------
# 3. Where the 211 identifying days actually are
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('3. THE IDENTIFYING SAMPLE')
print('=' * 74)
bd = d.groupby('date')['D'].agg(['sum', 'size'])
contrast = bd[(bd['sum'] > 0) & (bd['sum'] < bd['size'])]
print('   days with a within-day contrast: %d of %d'
      % (len(contrast), len(bd)))
mon = pd.Series(contrast.index).dt.month.value_counts().sort_index()
print('   %-8s %s' % ('month', ''.join('%5d' % m for m in range(1, 13))))
print('   %-8s %s' % ('days', ''.join('%5d' % mon.get(m, 0)
                                      for m in range(1, 13))))
yr = pd.Series(contrast.index).dt.year.value_counts().sort_index()
print('   by year: ' + ', '.join('%d: %d' % (k, v) for k, v in yr.items()))
print('   treated firms on a contrast day: median %.0f of %d'
      % (contrast['sum'].median(), int(bd['size'].median())))
print('')
print('   If these days sit in two or three months, say so in the paper.')
print('   The test is then identified off a seasonally narrow slice and')
print('   its external validity across the year is not established.')
