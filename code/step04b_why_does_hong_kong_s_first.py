# Source: paper2_meteorology_build.ipynb, code cell 14 (final version of step 4b as run).
# ============================================================
# Step 4b — why does Hong Kong's first stage run backwards?
#
# Still pollution only. Nothing here touches the A-H ratio.
#
# Three things at once:
#   1. the Shanghai coverage gap (620 days against Hong Kong's 996)
#   2. unconditional log BLH / log PM2.5 correlation per city
#   3. every BLH variant against a four-rung control ladder
#
# Reads first_stage_panel.csv, written by step 4.
# ============================================================

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = '/content/drive/MyDrive/bilateral/data'
POLL = '/content/drive/MyDrive/pollution-ah'

panel = pd.read_csv(os.path.join(DATA, 'first_stage_panel.csv'),
                    parse_dates=['date'])

# ------------------------------------------------------------
# 1. Where are Shanghai's missing winter days?
# ------------------------------------------------------------
print('=' * 60)
print('SHANGHAI COVERAGE')
print('=' * 60)

cn = pd.read_csv(os.path.join(POLL, 'cn_pm25_daily.csv'),
                 usecols=['date', 'Shanghai'], parse_dates=['date'])
print('cn_pm25_daily.csv spans %s to %s'
      % (cn['date'].min().date(), cn['date'].max().date()))
print('Shanghai non-missing: %s of %s rows'
      % (format(int(cn['Shanghai'].notna().sum()), ','), format(len(cn), ',')))

WINTER = [11, 12, 1, 2, 3]
met_days = panel[panel['city'] == 'hongkong'][['date']].copy()
cn_w = cn[cn['date'].dt.month.isin(WINTER)]
cn_w = cn_w[(cn_w['date'] >= '2013-12-02') & (cn_w['date'] <= '2020-02-28')]
cov = (cn_w.assign(yr=cn_w['date'].dt.year)
       .groupby('yr')['Shanghai']
       .agg(days='size', present=lambda s: int(s.notna().sum())))
cov['pct'] = (100 * cov['present'] / cov['days']).round(1)
print('\nShanghai winter-day coverage by calendar year:')
print(cov.to_string())

missing = set(met_days['date']) - set(
    panel[panel['city'] == 'shanghai']['date'])
print('\ndays with meteorology but no Shanghai PM2.5: %d' % len(missing))
if missing:
    mm = pd.Series(sorted(missing))
    print('  earliest %s, latest %s' % (mm.iloc[0].date(), mm.iloc[-1].date()))
    print('  by year:', mm.dt.year.value_counts().sort_index().to_dict())

# ------------------------------------------------------------
# 2 and 3. Unconditional sign, then the control ladder
# ------------------------------------------------------------
VARIANTS = ['blh_land_mean', 'blh_all_mean', 'blh_point', 'blh_station_mean']

LADDER = [
    ('raw      ', [], False),
    ('FE       ', [], True),
    ('FE+T,P   ', ['temperature_2m', 'precipitation'], True),
    ('FE+full  ', ['temperature_2m', 'precipitation', 'wind_speed_10m',
                   'cloud_cover', 'shortwave_radiation'], True),
]


def run(d, blh_col, controls, use_fe):
    d = d.dropna(subset=[blh_col, 'log_pm25'] + controls).copy()
    if len(d) < 50:
        return None
    d['log_blh'] = np.log(d[blh_col].clip(lower=1))

    parts = [d[['log_blh'] + controls].reset_index(drop=True)]
    if use_fe:
        fe = pd.get_dummies(d[['month', 'year', 'dow']].astype(str),
                            drop_first=True).astype(float)
        parts.append(fe.reset_index(drop=True))
    X = sm.add_constant(pd.concat(parts, axis=1))
    y = d['log_pm25'].reset_index(drop=True)

    full = sm.OLS(y, X).fit(cov_type='HC1')
    red = sm.OLS(y, X.drop(columns=['log_blh'])).fit(cov_type='HC1')
    pr2 = (full.rsquared - red.rsquared) / (1 - red.rsquared)
    return full.params['log_blh'], full.tvalues['log_blh'], pr2, int(full.nobs)


for city in ['shanghai', 'hongkong']:
    d0 = panel[panel['city'] == city]
    print('\n' + '=' * 60)
    print('%s' % city.upper())
    print('=' * 60)

    have = [v for v in VARIANTS if v in d0 and d0[v].notna().any()]

    print('\nunconditional correlations with log PM2.5:')
    for v in have:
        s = d0[[v, 'log_pm25']].dropna()
        r = np.corrcoef(np.log(s[v].clip(lower=1)), s['log_pm25'])[0, 1]
        print('   log %-18s r = %+0.4f   (n %d)' % (v, r, len(s)))

    print('\nlog BLH coefficient across the control ladder:')
    print('   %-18s %-10s %9s %8s %10s %6s'
          % ('variant', 'spec', 'coef', 't', 'partial R2', 'N'))
    for v in have:
        for label, ctrl, fe in LADDER:
            out = run(d0, v, ctrl, fe)
            if out is None:
                continue
            b, t, pr2, n = out
            flag = '  <-- WRONG SIGN' if b > 0 else ''
            print('   %-18s %-10s %+9.4f %8.2f %10.4f %6d%s'
                  % (v, label, b, t, pr2, n, flag))
        print()

# ------------------------------------------------------------
# 4. Hong Kong network composition: is the 13-station mean balanced?
# ------------------------------------------------------------
print('=' * 60)
print('NOTE ON STATION COMPOSITION')
print('=' * 60)
print("""TSEUNG KWAN O sits at 60.4% coverage while the other twelve are all
above 85%, so it enters the network mean partway through the sample. A
city average whose membership changes mid-sample carries a measurement
discontinuity at the join. NORTH and SOUTHERN are at exactly 0.0%, which
means PM2.5 was not measured there in this window at all; if the sample
is ever extended past Feb 2020 for the mask test, those two may come
online and silently change the network again.

Decide now and pre-register it: either fix the set at the twelve
balanced stations, or keep thirteen and demean each station before
averaging so composition changes cannot move the city series.""")
