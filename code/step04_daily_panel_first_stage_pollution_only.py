# Source: paper2_meteorology_build.ipynb, code cell 13 (final version of step 4 as run).
# ============================================================
# Step 4 — daily panel + first stage, POLLUTION ONLY
#
# Nothing here touches the A-H ratio. This asks a single question
# per city: do dispersion conditions predict PM2.5? That is
# instrument validation and is legitimate work before the
# pre-registration is filed.
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = '/content/drive/MyDrive/bilateral/data'
POLL = '/content/drive/MyDrive/pollution-ah'

# Exposure window, local time. Both cities are UTC+8.
# Shanghai trades 09:30-11:30 and 13:00-15:00; Hong Kong 09:30-12:00
# and 13:00-16:00. The window opens before the session, as in paper 1,
# because the benchmark finds morning effects on the commute.
WIN_START, WIN_END = 7, 16          # 07:00 to 15:59
N_WINDOW_HOURS = WIN_END - WIN_START        # 9
MIN_HOURS = 7                       # 7 of 9 = 78%, matching paper 1's 8 of 11

# Hong Kong roadside stations, excluded: they sit in traffic and are
# systematically higher than the general network.
HK_ROADSIDE = ['CENTRAL', 'MONG KOK', 'CAUSEWAY BAY']

# ------------------------------------------------------------
# 1. Hong Kong pollution: hourly -> station-daily -> network-daily
# ------------------------------------------------------------
frames = []
for p in sorted(glob.glob(os.path.join(POLL, 'hk_pm25_*.csv'))):
    d = pd.read_csv(p, skiprows=11, encoding='utf-8-sig',
                    na_values=['N.A.', 'N.A', ''], low_memory=False)
    d.columns = [c.strip() for c in d.columns]
    frames.append(d)
hk = pd.concat(frames, ignore_index=True)

hk['date'] = pd.to_datetime(hk['DATE'], dayfirst=True)
hk['HOUR'] = hk['HOUR'].astype(int)

# EPD labels hour H as the period ending at H:00, so hours 8..16
# cover 07:00 to 16:00 local.
hk_win = hk[(hk['HOUR'] >= WIN_START + 1) & (hk['HOUR'] <= WIN_END)].copy()

gen_stations = [c for c in hk.columns
                if c not in ('DATE', 'HOUR', 'POLLUTANT', 'date')
                and c not in HK_ROADSIDE]
print('general stations in file: %d' % len(gen_stations))

# Station-day means, enforcing the minimum-readings rule per station.
long = hk_win.melt(id_vars=['date', 'HOUR'], value_vars=gen_stations,
                   var_name='station', value_name='pm25')
long['pm25'] = pd.to_numeric(long['pm25'], errors='coerce')

g = long.groupby(['station', 'date'])['pm25']
stn = g.agg(['mean', 'count']).reset_index()
stn.loc[stn['count'] < MIN_HOURS, 'mean'] = np.nan

# Which stations actually have coverage over the estimation window?
SAMPLE_START, SAMPLE_END = '2013-12-02', '2020-02-28'
in_sample = stn[(stn['date'] >= SAMPLE_START) & (stn['date'] <= SAMPLE_END)]
cov = (in_sample.groupby('station')['mean']
       .apply(lambda s: s.notna().mean())
       .sort_values(ascending=False))
print('\nstation coverage over %s to %s:' % (SAMPLE_START, SAMPLE_END))
for s, v in cov.items():
    print('   %-18s %5.1f%%' % (s, 100 * v))

KEEP = cov[cov > 0.5].index.tolist()
print('\nstations retained (>50%% coverage): %d' % len(KEEP))
print('  ', KEEP)
print('dropped:', [s for s in gen_stations if s not in KEEP])

hk_daily = (stn[stn['station'].isin(KEEP)]
            .pivot(index='date', columns='station', values='mean')
            .mean(axis=1, skipna=True)
            .rename('pm25')
            .reset_index())
hk_daily['city'] = 'hongkong'

# ------------------------------------------------------------
# 2. Mainland pollution: already daily
# ------------------------------------------------------------
cn = pd.read_csv(os.path.join(POLL, 'cn_pm25_daily.csv'),
                 usecols=['date', 'Shanghai', 'Shenzhen'], parse_dates=['date'])
sh_daily = cn[['date', 'Shanghai']].rename(columns={'Shanghai': 'pm25'})
sh_daily['city'] = 'shanghai'

poll = pd.concat([sh_daily, hk_daily], ignore_index=True)

# ------------------------------------------------------------
# 3. Meteorology: hourly -> daily over the same window
# ------------------------------------------------------------
WEATHER_FIELDS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
                  'surface_pressure', 'cloud_cover', 'shortwave_radiation',
                  'u', 'v']

met = []
for city in ['shanghai', 'hongkong']:
    w = pd.read_csv(os.path.join(DATA, '%s_weather_hourly.csv' % city),
                    parse_dates=['time', 'date'])
    b = pd.read_csv(os.path.join(DATA, '%s_blh_hourly.csv' % city),
                    parse_dates=['time', 'date'])
    m = w.merge(b.drop(columns=['date', 'hour']), on='time', how='inner')

    m = m[(m['hour'] >= WIN_START) & (m['hour'] < WIN_END)]

    blh_cols = [c for c in m.columns if c.startswith('blh_')]
    agg = {f: 'mean' for f in WEATHER_FIELDS if f in m.columns}
    agg.update({c: 'mean' for c in blh_cols})

    d = m.groupby('date').agg({**agg, 'time': 'count'}).reset_index()
    d = d.rename(columns={'time': 'n_hours'})
    d = d[d['n_hours'] >= MIN_HOURS]
    d['city'] = city
    met.append(d)
    print('\n%s meteorology: %d daily obs, blh series %s'
          % (city, len(d), blh_cols))

met = pd.concat(met, ignore_index=True)

# ------------------------------------------------------------
# 4. Merge and run the first stage, per city
# ------------------------------------------------------------
panel = met.merge(poll, on=['date', 'city'], how='inner')
panel = panel[(panel['date'] >= SAMPLE_START) & (panel['date'] <= SAMPLE_END)]
panel = panel[panel['pm25'] > 0]
panel['log_pm25'] = np.log(panel['pm25'])
panel['month'] = panel['date'].dt.month
panel['year'] = panel['date'].dt.year
panel['dow'] = panel['date'].dt.dayofweek

panel.to_csv(os.path.join(DATA, 'first_stage_panel.csv'), index=False)
print('\npanel written: %d rows' % len(panel))

CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']

for city in ['shanghai', 'hongkong']:
    d = panel[panel['city'] == city].copy()
    blh_primary = ('blh_station_mean' if 'blh_station_mean' in d
                   and d['blh_station_mean'].notna().any() else 'blh_land_mean')

    d = d.dropna(subset=[blh_primary, 'log_pm25', 'u', 'v',
                         'surface_pressure'] + CONTROLS)
    d['log_blh'] = np.log(d[blh_primary].clip(lower=1))

    instruments = ['log_blh', 'u', 'v', 'surface_pressure']
    fe = pd.get_dummies(d[['month', 'year', 'dow']].astype(str),
                        drop_first=True).astype(float)

    X_full = pd.concat([d[instruments + CONTROLS].reset_index(drop=True),
                        fe.reset_index(drop=True)], axis=1)
    X_full = sm.add_constant(X_full)
    y = d['log_pm25'].reset_index(drop=True)

    full = sm.OLS(y, X_full).fit(cov_type='HC1')

    X_red = X_full.drop(columns=instruments)
    red = sm.OLS(y, X_red).fit(cov_type='HC1')

    R = np.zeros((len(instruments), X_full.shape[1]))
    for i, nm in enumerate(instruments):
        R[i, list(X_full.columns).index(nm)] = 1.0
    f = full.f_test(R)

    partial_r2 = (full.rsquared - red.rsquared) / (1 - red.rsquared)

    print('\n' + '=' * 56)
    print('%s  first stage  (instrument: %s)' % (city.upper(), blh_primary))
    print('  N                 %d' % int(full.nobs))
    print('  robust joint F    %.2f' % float(np.squeeze(f.fvalue)))
    print('  p                 %.4g' % float(np.squeeze(f.pvalue)))
    print('  partial R2        %.4f' % partial_r2)
    print('  full R2           %.4f' % full.rsquared)
    print('  coefficients:')
    for nm in instruments:
        print('    %-18s %9.4f   (t %6.2f)'
              % (nm, full.params[nm], full.tvalues[nm]))
