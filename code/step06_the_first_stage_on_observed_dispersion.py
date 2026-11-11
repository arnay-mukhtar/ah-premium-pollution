# Source: paper2_meteorology_build.ipynb, code cell 17 (final version of step 6 as run).
# ============================================================
# Step 6 — the first stage on OBSERVED dispersion, both cities
#
# Run in the same session as step 5; it reuses parse(), thermo(),
# blh_parcel(), blh_richardson(), inversion() and the constants.
#
# The question: ERA5 says Hong Kong's boundary layer is DEEP on dirty
# days, which inverts the dilution relationship. ERA5 correlates only
# ~0.52 with observed boundary layer depth there and runs 60% high, so
# the inversion could be non-classical reanalysis error rather than
# physics. Radiosondes settle it.
#
# Expected signs:
#   boundary layer height   NEGATIVE   deeper layer dilutes
#   inversion strength      POSITIVE   stronger cap traps
#
# Shanghai (Baoshan) is the positive control. If observed Shanghai comes
# out negative and observed Hong Kong positive, the contrast rests on
# observations and the reanalysis leaves the critical path.
#
# Still pollution and meteorology only. Nothing touches the A-H ratio.
# ============================================================

import io
import os
import zipfile

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

# WMO numbers: King's Park 45004 (Hong Kong), Baoshan 58362 (Shanghai)
WANT = {'hongkong': '45004', 'shanghai': '58362'}

lines = open(os.path.join(RAW, 'igra2-station-list.txt'),
             encoding='latin-1').read().splitlines()

ids = {}
for city, wmo in WANT.items():
    hit = [ln.split()[0] for ln in lines if wmo in ln[:12]]
    if hit:
        ids[city] = hit[0]
        print('%-9s -> %s' % (city, [ln for ln in lines
                                     if ln.startswith(hit[0])][0].strip()))
    else:
        print('%-9s -> WMO %s NOT FOUND in station list' % (city, wmo))


def soundings(city, station):
    """Download if needed, parse, return one row per sounding."""
    path = os.path.join(RAW, '%s-data.txt' % station)
    if not os.path.exists(path):
        url = ('https://www.ncei.noaa.gov/data/integrated-global-radiosonde'
               '-archive/access/data-por/%s-data.txt.zip' % station)
        print('  downloading %s ...' % station)
        r = requests.get(url, timeout=3600)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        open(path, 'wb').write(z.read(z.namelist()[0]))
    print('  %s: %.0f MB' % (station, os.path.getsize(path) / 1e6))

    out = []
    for (yr, mo, dy, hr), lev in parse(path):
        if hr not in (0, 12):
            continue
        d = thermo(lev)
        if d is None:
            continue
        ib, it, istr = inversion(d)
        out.append({
            'city': city,
            'date': pd.Timestamp(yr, mo, dy),
            'hour_local': 8 if hr == 0 else 20,
            'n_src_2km': int(d['n_src_2km'].iloc[0]),
            'obs_blh_parcel': blh_parcel(d),
            'obs_blh_ri': blh_richardson(d),
            'inv_base': ib, 'inv_top': it, 'inv_strength': istr,
        })
    df = pd.DataFrame(out)
    print('  %d soundings, %d at 08:00 local'
          % (len(df), int((df['hour_local'] == 8).sum())))
    return df


obs = []
for city, station in ids.items():
    print('\n%s' % city)
    obs.append(soundings(city, station))
obs = pd.concat(obs, ignore_index=True)

# Morning sounding only: it sits at the start of the 07:00-16:00 window.
morning = obs[obs['hour_local'] == 8].drop(columns=['hour_local'])

# ------------------------------------------------------------
# Merge: daily PM2.5 and window-mean controls from step 4, plus ERA5
# boundary layer at hour 8 so the comparison is like for like.
# ------------------------------------------------------------
panel = pd.read_csv(os.path.join(DATA, 'first_stage_panel.csv'),
                    parse_dates=['date'])

era8 = []
for city in panel['city'].unique():
    e = pd.read_csv(os.path.join(DATA, '%s_blh_hourly.csv' % city),
                    parse_dates=['date'])
    e = e[e['hour'] == 8]
    col = ('blh_station_mean' if 'blh_station_mean' in e.columns
           else 'blh_land_mean')
    era8.append(pd.DataFrame({'city': city, 'date': e['date'].values,
                              'era5_blh_h8': e[col].values}))
era8 = pd.concat(era8, ignore_index=True)

d = panel.merge(era8, on=['city', 'date'], how='left')
d = d.merge(morning, on=['city', 'date'], how='inner')
print('\nmerged: %s' % d.groupby('city').size().to_dict())

# ------------------------------------------------------------
# Unconditional signs first. This is the headline number.
# ------------------------------------------------------------
MEASURES = [
    ('era5_blh_h8', 'ERA5 BLH 08:00', 'log', 'neg'),
    ('obs_blh_parcel', 'observed BLH, parcel', 'log', 'neg'),
    ('obs_blh_ri', 'observed BLH, Richardson', 'log', 'neg'),
    ('inv_strength', 'observed inversion strength', 'lin', 'pos'),
]

print('\n' + '=' * 72)
print('UNCONDITIONAL correlation with log PM2.5')
print('=' * 72)
print('   %-28s %10s %10s   %s' % ('measure', 'shanghai', 'hongkong', 'want'))
for col, label, tf, want in MEASURES:
    row = []
    for city in ['shanghai', 'hongkong']:
        s = d[(d['city'] == city)][[col, 'log_pm25']].dropna()
        if len(s) < 50:
            row.append(np.nan)
            continue
        x = np.log(s[col].clip(lower=1)) if tf == 'log' else s[col]
        row.append(np.corrcoef(x, s['log_pm25'])[0, 1])
    print('   %-28s %+10.4f %+10.4f   %s' % (label, row[0], row[1], want))

# ------------------------------------------------------------
# Then the conditional first stage, same spec, one measure at a time.
# ------------------------------------------------------------
CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']


def first_stage(sub, col, tf):
    sub = sub.dropna(subset=[col, 'log_pm25'] + CONTROLS).copy()
    if len(sub) < 80:
        return None
    sub['x'] = (np.log(sub[col].clip(lower=1)) if tf == 'log' else sub[col])
    fe = pd.get_dummies(sub[['month', 'year', 'dow']].astype(str),
                        drop_first=True).astype(float)
    X = sm.add_constant(pd.concat(
        [sub[['x'] + CONTROLS].reset_index(drop=True),
         fe.reset_index(drop=True)], axis=1))
    y = sub['log_pm25'].reset_index(drop=True)
    full = sm.OLS(y, X).fit(cov_type='HC1')
    red = sm.OLS(y, X.drop(columns=['x'])).fit(cov_type='HC1')
    pr2 = (full.rsquared - red.rsquared) / (1 - red.rsquared)
    return full.params['x'], full.tvalues['x'], pr2, int(full.nobs)


print('\n' + '=' * 72)
print('FIRST STAGE, month/year/dow FE + weather controls')
print('=' * 72)
for city in ['shanghai', 'hongkong']:
    print('\n%s' % city.upper())
    print('   %-28s %10s %8s %11s %7s %s'
          % ('measure', 'coef', 't', 'partial R2', 'N', 'verdict'))
    sub = d[d['city'] == city]
    for col, label, tf, want in MEASURES:
        r = first_stage(sub, col, tf)
        if r is None:
            print('   %-28s %10s' % (label, 'n/a'))
            continue
        b, t, pr2, n = r
        sign = 'pos' if b > 0 else 'neg'
        if abs(t) < 1.96:
            verdict = 'not significant'
        elif sign == want:
            verdict = 'CORRECT SIGN'
        else:
            verdict = 'WRONG SIGN'
        print('   %-28s %+10.4f %8.2f %11.4f %7d %s'
              % (label, b, t, pr2, n, verdict))

out = os.path.join(DATA, 'observed_dispersion_panel.csv')
d.to_csv(out, index=False)
print('\nwritten: %s  (%d rows)' % (out, len(d)))

print("""
WHAT DECIDES WHAT
  Observed BLH negative in Shanghai, positive in Hong Kong:
      the contrast is real, built on observations, and ERA5 leaves the
      critical path. This is the finding and the paper has its spine.
  Observed BLH negative in BOTH:
      ERA5's non-classical error produced the Hong Kong inversion.
      Hong Kong stays in the design, instrumented on observations.
  Observed inversion strength positive and significant anywhere:
      that is a working instrument with an unambiguous expected sign,
      and it becomes the primary instrument for that city.
""")
