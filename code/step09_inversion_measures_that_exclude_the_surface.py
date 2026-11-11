# Source: paper2_meteorology_build.ipynb, code cell 21 (final version of step 9 as run).
# ============================================================
# Step 9 — inversion measures that exclude the surface
#
# Fast: reads surface_inversion_panel.csv, no re-parsing.
#
# WHY. Step 7's measures are dT_h = T(h) - T(2m), and the full control
# set contains temperature_2m. One of the two terms in the regressor was
# also a regressor, so the "full" column was misspecified rather than a
# robustness check the instrument failed. Clear-sky radiative cooling,
# which cloud_cover and shortwave_radiation capture, is also the physical
# cause of a morning surface inversion, compounding it.
#
# FIX. Build stability from two levels ALOFT, e.g. T(300m) - T(100m).
# No surface temperature appears, so temperature_2m can be controlled
# freely. Identity used: dT_a - dT_b = T(a) - T(b).
#
# Expected sign on every measure: POSITIVE. Warmer air over colder air
# is more stable, suppresses vertical exchange, traps emissions.
#
# Pollution and meteorology only. Nothing touches the A-H ratio.
# ============================================================

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DATA = '/content/drive/MyDrive/bilateral/data'

d = pd.read_csv(os.path.join(DATA, 'surface_inversion_panel.csv'),
                parse_dates=['date'])
print('loaded %d rows: %s' % (len(d), d.groupby('city').size().to_dict()))

# Aloft-only stability: T(a) - T(b) = dT_a - dT_b, no surface term.
d['dT_200_100'] = d['dT_200'] - d['dT_100']
d['dT_300_100'] = d['dT_300'] - d['dT_100']
d['dT_500_100'] = d['dT_500'] - d['dT_100']
d['dT_500_200'] = d['dT_500'] - d['dT_200']
d['dT_1000_300'] = d['dT_1000'] - d['dT_300']

SURFACE = [('dT_100', 'T(100) - T(sfc)'),
           ('dT_200', 'T(200) - T(sfc)'),
           ('dT_300', 'T(300) - T(sfc)')]
ALOFT = [('dT_200_100', 'T(200) - T(100)'),
         ('dT_300_100', 'T(300) - T(100)'),
         ('dT_500_100', 'T(500) - T(100)'),
         ('dT_500_200', 'T(500) - T(200)'),
         ('dT_1000_300', 'T(1000) - T(300)')]
ALL = SURFACE + ALOFT

print('\n' + '=' * 74)
print('UNCONDITIONAL correlation with log PM2.5   (want POSITIVE)')
print('=' * 74)
print('   %-22s %10s %10s' % ('measure', 'shanghai', 'hongkong'))
for col, label in ALL:
    row = []
    for city in ['shanghai', 'hongkong']:
        s = d[d['city'] == city][[col, 'log_pm25']].dropna()
        row.append(np.corrcoef(s[col], s['log_pm25'])[0, 1]
                   if len(s) > 50 else np.nan)
    tag = '  <- contains surface T' if col.count('_') == 1 else ''
    print('   %-22s %+10.4f %+10.4f%s' % (label, row[0], row[1], tag))

# Three control sets. The middle one is the point of this step: it keeps
# every mood-channel control including temperature, which only the aloft
# measures can accept without mechanical contamination.
SETS = [
    ('minimal      ', ['temperature_2m', 'precipitation']),
    ('no surface T ', ['precipitation', 'wind_speed_10m',
                       'cloud_cover', 'shortwave_radiation']),
    ('full         ', ['temperature_2m', 'precipitation', 'wind_speed_10m',
                       'cloud_cover', 'shortwave_radiation']),
]


def first_stage(sub, col, controls):
    sub = sub.dropna(subset=[col, 'log_pm25'] + controls).copy()
    if len(sub) < 80:
        return None
    fe = pd.get_dummies(sub[['month', 'year', 'dow']].astype(str),
                        drop_first=True).astype(float)
    X = sm.add_constant(pd.concat(
        [sub[[col] + controls].reset_index(drop=True),
         fe.reset_index(drop=True)], axis=1))
    y = sub['log_pm25'].reset_index(drop=True)
    full = sm.OLS(y, X).fit(cov_type='HC1')
    red = sm.OLS(y, X.drop(columns=[col])).fit(cov_type='HC1')
    pr2 = (full.rsquared - red.rsquared) / (1 - red.rsquared)
    return full.params[col], full.tvalues[col], pr2, int(full.nobs)


survivors = {}
for city in ['shanghai', 'hongkong']:
    print('\n' + '=' * 74)
    print('%s   (want POSITIVE)' % city.upper())
    print('=' * 74)
    print('   %-22s %-14s %9s %7s %10s %6s %s'
          % ('measure', 'controls', 'coef', 't', 'partial R2', 'N', ''))
    sub = d[d['city'] == city]
    for col, label in ALL:
        ok_all = True
        for sname, ctrl in SETS:
            # Surface-based measures cannot accept temperature_2m.
            if col.count('_') == 1 and 'temperature_2m' in ctrl:
                print('   %-22s %-14s %9s   (skipped: regressor contains T_sfc)'
                      % (label, sname, '-'))
                ok_all = False
                continue
            r = first_stage(sub, col, ctrl)
            if r is None:
                print('   %-22s %-14s %9s' % (label, sname, 'n/a'))
                ok_all = False
                continue
            b, t, pr2, n = r
            v = ('ns' if abs(t) < 1.96 else 'CORRECT' if b > 0 else 'WRONG')
            if not (abs(t) >= 1.96 and b > 0):
                ok_all = False
            print('   %-22s %-14s %+9.4f %7.2f %10.4f %6d %s'
                  % (label, sname, b, t, pr2, n, v))
        if ok_all:
            survivors.setdefault(city, []).append(label)
        print()

print('=' * 74)
print('SURVIVORS: positive and significant under every control set tried')
print('=' * 74)
for city in ['shanghai', 'hongkong']:
    s = survivors.get(city, [])
    print('   %-10s %s' % (city, ', '.join(s) if s else 'NONE'))

out = os.path.join(DATA, 'aloft_inversion_panel.csv')
d.to_csv(out, index=False)
print('\nwritten: %s' % out)

print("""
WHAT THIS DECIDES
  An aloft measure surviving in Hong Kong under 'no surface T' and 'full':
      Hong Kong is instrumentable after all, the bilateral design is back
      on both sides, and step 7's failure was my control set, not physics.
  Nothing surviving in Hong Kong:
      Hong Kong is settled as uninstrumentable across six independent
      dispersion measures and three control sets. The design goes
      one-sided on Shanghai and that failure becomes a documented result.
""")
