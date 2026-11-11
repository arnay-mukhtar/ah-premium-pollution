# Source: paper2_meteorology_build.ipynb, code cell 22 (final version of step 10 as run).
# ============================================================
# Step 10 — rebuild the mainland PM2.5 series, NDJFM, two windows
#
# WHY. cn_pm25_daily.csv is December/January/February by design (the
# MONTHS list in 01_build_pollution.ipynb cell 2). The ERA5 and sounding
# work covers November to March, so Shanghai currently contributes 555
# days against Hong Kong's 980, purely from the month list.
#
# AND. The Zenodo files are HOURLY. The original build collapsed them to
# a 24-hour mean, then step 4 applied a 07:00-16:00 window to Hong Kong
# alone, which is an asymmetry that should not exist. Both series are
# built here, so the window becomes a pre-registered choice:
#     pm_24h   24-hour mean, >=8 valid hours      (matches the old build)
#     pm_win   07:00-16:00 mean, >=7 of 9 hours   (matches paper 1's logic)
#
# Cleaning is copied from cell 2 unchanged, so the DJF months reproduce
# the existing file rather than quietly differing from it.
#
# Pollution only. Nothing touches the A-H ratio.
# ============================================================

import os
import urllib.request

import numpy as np
import pandas as pd
import xarray as xr

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DIR = '/content/drive/MyDrive/pollution-ah'
OUT = '/content/drive/MyDrive/bilateral/data'
RAW = '/content/cn_raw'            # ephemeral; only the CSVs are persisted
os.makedirs(RAW, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

BASE = 'https://zenodo.org/records/6404603/files'

# November to March, 2013-14 through 2019-20.
# March 2020 is deliberately excluded: COVID lockdowns make it a
# different pollution regime and the sample ends 2020-02-28 anyway.
MONTHS = []
for y in range(2013, 2020):
    MONTHS += ['%d11' % y, '%d12' % y]
    MONTHS += ['%d01' % (y + 1), '%d02' % (y + 1), '%d03' % (y + 1)]
MONTHS = sorted(set(m for m in MONTHS if '201301' <= m <= '202002'))
print('%d months: %s ... %s' % (len(MONTHS), MONTHS[0], MONTHS[-1]))

WIN_START, WIN_END = 7, 16     # 07:00 to 15:59 local
MIN_WIN_HOURS = 7              # 7 of 9
MIN_DAY_HOURS = 8              # matches the original build

frames_24, frames_win, missing = [], [], []

for m in MONTHS:
    p = os.path.join(RAW, 'aqi%s.nc' % m)
    if not os.path.exists(p):
        try:
            urllib.request.urlretrieve('%s/aqi%s.nc?download=1' % (BASE, m), p)
        except Exception as e:
            missing.append((m, type(e).__name__))
            print('  %s: NOT AVAILABLE (%s)' % (m, type(e).__name__))
            continue

    ds = xr.open_dataset(p)
    area = np.array([a.decode('utf-8', 'ignore').strip()
                     for a in ds.area.values])
    t = pd.to_datetime([s.decode().strip() for s in ds.time_point.values],
                       format='%Y-%m-%dT%H:%M:%SZ', utc=True)

    pm = ds.pm2_5.values.astype('float64')
    pm[(pm > 1e6) | (pm < 0)] = np.nan          # fill value is not auto-masked

    X = pd.DataFrame(pm, index=t, columns=area)
    X = X[~X.index.duplicated(keep='first')]
    X.index = X.index.tz_convert('Asia/Shanghai')
    X = X[(X.index.year == int(m[:4])) & (X.index.month == int(m[4:]))]

    hourly = X.T.groupby(level=0).mean().T          # stations -> city

    g = hourly.groupby(hourly.index.normalize())
    frames_24.append(g.mean().where(g.count() >= MIN_DAY_HOURS))

    w = hourly[(hourly.index.hour >= WIN_START) &
               (hourly.index.hour < WIN_END)]
    gw = w.groupby(w.index.normalize())
    frames_win.append(gw.mean().where(gw.count() >= MIN_WIN_HOURS))

    ds.close()
    print('  %s: %d cities, %d days' % (m, hourly.shape[1], len(g)))

if missing:
    print('\nmonths unavailable: %s' % missing)


def finish(frames, name):
    C = pd.concat(frames).sort_index()
    C.index = C.index.tz_localize(None)
    C.index.name = 'date'
    # Same ambiguous-name drop as notebook cell 5.
    import re
    AMBIG = ['Bozhou', 'Fuyang', 'Fuzhou', 'Suzhou', 'Taizhou']
    drop = [c for c in C.columns if c.rsplit('_', 1)[0] in AMBIG]
    C = C.drop(columns=drop)
    path = os.path.join(OUT, name)
    C.to_csv(path)
    print('\n%s: %d cities, %d days, %s to %s'
          % (name, C.shape[1], len(C), C.index.min().date(),
             C.index.max().date()))
    for c in ['Shanghai', 'Shenzhen']:
        if c in C.columns:
            s = C[c].dropna()
            print('   %-9s %4d days   mean %5.1f   max %6.1f'
                  % (c, len(s), s.mean(), s.max()))
    return C


C24 = finish(frames_24, 'cn_pm25_daily_24h.csv')
CW = finish(frames_win, 'cn_pm25_daily_win.csv')

# ------------------------------------------------------------
# Does the 24h rebuild reproduce the existing DJF file?
# ------------------------------------------------------------
old = pd.read_csv(os.path.join(DIR, 'cn_pm25_daily.csv'),
                  index_col='date', parse_dates=True)
j = pd.DataFrame({'old': old['Shanghai'], 'new': C24['Shanghai']}).dropna()
print('\nreproduction check against the existing file (Shanghai, %d'
      ' overlapping days):' % len(j))
if len(j):
    print('   correlation    %.6f' % j['old'].corr(j['new']))
    print('   max abs diff   %.4f ug/m3' % (j['old'] - j['new']).abs().max())
    print('   (want correlation 1.000000 and a difference near zero;'
          ' anything else means the cleaning drifted)')

# ------------------------------------------------------------
# What the extension buys, and how the two windows compare
# ------------------------------------------------------------
print('\nShanghai days by month, 24h series:')
s = C24['Shanghai'].dropna()
print(s.groupby(s.index.month).size().to_string())

k = pd.DataFrame({'h24': C24['Shanghai'], 'win': CW['Shanghai']}).dropna()
print('\n24h versus windowed (Shanghai, %d days):' % len(k))
print('   correlation      %.4f' % k['h24'].corr(k['win']))
print('   mean  24h %.1f   windowed %.1f' % (k['h24'].mean(), k['win'].mean()))
print('   sd    24h %.1f   windowed %.1f' % (k['h24'].std(), k['win'].std()))

print("""
NEXT
  Shanghai should land near 900 days against the present 555. Send the
  reproduction check first: if the 24h rebuild does not match the
  existing file almost exactly, the cleaning drifted and nothing
  downstream is trustworthy.
""")
