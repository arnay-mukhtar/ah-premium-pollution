# Source: paper2_meteorology_build.ipynb, code cells 0-12 (ERA5/CDS and Open-Meteo downloads, HK EPD parsing).
# Cells are separated by "# %%". Notebook shell commands (lines starting with "!") must be run in Colab or removed.

# %%
from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT = '/content/drive/MyDrive/bilateral'
DATA = os.path.join(PROJECT, 'data')
RAW  = os.path.join(DATA, 'raw_cds')
for d in (PROJECT, DATA, RAW):
    os.makedirs(d, exist_ok=True)

print('project folder:', PROJECT)

# %%
!pip install -q cdsapi xarray netcdf4
print('done')

# %%
import time
import numpy as np
import pandas as pd
import requests

START, END = '2013-11-01', '2023-03-31'

CITIES = {
    'shanghai': {'lat': 31.2304, 'lon': 121.4737, 'tz': 'Asia/Shanghai',
                 'area': [31.90, 120.90, 30.70, 122.00]},
    'hongkong': {'lat': 22.3193, 'lon': 114.1694, 'tz': 'Asia/Hong_Kong',
                 'area': [22.56, 113.83, 22.15, 114.44]},
}

FIELDS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
          'wind_direction_10m', 'surface_pressure', 'cloud_cover',
          'shortwave_radiation']

for name, c in CITIES.items():
    out = os.path.join(DATA, name + '_weather_hourly.csv')
    if os.path.exists(out):
        print(name, 'already built, skipping')
        continue

    print('fetching', name, '...')
    r = requests.get('https://archive-api.open-meteo.com/v1/archive',
                     params={'latitude': c['lat'], 'longitude': c['lon'],
                             'start_date': START, 'end_date': END,
                             'hourly': ','.join(FIELDS),
                             'timezone': c['tz']},
                     timeout=600)
    r.raise_for_status()

    w = pd.DataFrame(r.json()['hourly'])
    w['time'] = pd.to_datetime(w['time'])
    w['hour'] = w['time'].dt.hour
    w['date'] = w['time'].dt.normalize()

    rad = np.deg2rad(w['wind_direction_10m'])
    w['u'] = -w['wind_speed_10m'] * np.sin(rad)
    w['v'] = -w['wind_speed_10m'] * np.cos(rad)

    w.to_csv(out, index=False)
    print('  %s hourly obs -> %s' % (format(len(w), ','), out))
    for f in FIELDS:
        n = int(w[f].isna().sum())
        if n:
            print('  %s: %s missing hours' % (f, format(n, ',')))
    time.sleep(2)

# %%
CDS_KEY = os.environ.get('CDS_KEY', '')   # set your own Copernicus CDS key in the environment

with open('/root/.cdsapirc', 'w') as f:
    f.write('url: https://cds.climate.copernicus.eu/api\n')
    f.write('key: ' + CDS_KEY + '\n')

import cdsapi
client = cdsapi.Client()
print('client OK')

# %%
YEARS  = [str(y) for y in range(2013, 2024)]
MONTHS = ['01', '02', '03', '11', '12']   # winter; widen if the sample changes
DAYS   = ['%02d' % d for d in range(1, 32)]
TIMES  = ['%02d:00' % h for h in range(24)]

def get_blh(name, area, year):
    target = os.path.join(RAW, '%s_blh_%s.nc' % (name, year))
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target
    print('  requesting', name, year, '...')
    client.retrieve('reanalysis-era5-single-levels', {
        'product_type': ['reanalysis'],
        'variable': ['boundary_layer_height'],
        'year': [year], 'month': MONTHS, 'day': DAYS, 'time': TIMES,
        'area': area,
        'data_format': 'netcdf', 'download_format': 'unarchived',
    }, target)
    return target

def get_lsm(name, area):
    target = os.path.join(RAW, '%s_lsm.nc' % name)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target
    print('  requesting land-sea mask for', name, '...')
    client.retrieve('reanalysis-era5-single-levels', {
        'product_type': ['reanalysis'],
        'variable': ['land_sea_mask'],
        'year': ['2020'], 'month': ['01'], 'day': ['01'], 'time': ['00:00'],
        'area': area,
        'data_format': 'netcdf', 'download_format': 'unarchived',
    }, target)
    return target

for name, c in CITIES.items():
    print(name)
    get_lsm(name, c['area'])
    for y in YEARS:
        get_blh(name, c['area'], y)
print('all chunks present')

# %%
import glob
import xarray as xr

LOCAL_TZ = 'Asia/Shanghai'   # UTC+8, correct for both cities
LAND_THRESHOLD = 0.5

# Hong Kong general stations, converted from HK1980 Grid to WGS84.
# Your series uses 13 of these. Delete the two you did not use once you have
# checked the raw file; the weighting barely moves either way.
HK_STATIONS = {
    'Central/Western': (22.2864, 114.1419),
    'Eastern':         (22.2844, 114.2169),
    'Kwai Chung':      (22.3587, 114.1271),
    'Kwun Tong':       (22.3112, 114.2287),
    'North':           (22.4983, 114.1258),
    'Sha Tin':         (22.3778, 114.1821),
    'Sham Shui Po':    (22.3318, 114.1566),
    'Southern':        (22.2490, 114.1577),
    'Tai Po':          (22.4525, 114.1621),
    'Tap Mun':         (22.4730, 114.3581),
    'Tseung Kwan O':   (22.3192, 114.2571),
    'Tsuen Wan':       (22.3733, 114.1121),
    'Tuen Mun':        (22.3928, 113.9743),
    'Tung Chung':      (22.2906, 113.9411),
    'Yuen Long':       (22.4468, 114.0203),
}

def tname(ds):
    return 'valid_time' if 'valid_time' in ds.coords else 'time'

for name, c in CITIES.items():
    print(name)
    files = sorted(glob.glob(os.path.join(RAW, '%s_blh_*.nc' % name)))
    print('  %d yearly files' % len(files))

    parts = []
    for f in files:
        d = xr.open_dataset(f)
        v = 'blh' if 'blh' in d else list(d.data_vars)[0]
        parts.append(d[v])
    tdim = parts[0].dims[0]
    blh = xr.concat(parts, dim=tdim).sortby(tdim)

    lsm_ds = xr.open_dataset(os.path.join(RAW, '%s_lsm.nc' % name))
    lv = 'lsm' if 'lsm' in lsm_ds else list(lsm_ds.data_vars)[0]
    lsm = lsm_ds[lv]
    if tname(lsm_ds) in lsm.dims:
        lsm = lsm.isel({tname(lsm_ds): 0})

    n_total = int(lsm.size)
    land = lsm >= LAND_THRESHOLD
    n_land = int(land.sum())
    print('  grid cells: %d, land cells: %d' % (n_total, n_land))
    if n_land == 0:
        print('  WARNING: no land cells, falling back to all cells')
        land = xr.ones_like(lsm, dtype=bool)

    dims = ['latitude', 'longitude']
    cols = {}
    cols['blh_land_mean'] = blh.where(land).mean(dim=dims, skipna=True).values
    cols['blh_all_mean']  = blh.mean(dim=dims, skipna=True).values
    point = blh.sel(latitude=c['lat'], longitude=c['lon'], method='nearest')
    cols['blh_point'] = point.values
    print('  nearest grid point: %.2f N, %.2f E' %
          (float(point.latitude), float(point.longitude)))

    # Hong Kong only: build the instrument on the same stations as the
    # treatment, so the two share a spatial support by construction.
    if name == 'hongkong':
        series = [blh.sel(latitude=la, longitude=lo, method='nearest')
                  for la, lo in HK_STATIONS.values()]
        cols['blh_station_mean'] = (sum(series) / len(series)).values
        print('  station-matched instrument built on %d monitors'
              % len(HK_STATIONS))

    df = pd.DataFrame({'time_utc': pd.to_datetime(blh[tdim].values, utc=True)})
    for k, v in cols.items():
        df[k] = v

    df['time'] = df['time_utc'].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    df['hour'] = df['time'].dt.hour
    df['date'] = df['time'].dt.normalize()
    df = df.sort_values('time').drop_duplicates('time')

    base = 'blh_station_mean' if 'blh_station_mean' in df else 'blh_land_mean'
    print('  primary series: %s' % base)
    for other in [k for k in cols if k != base]:
        sub = df[[base, other]].dropna()
        r  = np.corrcoef(sub[base], sub[other])[0, 1]
        lr = np.corrcoef(np.log(sub[base].clip(lower=1)),
                         np.log(sub[other].clip(lower=1)))[0, 1]
        print('  corr(%s, %s) = %.4f   logs %.4f' % (base, other, r, lr))

    out = os.path.join(DATA, '%s_blh_hourly.csv' % name)
    keep = ['time', 'date', 'hour'] + list(cols.keys())
    df[keep].to_csv(out, index=False)
    print('  %s hourly obs -> %s' % (format(len(df), ','), out))
    print('  missing (%s): %s' % (base, format(int(df[base].isna().sum()), ',')))


# %%
import xarray as xr
lsm_ds = xr.open_dataset(os.path.join(RAW, 'hongkong_lsm.nc'))
lv = 'lsm' if 'lsm' in lsm_ds else list(lsm_ds.data_vars)[0]
lsm = lsm_ds[lv]
if ('valid_time' in lsm.dims) or ('time' in lsm.dims):
    lsm = lsm.isel({[d for d in lsm.dims if d in ('valid_time','time')][0]: 0})

print('Hong Kong box, land fraction per cell:')
for la in lsm.latitude.values:
    for lo in lsm.longitude.values:
        print('  %.2f N, %.2f E  land fraction %.3f' %
              (la, lo, float(lsm.sel(latitude=la, longitude=lo))))


# %%
import glob, os
import pandas as pd

paths = sorted(glob.glob('/content/drive/MyDrive/**/*.csv', recursive=True))
print('%d csv files on Drive\n' % len(paths))
for p in paths:
    print('%8.1f MB  %s' % (os.path.getsize(p)/1e6, p))


# %%
for p in ['/content/drive/MyDrive/pollution-ah/cn_pm25_daily.csv',
          '/content/drive/MyDrive/pollution-ah/hk_pm25_2013_2017.csv']:
    df = pd.read_csv(p, nrows=6)
    print('\n===', p)
    print('columns:', list(df.columns))
    print(df.to_string())


# %%
p = '/content/drive/MyDrive/pollution-ah/hk_pm25_2013_2017.csv'

with open(p, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print('total lines:', len(lines))
print('\n--- first 25 ---')
for i, l in enumerate(lines[:25]):
    print(i, repr(l[:160]))
print('\n--- lines 25 to 35 ---')
for i, l in enumerate(lines[25:35], start=25):
    print(i, repr(l[:160]))


# %%
p = '/content/drive/MyDrive/pollution-ah/hk_pm25_2013_2017.csv'

with open(p, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print('total lines:', len(lines))
print('\n--- first 25 ---')
for i, l in enumerate(lines[:25]):
    print(i, repr(l[:160]))
print('\n--- lines 25 to 35 ---')
for i, l in enumerate(lines[25:35], start=25):
    print(i, repr(l[:160]))


# %%
import glob

hk_files = sorted(glob.glob('/content/drive/MyDrive/pollution-ah/hk_pm25_*.csv'))
frames = []
for p in hk_files:
    df = pd.read_csv(p, skiprows=11, encoding='utf-8-sig',
                     na_values=['N.A.', 'N.A', ''], low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    frames.append(df)
    print(p.split('/')[-1], df.shape)

raw = pd.concat(frames, ignore_index=True)

print('\nFULL COLUMN LIST')
for c in raw.columns:
    print('  ', c)

print('\npollutants:', sorted(raw['POLLUTANT'].astype(str).str.strip().unique()))
print('hours:', sorted(raw['HOUR'].dropna().unique()))

first = raw['DATE'].astype(str).str.split('/').str[0].astype(int)
second = raw['DATE'].astype(str).str.split('/').str[1].astype(int)
print('max first component:', first.max(), ' max second:', second.max())
print('=> dayfirst should be', first.max() > 12)

