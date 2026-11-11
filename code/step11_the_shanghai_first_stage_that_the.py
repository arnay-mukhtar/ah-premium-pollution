# Source: paper2_meteorology_build.ipynb, code cell 23 (final version of step 11 as run).
# ============================================================
# Step 11 — the Shanghai first stage that the paper rests on
#
# Run after step 10. Standalone.
#
# Hong Kong is out as an instrumented side (six measure families, three
# control sets, nothing survives). The design is one-sided: instrument
# Shanghai pollution, outcome is the A-H premium. So the viability of
# the whole paper is one number, the strength of the Shanghai first
# stage, and this computes it properly.
#
# Three instrument sets x two window definitions:
#   ERA5       log BLH, u, v, surface pressure        (window means)
#   soundings  T(200)-T(100), T(1000)-T(300)          (08:00, one per day)
#   combined   both
#
#   24h   24-hour pollution mean, matching the original build
#   win   07:00-16:00 mean, matching paper 1's market-timing argument
#
# Reports the robust joint F AND the Montiel Olea-Pflueger effective F,
# which is the one referees ask for when instruments may be weak.
#
# Also saves sounding_daily.csv so the profiles never need re-parsing.
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
RAW = os.path.join(DATA, 'raw_igra')

WIN_START, WIN_END = 7, 16
MIN_WIN_HOURS = 7
MIN_DAY_HOURS = 8
SAMPLE_START, SAMPLE_END = '2013-11-01', '2020-02-29'

# ------------------------------------------------------------
# 1. Sounding measures for every date, saved once
# ------------------------------------------------------------
SND = os.path.join(DATA, 'sounding_daily.csv')

if os.path.exists(SND):
    snd = pd.read_csv(SND, parse_dates=['date'])
    print('sounding measures loaded: %d rows' % len(snd))
else:
    print('parsing soundings (one time, then cached) ...')
    G, RD, KAPPA, ZTOP, DZ = 9.80665, 287.05, 0.286, 4000.0, 25.0
    GRID = np.arange(0.0, ZTOP + DZ, DZ)
    MISSV = (-9999, -8888)

    def fld(ln, a, b, scale=1.0):
        s = ln[a:b]
        if not s.strip():
            return np.nan
        try:
            v = int(s)
        except ValueError:
            return np.nan
        return np.nan if v in MISSV else v / scale

    def parse(path, y0=2013, y1=2021):
        cur, rows = None, []
        with open(path, encoding='latin-1') as f:
            for ln in f:
                if ln.startswith('#'):
                    if cur is not None and rows:
                        yield cur, pd.DataFrame(rows)
                    rows = []
                    yr = int(ln[13:17])
                    if not (y0 <= yr <= y1):
                        cur = None
                        continue
                    cur = (yr, int(ln[18:20]), int(ln[21:23]), int(ln[24:26]))
                    continue
                if cur is None:
                    continue
                rows.append({'p': fld(ln, 9, 15, 100.0), 'z': fld(ln, 16, 21),
                             'T': fld(ln, 22, 27, 10.0),
                             'dpdp': fld(ln, 34, 39, 10.0)})
        if cur is not None and rows:
            yield cur, pd.DataFrame(rows)

    def thermo(d):
        d = d.dropna(subset=['p', 'T']).copy()
        d = d[(d['p'] > 100) & (d['p'] < 1100)]
        d = d[(d['T'] > -90) & (d['T'] < 60)]
        d = d.sort_values('p', ascending=False).drop_duplicates('p')
        if len(d) < 8:
            return None
        p = d['p'].values
        t = d['T'].values + 273.15
        td = d['T'].values - np.nan_to_num(d['dpdp'].values, nan=0.0)
        e = 6.112 * np.exp(17.67 * td / (td + 243.5))
        q = 0.622 * e / (p - 0.378 * e)
        tv = t * (1 + 0.608 * q)
        z = np.zeros_like(p)
        z[1:] = np.cumsum((RD / G) * 0.5 * (tv[:-1] + tv[1:])
                          * np.log(p[:-1] / p[1:]))
        keep = z <= ZTOP
        if keep.sum() < 8:
            return None
        return pd.DataFrame({
            'zagl': GRID,
            'T': np.interp(GRID, z[keep], d['T'].values[keep],
                           left=np.nan, right=np.nan)}).dropna(subset=['T'])

    def at_h(g, h):
        s = g.loc[(g['zagl'] - h).abs() < 1e-6, 'T']
        return float(s.iloc[0]) if len(s) else np.nan

    STATIONS = {'hongkong': '45004', 'shanghai': '58362'}
    lines = open(os.path.join(RAW, 'igra2-station-list.txt'),
                 encoding='latin-1').read().splitlines()
    rows = []
    for city, wmo in STATIONS.items():
        sid = [ln.split()[0] for ln in lines if wmo in ln[:12]][0]
        path = os.path.join(RAW, '%s-data.txt' % sid)
        n = 0
        for (yr, mo, dy, hr), lev in parse(path):
            if hr != 0:
                continue
            g = thermo(lev)
            if g is None:
                continue
            t = {h: at_h(g, h) for h in (100, 200, 300, 500, 1000)}
            rows.append({'city': city, 'date': pd.Timestamp(yr, mo, dy),
                         'dT_200_100': t[200] - t[100],
                         'dT_300_100': t[300] - t[100],
                         'dT_500_100': t[500] - t[100],
                         'dT_1000_300': t[1000] - t[300]})
            n += 1
        print('  %-9s %d morning profiles' % (city, n))
    snd = pd.DataFrame(rows)
    snd.to_csv(SND, index=False)
    print('cached -> %s' % SND)

# ------------------------------------------------------------
# 2. Pollution, both windows
# ------------------------------------------------------------
poll = {}
for tag, fn in [('24h', 'cn_pm25_daily_24h.csv'),
                ('win', 'cn_pm25_daily_win.csv')]:
    C = pd.read_csv(os.path.join(DATA, fn), index_col='date', parse_dates=True)
    poll[tag] = C['Shanghai'].rename('pm25').reset_index()
    print('%s: %d Shanghai days' % (tag, poll[tag]['pm25'].notna().sum()))

# ------------------------------------------------------------
# 3. Meteorology, aggregated to match each window
# ------------------------------------------------------------
w = pd.read_csv(os.path.join(DATA, 'shanghai_weather_hourly.csv'),
                parse_dates=['time', 'date'])
b = pd.read_csv(os.path.join(DATA, 'shanghai_blh_hourly.csv'),
                parse_dates=['time', 'date'])
met_h = w.merge(b.drop(columns=['date', 'hour']), on='time', how='inner')

FIELDS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
          'surface_pressure', 'cloud_cover', 'shortwave_radiation', 'u', 'v',
          'blh_land_mean', 'blh_all_mean']
FIELDS = [f for f in FIELDS if f in met_h.columns]

met = {}
for tag, lo, hi, need in [('24h', 0, 24, MIN_DAY_HOURS),
                          ('win', WIN_START, WIN_END, MIN_WIN_HOURS)]:
    m = met_h[(met_h['hour'] >= lo) & (met_h['hour'] < hi)]
    g = m.groupby('date')
    d = g[FIELDS].mean().reset_index()
    d['n_hours'] = g.size().values
    met[tag] = d[d['n_hours'] >= need]
    print('%s meteorology: %d days' % (tag, len(met[tag])))

# ------------------------------------------------------------
# 4. Montiel Olea-Pflueger effective F
# ------------------------------------------------------------
def effective_f(y, Z, W):
    """F_eff for one endogenous regressor.

    Controls are partialled out first (FWL), then
        F_eff = pi' (Z'Z) pi / tr( (Z'Z) V_robust(pi) )
    which reduces to the classical F under homoskedasticity, and is the
    statistic Montiel Olea & Pflueger tabulate critical values for.
    """
    Wc = sm.add_constant(W, has_constant='add')
    P = Wc.values @ np.linalg.pinv(Wc.values)
    yt = y.values - P @ y.values
    Zt = Z.values - P @ Z.values

    ZtZ = Zt.T @ Zt
    pi = np.linalg.solve(ZtZ, Zt.T @ yt)
    e = yt - Zt @ pi
    n, k = Zt.shape
    kw = Wc.shape[1]
    bread = np.linalg.inv(ZtZ)
    meat = Zt.T @ (Zt * (e ** 2)[:, None])
    V = bread @ meat @ bread * (n / max(n - k - kw, 1))
    return float(pi @ ZtZ @ pi / np.trace(ZtZ @ V))


CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']
SETS = {
    'ERA5      ': ['log_blh', 'u', 'v', 'surface_pressure'],
    'soundings ': ['dT_200_100', 'dT_1000_300'],
    'combined  ': ['log_blh', 'u', 'v', 'surface_pressure',
                   'dT_200_100', 'dT_1000_300'],
}

print('\n' + '=' * 78)
print('SHANGHAI FIRST STAGE   outcome: log PM2.5')
print('=' * 78)
print('%-6s %-11s %6s %9s %9s %11s' %
      ('window', 'instruments', 'N', 'joint F', 'F_eff', 'partial R2'))

store = {}
for tag in ('24h', 'win'):
    d = met[tag].merge(poll[tag], on='date', how='inner')
    d = d.merge(snd[snd['city'] == 'shanghai'].drop(columns='city'),
                on='date', how='left')
    d = d[(d['date'] >= SAMPLE_START) & (d['date'] <= SAMPLE_END)]
    d = d[d['pm25'] > 0].copy()
    d['log_pm25'] = np.log(d['pm25'])
    d['log_blh'] = np.log(d['blh_land_mean'].clip(lower=1))
    d['month'] = d['date'].dt.month
    d['year'] = d['date'].dt.year
    d['dow'] = d['date'].dt.dayofweek
    store[tag] = d

    for name, inst in SETS.items():
        s = d.dropna(subset=inst + ['log_pm25'] + CONTROLS).copy()
        if len(s) < 100:
            print('%-6s %-11s %6s' % (tag, name, 'n/a'))
            continue
        fe = pd.get_dummies(s[['month', 'year', 'dow']].astype(str),
                            drop_first=True).astype(float)
        W = pd.concat([s[CONTROLS].reset_index(drop=True),
                       fe.reset_index(drop=True)], axis=1)
        X = sm.add_constant(pd.concat([s[inst].reset_index(drop=True), W],
                                      axis=1))
        y = s['log_pm25'].reset_index(drop=True)

        full = sm.OLS(y, X).fit(cov_type='HC1')
        red = sm.OLS(y, X.drop(columns=inst)).fit(cov_type='HC1')
        pr2 = (full.rsquared - red.rsquared) / (1 - red.rsquared)

        R = np.zeros((len(inst), X.shape[1]))
        for i, nm in enumerate(inst):
            R[i, list(X.columns).index(nm)] = 1.0
        jf = float(np.squeeze(full.f_test(R).fvalue))
        fe_eff = effective_f(y, s[inst].reset_index(drop=True), W)

        print('%-6s %-11s %6d %9.2f %9.2f %11.4f'
              % (tag, name, int(full.nobs), jf, fe_eff, pr2))
    print()

# ------------------------------------------------------------
# 5. How much does the window choice actually matter?
# ------------------------------------------------------------
k = store['24h'][['date', 'log_pm25']].merge(
    store['win'][['date', 'log_pm25']], on='date', suffixes=('_24h', '_win'))
print('window comparison on %d common days:' % len(k))
print('   corr(log PM2.5 24h, windowed)  %.4f'
      % k['log_pm25_24h'].corr(k['log_pm25_win']))

for tag in ('24h', 'win'):
    d = store[tag]
    print('   %s: %d days, %s to %s'
          % (tag, len(d), d['date'].min().date(), d['date'].max().date()))

print("""
READING IT
  F_eff is the number that matters. Montiel Olea & Pflueger's critical
  values depend on the instrument count and the bias you will tolerate;
  look up the exact one for your k rather than using 10 as a rule of
  thumb, since 10 comes from Stock-Yogo under homoskedasticity and does
  not apply here.

  If 'combined' clears comfortably in both windows, the design is viable
  and the window choice is yours to make on the market-timing argument.
  If only one window clears, the pre-registration has to justify that
  window on physical grounds BEFORE this result is cited, or it is
  specification search.
""")
