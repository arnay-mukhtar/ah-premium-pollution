# ============================================================
# Step 51: the Zenodo robustness row on the FULL registered sample
# The Zenodo series on Drive (cn_pm25_daily_24h.csv) covers November-
# March only, so step 47's Zenodo row ran on cold-season days alone.
# This downloads the hourly Zenodo files for every month May 2014 -
# December 2019 (record 6404603), keeps Shanghai only, builds the
# 24-hour station average exactly as step 10 did, saves it as a new
# file, and reruns the row. Download-heavy: allow 20-60 minutes.
# ============================================================
import os, numpy as np, pandas as pd, requests, xarray as xr
from scipy import stats
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass
DATA = '/content/drive/MyDrive/bilateral/data'
START, END = '2014-05-13', '2019-12-31'
LAGS, SESOI = 7, 25.0
INST = ['log_blh', 'u', 'v', 'surface_pressure']
CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']
def dk(S, lags):
    om = S.T @ S
    for j in range(1, int(lags) + 1):
        if j >= len(S):
            break
        G = S[j:].T @ S[:-j]
        om += (1 - j / (lags + 1.0)) * (G + G.T)
    return om


def tsls(y, X, Z, W, codes, G, lags=LAGS, w=None):
    n = len(y)
    X = X.reshape(n, -1)
    Zf = np.column_stack([np.ones(n), Z, W])
    Xf = np.column_stack([X, np.ones(n), W])
    P, *_ = np.linalg.lstsq(Zf, Xf, rcond=None)
    Xh = Zf @ P
    A = Xh.T @ Xf
    b = np.linalg.solve(A, Xh.T @ y)
    e = y - Xf @ b
    S = np.zeros((G, Xh.shape[1]))
    np.add.at(S, codes, Xh * e[:, None])
    Ai = np.linalg.inv(A)
    V = Ai @ dk(S, lags) @ Ai.T * (G / (G - 1)) * ((n - 1) / (n - Xf.shape[1]))
    k = X.shape[1]
    return b[:k], np.sqrt(np.diag(V)[:k])


def ar_curve(y, x, Z, W, codes, G, grid, lags=LAGS):
    X = np.column_stack([np.ones(len(y)), Z, W])
    n, kt = X.shape
    k = Z.shape[1]
    Q, R_ = np.linalg.qr(X)
    XtXi = np.linalg.inv(X.T @ X)
    adj = (G / (G - 1)) * ((n - 1) / (n - kt))
    out = np.empty(len(grid))
    for j, b0 in enumerate(grid):
        yt = y - b0 * x
        b = np.linalg.solve(R_, Q.T @ yt)
        e = yt - X @ b
        S = np.zeros((G, kt))
        np.add.at(S, codes, X * e[:, None])
        V = XtXi @ dk(S, lags) @ XtXi * adj
        Rb = b[1:1 + k]
        F = Rb @ np.linalg.solve(V[1:1 + k, 1:1 + k], Rb) / k
        out[j] = 1 - stats.f.cdf(F, k, G - 1)
    return out


def ar_sets(y, x, Z, W, codes, G, b, se):
    hw = 8 * se
    for _ in range(4):
        grid = np.linspace(b - hw, b + hw, 401)
        p = ar_curve(y, x, Z, W, codes, G, grid)
        res = {}
        for a in (0.05, 0.10):
            keep = p > a
            if not keep.any():
                res[a] = ('empty', None, None)
            elif keep[0] or keep[-1]:
                res[a] = None
            else:
                i = np.where(keep)[0]
                res[a] = ('disjoint' if (np.diff(i) > 1).any() else 'ok',
                          grid[i[0]], grid[i[-1]])
        if all(v is not None for v in res.values()):
            return res, p.max()
        hw *= 4
    return {a: ('unbounded', None, None) for a in (0.05, 0.10)}, p.max()


def eff_f(x, Z, W):
    Wc = np.column_stack([np.ones(len(x)), W])
    M = lambda A: A - Wc @ np.linalg.lstsq(Wc, A, rcond=None)[0]
    xt, Zt = M(x), M(Z)
    ZtZ = Zt.T @ Zt
    pi = np.linalg.solve(ZtZ, Zt.T @ xt)
    e = xt - Zt @ pi
    n, k = Zt.shape
    br = np.linalg.inv(ZtZ)
    V = br @ (Zt.T @ (Zt * (e ** 2)[:, None])) @ br * (n / max(n - k - Wc.shape[1], 1))
    return float(pi @ ZtZ @ pi / np.trace(ZtZ @ V))



def cal(s):
    s = s.copy()
    s['month'] = s['date'].dt.month
    s['year'] = s['date'].dt.year
    s['dow'] = s['date'].dt.dayofweek
    return s


def build(s, endog, inst, ctrl, outcome):
    s = cal(s).dropna(subset=endog + inst + ctrl + [outcome])
    s = s.sort_values('date')
    fe = pd.get_dummies(s[['month', 'year', 'dow']].astype(str),
                        drop_first=True).astype(float)
    W = np.column_stack([s[ctrl].values, fe.values])
    _, codes = np.unique(s['date'].values, return_inverse=True)
    return s, W, codes, codes.max() + 1


rows = []


def run(label, s, outcome='d_prem_raw', inst=INST, ctrl=CONTROLS,
        endog='log_pm25', weight=None, note=''):
    try:
        s, W, codes, G = build(s, [endog], inst, ctrl, outcome)
        y, x, Z = s[outcome].values, s[endog].values, s[inst].values
        b, se = tsls(y, x, Z, W, codes, G)
        b, se = float(b[0]), float(se[0])
        day = s.drop_duplicates('date')
        dfe = pd.get_dummies(day[['month', 'year', 'dow']].astype(str),
                             drop_first=True).astype(float)
        Wd = np.column_stack([day[ctrl].values, dfe.values])
        F = eff_f(day[endog].values, day[inst].values, Wd)
        bp = 1e4 * day[endog].std()
        ar, pmax = ar_sets(y, x, Z, W, codes, G, b, se)
        f = lambda r: r[0] if r[1] is None else '[%+.1f, %+.1f]' % (r[1] * bp, r[2] * bp)
        eq = ar[0.10][1] is not None and abs(ar[0.10][1] * bp) < SESOI and abs(ar[0.10][2] * bp) < SESOI
        print('%-36s %6d %5d F %6.1f  %+7.2f bp  t %+5.2f  95%% %s  90%% %s  AR max p %.3f  %s'
              % (label, len(s), G, F, b * bp, b / se, f(ar[0.05]), f(ar[0.10]), pmax,
                 'EQUIV' if eq else 'not equiv'))
        rows.append(dict(check=label, firm_days=len(s), days=G, F_eff=F, bp_per_sd=b * bp, t=b / se,
                         ci95=f(ar[0.05]), ci90=f(ar[0.10]), ar_max_p=pmax, equivalent=eq))
    except Exception as e:
        print('%-36s NOT RUN  %s: %s' % (label, type(e).__name__, e))
        rows.append(dict(check=label, note='NOT RUN: %s' % e))

def window(s, a=START, b=END):
    return s[(s['date'] >= a) & (s['date'] <= b)]

OUTF = f'{DATA}/zenodo_shanghai_fullyear.csv'
RAW = '/content/zen_raw'
os.makedirs(RAW, exist_ok=True)
done = pd.read_csv(OUTF, parse_dates=['date']) if os.path.exists(OUTF) else pd.DataFrame(columns=['date', 'pm_zen'])
have = set(done['date'].dt.strftime('%Y%m')) if len(done) else set()
months = [m.strftime('%Y%m') for m in pd.period_range('2014-05', '2019-12', freq='M')]
parts, missing = [done], []
for m in months:
    if m in have:
        continue
    p = f'{RAW}/aqi{m}.nc'
    try:
        with requests.get(f'https://zenodo.org/records/6404603/files/aqi{m}.nc?download=1',
                          stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(p, 'wb') as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        ds = xr.open_dataset(p)
        dec = lambda a: (a.decode('utf-8', 'ignore') if isinstance(a, bytes) else str(a)).strip()
        area = np.array([dec(a) for a in ds.area.values])
        keep = np.where(area == 'Shanghai')[0]
        t = pd.to_datetime([dec(s) for s in ds.time_point.values],
                           format='%Y-%m-%dT%H:%M:%SZ', utc=True)
        pm = ds.pm2_5.values[:, keep].astype('float64')   # (time, station), as in step 10
        ds.close()
        pm[(pm > 1e6) | (pm < 0)] = np.nan
        X = pd.DataFrame(pm, index=t)
        X = X[~X.index.duplicated(keep='first')]
        X.index = X.index.tz_convert('Asia/Shanghai')
        X = X[(X.index.year == int(m[:4])) & (X.index.month == int(m[4:]))]
        h = X.mean(axis=1)
        g = h.groupby(h.index.normalize())
        dd = g.mean().where(g.count() >= 8).rename('pm_zen').reset_index()
        dd.columns = ['date', 'pm_zen']
        dd['date'] = dd['date'].dt.tz_localize(None)
        parts.append(dd)
        print('   %s: %d stations, %d days' % (m, len(keep), dd['pm_zen'].notna().sum()))
        pd.concat(parts).drop_duplicates('date').sort_values('date').to_csv(OUTF, index=False)
    except Exception as e:
        missing.append(m)
        print('   %s: FAILED (%s: %s)' % (m, type(e).__name__, str(e)[:80]))
    finally:
        if os.path.exists(p):
            os.remove(p)
Z = pd.read_csv(OUTF, parse_dates=['date']).dropna()
Z = Z[Z['pm_zen'] > 0]
Z['log_pm25_z'] = np.log(Z['pm_zen'])
print('\nZenodo Shanghai: %d days, %s to %s; months failed: %s'
      % (len(Z), Z['date'].min().date(), Z['date'].max().date(), missing or 'none'))

C = pd.read_csv(f'{DATA}/ah_premium_clean.csv', parse_dates=['date'], dtype={'h_code': str})
C['h_code'] = C['h_code'].str.zfill(5)
fs = pd.read_csv(f'{DATA}/shanghai_first_stage_fullyear.csv', parse_dates=['date']).drop(
    columns=['month', 'year', 'dow'])
base = window(C.merge(fs, on='date', how='inner'))
dd = base.drop_duplicates('date').merge(Z, on='date')
print('correlation with the registered series in logs, trading days: %.3f (n %d)'
      % (dd['log_pm25'].corr(dd['log_pm25_z']), len(dd)))
run('Primary (reproduction)', base)
run('Zenodo station average, full year', base.merge(Z[['date', 'log_pm25_z']], on='date'),
    endog='log_pm25_z')
run('Registered series, same days', base[base['date'].isin(set(Z['date']))])
pd.DataFrame(rows).to_csv(f'{DATA}/step51_zenodo.csv', index=False)
print('\nwritten: step51_zenodo.csv\nDone. Paste everything above back.')
