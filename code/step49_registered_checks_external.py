# ============================================================
# Step 49: the registered checks that need data from outside the folder
#   1. Hong Kong reduced form (EPD PM2.5 on the premium, HK weather)
#   2. Secondary placebo: the three cities least correlated with
#      Shanghai PM2.5 (>=90% coverage), each on its own instruments,
#      entered alongside Shanghai; and all three together
#   3. Offshore CNH conversion (Yahoo Finance via yfinance)
#   4. Cap-weighted premium (total shares from akshare, else Yahoo)
# Downloads go to the data folder; nothing existing is overwritten.
# Writes step49_results.csv.
# ============================================================
import os, glob, time, json, numpy as np, pandas as pd, requests
from scipy import stats
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DATA = '/content/drive/MyDrive/bilateral/data'
DIR = '/content/drive/MyDrive/pollution-ah'
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
    """2SLS with Driscoll-Kraay covariance; X may hold several
    endogenous columns. With Z = X it is OLS. w = optional weights."""
    n = len(y)
    X = X.reshape(n, -1)
    Zf = np.column_stack([np.ones(n), Z, W])
    Xf = np.column_stack([X, np.ones(n), W])
    if w is not None:
        r = np.sqrt(w / w.mean())
        Zf, Xf, y = Zf * r[:, None], Xf * r[:, None], y * r
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
    """95% and 90% AR sets from one curve, widening until not binding."""
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
    """Montiel Olea-Pflueger effective F, day-level first stage."""
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


def sw_f(Xd, Z, W):
    """Sanderson-Windmeijer conditional F for each endogenous variable
    (conventional), day-level."""
    Wc = np.column_stack([np.ones(len(Xd)), W])
    M = lambda A: A - Wc @ np.linalg.lstsq(Wc, A, rcond=None)[0]
    Xt, Zt = M(Xd), M(Z)
    n, kz = Zt.shape
    m = Xt.shape[1]
    Ph = Zt @ np.linalg.lstsq(Zt, Xt, rcond=None)[0]
    out = []
    for j in range(m):
        o = [i for i in range(m) if i != j]
        r = Xt[:, j]
        if o:
            d = np.linalg.lstsq(Ph[:, o], Xt[:, j], rcond=None)[0]
            r = Xt[:, j] - Xt[:, o] @ d
        g = np.linalg.lstsq(Zt, r, rcond=None)[0]
        e = r - Zt @ g
        df1, df2 = kz - (m - 1), n - kz - Wc.shape[1]
        out.append(((r @ r - e @ e) / df1) / ((e @ e) / df2))
    return out


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
        w = s[weight].values if weight else None
        b, se = tsls(y, x, Z, W, codes, G, w=w)
        b, se = float(b[0]), float(se[0])
        day = s.drop_duplicates('date')
        dfe = pd.get_dummies(day[['month', 'year', 'dow']].astype(str),
                             drop_first=True).astype(float)
        Wd = np.column_stack([day[ctrl].values, dfe.values])
        ols = inst == [endog]
        F = np.nan if ols else eff_f(day[endog].values, day[inst].values, Wd)
        bp = 1e4 * day[endog].std()
        if ols or w is not None:
            ar = {a: ('wald', b - z * se, b + z * se)
                  for a, z in ((0.05, 1.96), (0.10, 1.645))}
        else:
            ar, _ = ar_sets(y, x, Z, W, codes, G, b, se)
        lo95 = ar[0.05][1] * bp if ar[0.05][1] is not None else np.nan
        hi95 = ar[0.05][2] * bp if ar[0.05][2] is not None else np.nan
        lo90 = ar[0.10][1] * bp if ar[0.10][1] is not None else np.nan
        hi90 = ar[0.10][2] * bp if ar[0.10][2] is not None else np.nan
        eq = bool(np.isfinite(lo90) and abs(lo90) < SESOI and abs(hi90) < SESOI)
        r = dict(check=label, firm_days=len(s), days=G, F_eff=F, beta=b,
                 se=se, t=b / se, bp_per_sd=b * bp, ci95_lo=lo95,
                 ci95_hi=hi95, ci90_lo=lo90, ci90_hi=hi90,
                 equivalent=eq, interval=ar[0.05][0], note=note)
        print('%-34s %7d %5d F %6.1f  %+7.2f bp  t %+5.2f  95%% [%+6.1f, %+6.1f]  %s'
              % (label, len(s), G, F, b * bp, b / se, lo95, hi95,
                 'EQUIV' if eq else 'not equiv'))
    except Exception as e:
        r = dict(check=label, note='NOT RUN: %s: %s' % (type(e).__name__, e))
        print('%-34s NOT RUN  %s' % (label, r['note']))
    rows.append(r)


def window(s, a=START, b=END):
    return s[(s['date'] >= a) & (s['date'] <= b)]


C = pd.read_csv(f'{DATA}/ah_premium_clean.csv', parse_dates=['date'],
                dtype={'h_code': str, 'a_code': str})
C['h_code'] = C['h_code'].str.zfill(5)
C['a_code'] = C['a_code'].str.zfill(6)
fs = pd.read_csv(f'{DATA}/shanghai_first_stage_fullyear.csv',
                 parse_dates=['date']).drop(columns=['month', 'year', 'dow'])
base = window(C.merge(fs, on='date', how='inner'))
print('primary sample: %d firm-days, %d days' % (len(base), base['date'].nunique()))
run('Primary (reproduction)', base)

# ---------------- 1. Hong Kong reduced form ----------------
print('\n1. HONG KONG REDUCED FORM (registered: suggestive, not causal)')
try:
    from collections import Counter

    def read_messy(path):
        for enc in ('utf-8-sig', 'utf-8', 'gb18030', 'latin-1'):
            try:
                with open(path, 'r', encoding=enc, errors='strict') as fh:
                    raw = [fh.readline() for _ in range(80)]
                break
            except (UnicodeDecodeError, LookupError):
                continue
        counts = [ln.count(',') for ln in raw]
        body = [c for c in counts[3:] if c > 0]
        mode = Counter(body).most_common(1)[0][0]
        skip = next(i for i, c in enumerate(counts) if c == mode)
        return pd.read_csv(path, skiprows=skip, encoding=enc, low_memory=False)

    parts = [read_messy(f'{DIR}/{f}') for f in ('hk_pm25_2013_2017.csv', 'hk_pm25_2018_2022.csv')
             if os.path.exists(f'{DIR}/{f}')]
    Pk = pd.concat(parts, ignore_index=True)
    st = [c for c in Pk.columns if c.upper() not in ('DATE', 'HOUR', 'POLLUTANT')]
    Pk['date'] = pd.to_datetime(Pk['DATE'], dayfirst=True, errors='coerce').dt.normalize()
    lg = Pk.melt(id_vars=['date'], value_vars=st, value_name='pm25')
    lg['pm25'] = pd.to_numeric(lg['pm25'], errors='coerce')
    hk = lg[(lg['pm25'] > 0)].dropna(subset=['date']).groupby('date', as_index=False)['pm25'].mean()
    hk['hk_log_pm25'] = np.log(hk['pm25'])
    Wh = pd.read_csv(f'{DATA}/hongkong_weather_hourly.csv')
    Wh['date'] = pd.to_datetime(Wh['time']).dt.normalize()
    g = Wh.groupby('date')
    hm = g[CONTROLS].mean().add_prefix('hk_').reset_index()
    hm['n'] = g.size().values
    hm = hm[hm['n'] >= 8].drop(columns='n')
    HKC = ['hk_' + c for c in CONTROLS]
    t = base.merge(hk[['date', 'hk_log_pm25']], on='date').merge(hm, on='date')
    print('   HK days with pollution and weather in the sample: %d' % t['date'].nunique())
    run('Hong Kong reduced form, HK weather', t, endog='hk_log_pm25',
        inst=['hk_log_pm25'], ctrl=HKC, note='OLS, raw EPD PM2.5 (13 general stations)')
    run('Hong Kong reduced form, HK + Shanghai weather', t, endog='hk_log_pm25',
        inst=['hk_log_pm25'], ctrl=HKC + CONTROLS, note='OLS')
except Exception as e:
    rows.append(dict(check='Hong Kong reduced form', note='NOT RUN: %s: %s' % (type(e).__name__, e)))
    print('   NOT RUN:', e)

# ---------------- 2. Secondary placebo: three least-correlated cities ----------------
print('\n2. SECONDARY PLACEBO: three cities least correlated with Shanghai')
prows = []
try:
    D = pd.read_csv(f'{DATA}/quotsoft_cities_daily.csv', parse_dates=['date'])
    keep = [c for c in D.columns if c not in ('date', '上海') and D[c].notna().mean() >= 0.90]
    cc = {}
    for c in keep:
        s = D[['上海', c]].dropna()
        s = s[(s > 0).all(axis=1)]
        if len(s) > 500:
            cc[c] = np.log(s['上海']).corr(np.log(s[c]))
    cc = pd.Series(cc).sort_values()
    three = list(cc.index[:3])
    print('   cities with >=90%% coverage: %d; least correlated: %s'
          % (len(cc), ', '.join('%s (%.3f)' % (c, cc[c]) for c in three)))
    geo = pd.read_csv(f'{DATA}/city_coords.csv', dtype={'city': str}).dropna(subset=['lat', 'lon'])
    geo = {r.city: (r.lat, r.lon) for r in geo.itertuples()}
    safe = lambda name: ''.join('%04x' % ord(ch) for ch in name)
    FIELDS = ['boundary_layer_height', 'wind_speed_10m', 'wind_direction_10m',
              'surface_pressure', 'temperature_2m', 'precipitation',
              'cloud_cover', 'shortwave_radiation']
    AGG = ['boundary_layer_height', 'wind_speed_10m', 'surface_pressure', 'temperature_2m',
           'precipitation', 'cloud_cover', 'shortwave_radiation', 'u', 'v']
    os.makedirs(f'{DATA}/city_meteo_placebo', exist_ok=True)
    Jp = base[['date', 'h_code', 'd_prem_raw', 'log_pm25'] + INST + CONTROLS].copy()
    tags = []
    for k, c in enumerate(three):
        tag = 'p%d' % (k + 1)
        met = None
        for folder in ('city_meteo_placebo', 'city_meteo'):
            f = f'{DATA}/{folder}/{safe(c)}.csv'
            if os.path.exists(f):
                m = pd.read_csv(f, parse_dates=['date'])
                if m['date'].min() <= pd.Timestamp(START) and m['date'].max() >= pd.Timestamp(END):
                    met = m
                    break
        if met is None:
            lat, lon = geo[c]
            hp = []
            for y0 in range(2014, 2020, 2):
                r = requests.get('https://archive-api.open-meteo.com/v1/archive',
                                 params={'latitude': lat, 'longitude': lon,
                                         'start_date': '%d-01-01' % y0,
                                         'end_date': '%d-12-31' % (y0 + 1),
                                         'hourly': ','.join(FIELDS),
                                         'timezone': 'Asia/Shanghai'}, timeout=600)
                r.raise_for_status()
                hp.append(pd.DataFrame(r.json()['hourly']))
                time.sleep(2)
            om = pd.concat(hp).drop_duplicates('time')
            om['time'] = pd.to_datetime(om['time'])
            rad = np.deg2rad(om['wind_direction_10m'])
            om['u'] = -om['wind_speed_10m'] * np.sin(rad)
            om['v'] = -om['wind_speed_10m'] * np.cos(rad)
            om['date'] = om['time'].dt.normalize()
            gg = om.groupby('date')
            met = gg[AGG].mean().reset_index()
            met['n_hours'] = gg.size().values
            met = met[met['n_hours'] >= 8]
            met.insert(1, 'city', c)
            met.insert(2, 'lat', lat)
            met.insert(3, 'lon', lon)
            met.to_csv(f'{DATA}/city_meteo_placebo/{safe(c)}.csv', index=False)
            print('   downloaded meteorology for %s (%.2f, %.2f)' % (c, lat, lon))
        met = met.rename(columns={'boundary_layer_height': 'blh'})
        met['log_blh'] = np.log(met['blh'].clip(lower=1))
        pm = D[['date', c]].rename(columns={c: 'pm25'})
        pm = pm[pm['pm25'] > 0]
        pm['log_pm25'] = np.log(pm['pm25'])
        cm = met[['date'] + INST + CONTROLS].merge(pm[['date', 'log_pm25']], on='date')
        cm = cm.rename(columns={x: '%s_%s' % (tag, x) for x in INST + CONTROLS + ['log_pm25']})
        Jp = Jp.merge(cm, on='date', how='inner')
        tags.append((tag, c))
    for group in [[t] for t in tags] + [tags]:
        endog = ['log_pm25'] + ['%s_log_pm25' % t for t, _ in group]
        inst = INST + ['%s_%s' % (t, i) for t, _ in group for i in INST]
        ctrl = CONTROLS + ['%s_%s' % (t, x) for t, _ in group for x in CONTROLS]
        s, W, codes, G = build(Jp, endog, inst, ctrl, 'd_prem_raw')
        b, se = tsls(s['d_prem_raw'].values, s[endog].values, s[inst].values, W, codes, G)
        day = s.drop_duplicates('date')
        dfe = pd.get_dummies(day[['month', 'year', 'dow']].astype(str), drop_first=True).astype(float)
        Wd = np.column_stack([day[ctrl].values, dfe.values])
        swf = sw_f(day[endog].values, day[inst].values, Wd)
        name = 'Shanghai + ' + ' + '.join(c for _, c in group)
        print('\n   %s   (%d firm-days, %d days)' % (name, len(s), G))
        lab = {'log_pm25': 'Shanghai'}
        lab.update({'%s_log_pm25' % t: c for t, c in group})
        for j, e_ in enumerate(endog):
            sd = day[e_].std()
            print('      %-12s beta %+.6f  t %+5.2f  %+7.2f bp per own sd  SW F %7.1f'
                  % (lab[e_], b[j], b[j] / se[j], 1e4 * b[j] * sd, swf[j]))
            prows.append(dict(check='secondary placebo', spec=name, variable=lab[e_], beta=b[j],
                              se=se[j], t=b[j] / se[j], bp_per_own_sd=1e4 * b[j] * sd,
                              sw_f=swf[j], firm_days=len(s), days=G))
    for t, c in tags:
        day = Jp.drop_duplicates('date')
        day = cal(day).dropna(subset=['%s_%s' % (t, x) for x in INST + CONTROLS + ['log_pm25']])
        dfe = pd.get_dummies(day[['month', 'year', 'dow']].astype(str), drop_first=True).astype(float)
        Wd = np.column_stack([day[['%s_%s' % (t, x) for x in CONTROLS]].values, dfe.values])
        print('   own first stage, %s: effective F %.1f on %d days'
              % (c, eff_f(day['%s_log_pm25' % t].values, day[['%s_%s' % (t, i) for i in INST]].values, Wd), len(day)))
except Exception as e:
    prows.append(dict(check='secondary placebo', note='NOT RUN: %s: %s' % (type(e).__name__, e)))
    print('   NOT RUN:', type(e).__name__, e)

# ---------------- 3. Offshore CNH conversion ----------------
print('\n3. OFFSHORE CNH CONVERSION')
try:
    import yfinance as yf
    px = {}
    for tk in ('CNH=X', 'HKD=X'):
        h = yf.download(tk, start='2014-01-01', end='2020-12-31', progress=False, auto_adjust=False)
        cl = h['Close']
        cl = cl.iloc[:, 0] if isinstance(cl, pd.DataFrame) else cl
        px[tk] = cl
    fx = pd.DataFrame(px).dropna()
    fx.index = pd.to_datetime(fx.index).tz_localize(None).normalize()
    fx['hkd_cnh'] = fx['CNH=X'] / fx['HKD=X']
    fx = fx[['hkd_cnh']].reset_index().rename(columns={'index': 'date', 'Date': 'date'})
    fx.to_csv(f'{DATA}/hkd_cnh_daily_yahoo.csv', index=False)
    print('   CNH days: %d, %s to %s, mean %.4f' % (len(fx), fx['date'].min().date(),
                                                    fx['date'].max().date(), fx['hkd_cnh'].mean()))
    try:
        P = pd.read_csv(f'{DATA}/ah_premium_panel.csv', parse_dates=['date'], dtype={'h_code': str})
        P['h_code'] = P['h_code'].str.zfill(5)
    except Exception:
        P = C.copy()
    dates = pd.DataFrame({'date': sorted(P['date'].unique())})
    fxa = dates.merge(fx, on='date', how='left').sort_values('date')
    fxa['hkd_cnh'] = fxa['hkd_cnh'].ffill()
    P = P.merge(fxa, on='date', how='left').sort_values(['h_code', 'date'])
    P['prem_cnh'] = np.log(P['a_raw']) - np.log(P['h_raw'] * P['hkd_cnh'])
    P['d_prem_cnh'] = P.groupby('h_code')['prem_cnh'].diff()
    t = base.merge(P[['h_code', 'date', 'd_prem_cnh']], on=['h_code', 'date'], how='inner')
    print('   correlation of daily changes, CNH vs parity premium: %.4f'
          % t['d_prem_cnh'].corr(t['d_prem_raw']))
    run('Offshore CNH conversion', t, outcome='d_prem_cnh',
        note='Yahoo CNH=X / HKD=X, London-close quotes, forward-filled')
except Exception as e:
    rows.append(dict(check='Offshore CNH conversion', note='NOT RUN: %s: %s' % (type(e).__name__, e)))
    print('   NOT RUN:', type(e).__name__, e)

# ---------------- 4. Cap-weighted premium ----------------
print('\n4. CAP-WEIGHTED PREMIUM')
try:
    shf = f'{DATA}/a_total_shares.csv'
    if os.path.exists(shf):
        sh = pd.read_csv(shf, dtype={'a_code': str})
    else:
        codes_ = sorted(C['a_code'].unique())
        got = {}
        try:
            import akshare as ak
        except ImportError:
            os.system('pip -q install akshare')
            import akshare as ak
        for a in codes_:
            try:
                info = ak.stock_individual_info_em(symbol=a)
                v = info.loc[info['item'] == '总股本', 'value'].iloc[0]
                got[a] = float(v)
            except Exception:
                pass
            time.sleep(0.3)
        if len(got) < len(codes_):
            import yfinance as yf
            for a in codes_:
                if a in got:
                    continue
                try:
                    so = yf.Ticker(a + '.SS').info.get('sharesOutstanding')
                    if so:
                        got[a] = float(so)
                except Exception:
                    pass
        sh = pd.DataFrame({'a_code': list(got), 'total_shares': list(got.values())})
        sh.to_csv(shf, index=False)
    print('   total shares found for %d of %d firms' % (len(sh), C['a_code'].nunique()))
    t = base.merge(sh, on='a_code', how='inner').sort_values(['h_code', 'date'])
    t['prev_a'] = t.groupby('h_code')['a_raw'].shift(1)
    t['cap_w'] = t['total_shares'] * t['prev_a']
    t = t.dropna(subset=['cap_w'])
    run('Cap-weighted premium', t, weight='cap_w',
        note='weight = current total shares x previous-day A price; Wald interval')
except Exception as e:
    rows.append(dict(check='Cap-weighted premium', note='NOT RUN: %s: %s' % (type(e).__name__, e)))
    print('   NOT RUN:', type(e).__name__, e)

pd.concat([pd.DataFrame(rows), pd.DataFrame(prows)], ignore_index=True).to_csv(
    f'{DATA}/step49_results.csv', index=False)
print('\nwritten: step49_results.csv\nDone. Paste everything above back.')
