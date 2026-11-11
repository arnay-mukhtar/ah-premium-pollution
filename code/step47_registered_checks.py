# ============================================================
# Step 47: the registered checks the paper does not yet report
#
# Read-only on the data. Writes two result files to the data folder:
#   step47_robustness.csv   the registered robustness table
#   step47_placebos.csv     the three-city test as registered (joint)
#
#   A. Robustness table (registration, "Robustness and deviations"):
#      Zenodo station average, 07:00-16:00 window, winter only,
#      including 2020, sounding instruments added, surface pressure
#      dropped, winsorised 0.5% and 1%, back-adjusted prices,
#      excluding the 2015 bubble, two-day change, dropping post-closure
#      days, Hong Kong reduced form. Cap-weighted and CNH need data
#      this folder does not hold; they are listed as NOT RUN here.
#   B. Placebo cities entered ALONGSIDE Shanghai pollution, each with
#      its own instruments, as the registration specifies.
#
# Self-contained: the Anderson-Rubin code is inside this cell.
# ============================================================
import os, glob, numpy as np, pandas as pd
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

print('files in the data folder:')
for p in sorted(glob.glob(DATA + '/*.csv')):
    print('   %-45s %7.1f MB' % (os.path.basename(p), os.path.getsize(p) / 1e6))


# ---------------- estimators ----------------
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


# ---------------- data ----------------
C = pd.read_csv(f'{DATA}/ah_premium_clean.csv', parse_dates=['date'],
                dtype={'h_code': str, 'a_code': str})
C['h_code'] = C['h_code'].str.zfill(5)
fs = pd.read_csv(f'{DATA}/shanghai_first_stage_fullyear.csv',
                 parse_dates=['date']).drop(columns=['month', 'year', 'dow'])
base = window(C.merge(fs, on='date', how='inner'))
print('\nprimary sample: %d firm-days, %d firms, %d days'
      % (len(base), base['h_code'].nunique(), base['date'].nunique()))

print('\n' + '=' * 100)
print('A. REGISTERED ROBUSTNESS TABLE')
print('=' * 100)
run('Primary (reproduction)', base,
    note='must equal +0.73 bp, AR95 [-15.5, +18.1]')

# 1. Zenodo station-average series
try:
    zz = None
    for fn in ('cn_pm25_daily_24h.csv',):
        if os.path.exists(f'{DATA}/{fn}'):
            zz = pd.read_csv(f'{DATA}/{fn}', parse_dates=['date'])[['date', 'Shanghai']]
    if zz is None:
        zz = pd.read_csv(f'{DIR}/cn_pm25_daily.csv', parse_dates=['date'])[['date', 'Shanghai']]
    zz = zz[zz['Shanghai'] > 0]
    zz['log_pm25_z'] = np.log(zz['Shanghai'])
    run('Zenodo station-average pollution', base.merge(zz[['date', 'log_pm25_z']], on='date'),
        endog='log_pm25_z')
except Exception as e:
    rows.append(dict(check='Zenodo station-average pollution', note='NOT RUN: %s' % e))

# 2. 07:00-16:00 exposure window (hours 7-15 local, at least 7 hours)
try:
    H = pd.read_csv(f'{DATA}/quotsoft_cities_hourly.csv', usecols=['ts', '上海'],
                    parse_dates=['ts'])
    H = H[(H['ts'].dt.hour >= 7) & (H['ts'].dt.hour < 16) & (H['上海'] > 0)]
    H['date'] = H['ts'].dt.normalize()
    g = H.groupby('date')['上海'].agg(['mean', 'size'])
    g = g[g['size'] >= 7].reset_index()
    g['log_pm25_w'] = np.log(g['mean'])
    run('07:00-16:00 exposure window', base.merge(g[['date', 'log_pm25_w']], on='date'),
        endog='log_pm25_w')
except Exception as e:
    rows.append(dict(check='07:00-16:00 exposure window', note='NOT RUN: %s' % e))

# 3. Winter only: December-February, the original design (2013-12 to 2020-02)
allyrs = C.merge(fs, on='date', how='inner')
wint = allyrs[allyrs['date'].dt.month.isin([12, 1, 2])]
run('Winter only (Dec-Feb)', window(wint, '2013-12-01', '2020-02-29'),
    note='registration: 390 days')

# 4. Including 2020
run('Including 2020', window(allyrs, START, '2020-12-31'), note='registration: 1,560 days')

# 5. Sounding instruments added
run('Sounding instruments added', base, inst=INST + ['dT_200_100', 'dT_1000_300'])

# 6. Surface pressure dropped
run('Surface pressure dropped', base, inst=['log_blh', 'u', 'v'])

# 7-8. Winsorised instead of filtered (unfiltered panel, pooled quantiles)
try:
    P = pd.read_csv(f'{DATA}/ah_premium_panel.csv', parse_dates=['date'],
                    dtype={'h_code': str})
    P['h_code'] = P['h_code'].str.zfill(5)
except Exception:
    P = C.copy()
    print('   (ah_premium_panel.csv not found; winsorising the filtered panel instead)')
Pw = window(P.dropna(subset=['d_prem_raw']).merge(fs, on='date', how='inner'))
for q in (0.005, 0.01):
    lo, hi = Pw['d_prem_raw'].quantile([q, 1 - q])
    t = Pw.assign(d_prem_w=Pw['d_prem_raw'].clip(lo, hi))
    run('Winsorised at %.1f%%' % (100 * q), t, outcome='d_prem_w',
        note='unfiltered panel, bounds %+.4f %+.4f' % (lo, hi))

# 9. Back-adjusted (hfq) prices
run('Back-adjusted (hfq) prices', base, outcome='d_prem_hfq')

# 10. Excluding the 2015 bubble
run('Excluding June-September 2015',
    base[~base['date'].between('2015-06-01', '2015-09-30')])

# 11. Two-day premium change (both component days pass the filters)
Q = P.sort_values(['h_code', 'date']).copy()
if 'clean' not in Q.columns:
    Q['d_logA'] = np.log(Q['a_raw']).groupby(Q['h_code']).diff()
    Q['d_logH'] = np.log(Q['h_raw']).groupby(Q['h_code']).diff()
    Q['gap'] = Q.groupby('h_code')['date'].diff().dt.days
    Q['clean'] = ~((Q['d_logA'].abs() > 0.11) | (Q['d_logH'].abs() > 0.35)
                   | (Q['gap'] > 5) | Q['gap'].isna()) & Q['d_prem_raw'].notna()
gq = Q.groupby('h_code')
Q['d2_prem'] = Q['prem_raw'] - gq['prem_raw'].shift(2)
ok = Q['clean'].astype(bool) & gq['clean'].shift(1).fillna(False).astype(bool)
Q2 = window(Q[ok].dropna(subset=['d2_prem']).merge(fs, on='date', how='inner'))
run('Two-day premium change', Q2, outcome='d2_prem',
    note='overlapping two-day changes; DK(7) covers the overlap')

# 12. Dropping days after a one-sided closure
try:
    A = pd.read_csv(f'{DATA}/a_prices_raw.csv', usecols=['date'], parse_dates=['date'])
    Hh = pd.read_csv(f'{DIR}/h_prices.csv', usecols=['date'], parse_dates=['date'])
    a_days, h_days = set(A['date']), set(Hh['date'])
    one_sided = np.array(sorted(a_days ^ h_days), dtype='datetime64[ns]')
    pdays = np.array(sorted(set(C['date'])), dtype='datetime64[ns]')
    prev = np.r_[pdays[:1], pdays[:-1]]
    i0 = np.searchsorted(one_sided, prev, side='right')
    i1 = np.searchsorted(one_sided, pdays, side='left')
    stale = set(pd.to_datetime(pdays[i1 > i0]))
    t = base[~base['date'].isin(stale)]
    run('Dropping post-closure days', t,
        note='%d panel days follow a one-sided closure' % len(stale & set(base['date'])))
except Exception as e:
    rows.append(dict(check='Dropping post-closure days', note='NOT RUN: %s' % e))

# 13. Hong Kong reduced form (OLS on raw HK PM2.5, full weather controls)
cand = [p for p in glob.glob(DATA + '/*.csv') + glob.glob(DIR + '/*.csv')
        if any(k in os.path.basename(p).lower() for k in ('hk', 'hong', 'kowloon'))]
print('\n   Hong Kong candidate files:')
for p in cand:
    try:
        print('     %s  %s' % (p, list(pd.read_csv(p, nrows=2).columns)[:25]))
    except Exception:
        pass
rows.append(dict(check='Hong Kong reduced form',
                 note='NOT RUN in this pass: needs the HK PM2.5 and HK weather files listed above'))

# 14-15. Data not in this folder
rows.append(dict(check='Cap-weighted premium', note='NOT RUN: no market-cap series in the data folder'))
rows.append(dict(check='Offshore CNH conversion', note='NOT RUN: no CNH series in the data folder'))

R = pd.DataFrame(rows)
R.to_csv(f'{DATA}/step47_robustness.csv', index=False)
print('\nwritten: step47_robustness.csv')

# ============================================================
# B. THE THREE-CITY TEST AS REGISTERED: each city ALONGSIDE Shanghai
# ============================================================
print('\n' + '=' * 100)
print('B. PLACEBO CITIES ENTERED ALONGSIDE SHANGHAI (registered form)')
print('=' * 100)
T = pd.read_csv(f'{DATA}/three_city_panel.csv', parse_dates=['date'])
J = base[['date', 'h_code', 'd_prem_raw', 'log_pm25'] + INST + CONTROLS].merge(T, on='date', how='inner')
prows = []
for cities in (['urumqi'], ['beijing'], ['beijing', 'urumqi']):
    endog = ['log_pm25'] + ['%s_log_pm25' % c for c in cities]
    inst = INST + ['%s_%s' % (c, i) for c in cities for i in INST]
    ctrl = CONTROLS + ['%s_%s' % (c, k) for c in cities for k in CONTROLS]
    s, W, codes, G = build(J, endog, inst, ctrl, 'd_prem_raw')
    b, se = tsls(s['d_prem_raw'].values, s[endog].values, s[inst].values, W, codes, G)
    day = s.drop_duplicates('date')
    dfe = pd.get_dummies(day[['month', 'year', 'dow']].astype(str), drop_first=True).astype(float)
    Wd = np.column_stack([day[ctrl].values, dfe.values])
    swf = sw_f(day[endog].values, day[inst].values, Wd)
    tag = 'Shanghai + ' + ' + '.join(c.capitalize() for c in cities)
    print('\n%s   (%d firm-days, %d days)' % (tag, len(s), G))
    for j, e_ in enumerate(endog):
        sd = day[e_].std()
        print('   %-22s beta %+.6f  se %.6f  t %+5.2f  %+7.2f bp per own sd  SW F %7.1f'
              % (e_, b[j], se[j], b[j] / se[j], 1e4 * b[j] * sd, swf[j]))
        prows.append(dict(spec=tag, variable=e_, beta=b[j], se=se[j], t=b[j] / se[j],
                          bp_per_own_sd=1e4 * b[j] * sd,
                          bp_per_shanghai_sd=1e4 * b[j] * day['log_pm25'].std(),
                          sw_f=swf[j], firm_days=len(s), days=G))
    print('   correlations of the instrumented series (days):')
    print(day[endog].corr().round(3).to_string())
pd.DataFrame(prows).to_csv(f'{DATA}/step47_placebos.csv', index=False)
print('\nwritten: step47_placebos.csv')
print('\nDone. Paste everything above back.')
