# ============================================================
# Step 50: four follow-ups to steps 47-48 (read-only; ~10-15 min)
#   A. step47 table in full: 90% sets and AR curve maxima, so the two
#      rows marked "not equiv" despite tight 95% sets are explained
#   B. Romano-Wolf over the SIX-test family exactly as the filed
#      registration lists it (future, Beijing, Urumqi, FX, t+1, t+2),
#      placebo cities entered alongside Shanghai (H4)
#   C. Announcement-date coverage for H7 (registered: >= 50 of 61 firms)
# Writes step50_romano_wolf_six.csv.
# ============================================================
import os, numpy as np, pandas as pd
from math import erf, sqrt
from scipy import stats
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass
DATA = '/content/drive/MyDrive/bilateral/data'
START, END = '2014-05-13', '2019-12-31'
LAGS, B, SEED, SESOI = 7, 999, 20261105, 25.0
INST = ['log_blh', 'u', 'v', 'surface_pressure']
CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']
pd.set_option('display.width', 200)

# ---------------- A ----------------
R = pd.read_csv(f'{DATA}/step47_robustness.csv')
print('A. STEP 47 TABLE')
print(R[['check', 'firm_days', 'days', 'F_eff', 'bp_per_sd', 't', 'ci95_lo', 'ci95_hi',
         'ci90_lo', 'ci90_hi', 'interval', 'equivalent']].round(2).to_string(index=False))

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



def window(s, a=START, b=END):
    return s[(s['date'] >= a) & (s['date'] <= b)]

C = pd.read_csv(f'{DATA}/ah_premium_clean.csv', parse_dates=['date'], dtype={'h_code': str})
C['h_code'] = C['h_code'].str.zfill(5)
fs = pd.read_csv(f'{DATA}/shanghai_first_stage_fullyear.csv', parse_dates=['date']).drop(
    columns=['month', 'year', 'dow'])
base = window(C.merge(fs, on='date', how='inner'))
P = pd.read_csv(f'{DATA}/ah_premium_panel.csv', parse_dates=['date'], dtype={'h_code': str})
P['h_code'] = P['h_code'].str.zfill(5)
P = P.sort_values(['h_code', 'date'])
P['d_logA'] = np.log(P['a_raw']).groupby(P['h_code']).diff()
P['d_logH'] = np.log(P['h_raw']).groupby(P['h_code']).diff()
P['gap'] = P.groupby('h_code')['date'].diff().dt.days
P['clean'] = (~((P['d_logA'].abs() > 0.11) | (P['d_logH'].abs() > 0.35) | (P['gap'] > 5)
                | P['gap'].isna()) & P['d_prem_raw'].notna()).astype(bool)
gq = P.groupby('h_code')
P['d2_prem'] = P['prem_raw'] - gq['prem_raw'].shift(2)
ok = P['clean'] & gq['clean'].shift(1).eq(True)
Q2 = window(P[ok].dropna(subset=['d2_prem']).merge(fs, on='date', how='inner'))
cases = [('Primary', base, 'd_prem_raw'),
         ('Excluding June-September 2015', base[~base['date'].between('2015-06-01', '2015-09-30')], 'd_prem_raw'),
         ('Two-day premium change', Q2, 'd2_prem'),
         ('Winter only (Dec-Feb)', window(C.merge(fs, on='date')[lambda x: x['date'].dt.month.isin([12, 1, 2])],
                                          '2013-12-01', '2020-02-29'), 'd_prem_raw')]
print('\n   Anderson-Rubin curve maximum (a conservative overidentification p-value)')
for lab, s, out in cases:
    s, W, codes, G = build(s, ['log_pm25'], INST, CONTROLS, out)
    y, x, Z = s[out].values, s['log_pm25'].values, s[INST].values
    b, se = tsls(y, x, Z, W, codes, G)
    res, pmax = ar_sets(y, x, Z, W, codes, G, float(b[0]), float(se[0]))
    bp = 1e4 * s.drop_duplicates('date')['log_pm25'].std()
    f = lambda r: '%s' % r[0] if r[1] is None else '[%+.1f, %+.1f] (%s)' % (r[1] * bp, r[2] * bp, r[0])
    print('   %-32s AR max p %.3f   95%% %s   90%% %s' % (lab, pmax, f(res[0.05]), f(res[0.10])))


# ---------------- B ----------------
class Test(object):
    def __init__(self, label, y, X, Z, W, codes, j=0, scale=1.0):
        X = X.reshape(len(y), -1)
        Wc = np.column_stack([np.ones(len(y)), W])
        M = np.column_stack([y, X, Z])
        Rr = M - Wc @ np.linalg.lstsq(Wc, M, rcond=None)[0]
        m = X.shape[1]
        self.label, self.j, self.scale = label, j, scale
        self.y, self.X, self.Z = Rr[:, 0], Rr[:, 1:1 + m], Rr[:, 1 + m:]
        self.codes, self.G, self.n = codes, int(codes.max()) + 1, len(y)
        self.df = self.n - Wc.shape[1] - m
        self.Zpinv = np.linalg.inv(self.Z.T @ self.Z) @ self.Z.T

    def fit(self, y=None, X=None):
        y = self.y if y is None else y
        X = self.X if X is None else X
        Xh = self.Z @ (self.Zpinv @ X)
        A = Xh.T @ X
        b = np.linalg.solve(A, Xh.T @ y)
        e = y - X @ b
        S = np.zeros((self.G, X.shape[1]))
        np.add.at(S, self.codes, Xh * e[:, None])
        Ai = np.linalg.inv(A)
        V = Ai @ dk(S, LAGS) @ Ai.T * (self.G / (self.G - 1.0)) * ((self.n - 1.0) / self.df)
        return float(b[self.j]), float(np.sqrt(max(V[self.j, self.j], 1e-300)))

    def prepare_null(self):
        o = [i for i in range(self.X.shape[1]) if i != self.j]
        if o:
            Xo = self.X[:, o]
            Xh = self.Z @ (self.Zpinv @ Xo)
            self.fit0 = Xo @ np.linalg.solve(Xh.T @ Xo, Xh.T @ self.y)
        else:
            self.fit0 = np.zeros(self.n)
        self.e0 = self.y - self.fit0
        self.Xhat = self.Z @ (self.Zpinv @ self.X)
        self.V = self.X - self.Xhat

    def resample(self, w_by_date):
        w = w_by_date[self.codes]
        return self.fit(y=self.fit0 + w * self.e0, X=self.Xhat + w[:, None] * self.V)


prem = C[['date', 'h_code', 'd_prem_raw', 'hkd_cny']]
three = pd.read_csv(f'{DATA}/three_city_panel.csv', parse_dates=['date'])
d = window(prem.merge(fs, on='date', how='inner')).sort_values(['h_code', 'date'])
BP = 1e4 * d.groupby('date')['log_pm25'].first().std()
all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(d['date'].unique())))
date_ix = {t: i for i, t in enumerate(all_dates)}
NG, TRADING = len(all_dates), list(all_dates)


def mk(frame, endog, inst, ctrl, outcome, label, j=0, scale=BP):
    s = cal(frame).dropna(subset=endog + inst + ctrl + [outcome])
    fe = pd.get_dummies(s[['month', 'year', 'dow']].astype(str), drop_first=True).astype(float)
    W = np.column_stack([s[ctrl].values, fe.values])
    codes = s['date'].map(date_ix).values.astype(int)
    o = np.argsort(codes, kind='stable')
    return Test(label, s[outcome].values[o], s[endog].values[o], s[inst].values[o], W[o], codes[o], j, scale)


def lead_frame(k, trading):
    lead = d[['date', 'h_code', 'd_prem_raw']].copy()
    if trading:
        sh = {t: TRADING[i - k] for i, t in enumerate(TRADING) if i >= k}
        lead['date'] = lead['date'].map(sh)
        lead = lead.dropna(subset=['date'])
    else:
        lead['date'] = lead['date'] - pd.Timedelta(days=k)
    lead = lead.rename(columns={'d_prem_raw': 'lead_prem'})
    return d.drop(columns=['d_prem_raw']).merge(lead, on=['date', 'h_code'], how='inner')


def future_frame(trading):
    nxt = fs[['date', 'log_pm25'] + INST].copy()
    if trading:
        sh = {t: TRADING[i - 1] for i, t in enumerate(TRADING) if i >= 1}
        nxt['date'] = nxt['date'].map(sh)
        nxt = nxt.dropna(subset=['date'])
    else:
        nxt['date'] = nxt['date'] - pd.Timedelta(days=1)
    nxt = nxt.rename(columns={c: 'f_' + c for c in ['log_pm25'] + INST})
    return d[['date', 'h_code', 'd_prem_raw'] + CONTROLS].merge(nxt, on='date', how='inner')


fxd = d.drop_duplicates('date')[['date', 'hkd_cny']].sort_values('date')
fxd['d_fx'] = np.log(fxd['hkd_cny']).diff()
fxd = fxd.merge(fs, on='date', how='inner').dropna(subset=['d_fx'])
J = d[['date', 'h_code', 'd_prem_raw', 'log_pm25'] + INST + CONTROLS].merge(three, on='date', how='inner')


def joint(tag):
    sd = J.drop_duplicates('date')['%s_log_pm25' % tag].std()
    return mk(J, ['log_pm25', '%s_log_pm25' % tag], INST + ['%s_%s' % (tag, i) for i in INST],
              CONTROLS + ['%s_%s' % (tag, c) for c in CONTROLS], 'd_prem_raw',
              '%s pollution | Shanghai' % tag.capitalize(), j=1, scale=1e4 * sd)


F_INST = ['f_' + c for c in INST]
common = [mk(fxd, ['log_pm25'], INST, CONTROLS, 'd_fx', 'FX placebo (HKD/CNY)'),
          joint('urumqi'), joint('beijing')]


def family(trading):
    tag = ', trading day' if trading else ''
    return common + [
        mk(future_frame(trading), ['f_log_pm25'], F_INST, CONTROLS, 'd_prem_raw', 'future pollution (t+1)' + tag),
        mk(lead_frame(1, trading), ['log_pm25'], INST, CONTROLS, 'lead_prem', 'premium at t+1' + tag),
        mk(lead_frame(2, trading), ['log_pm25'], INST, CONTROLS, 'lead_prem', 'premium at t+2' + tag)]


out = []


def romano_wolf(tests, title):
    for t in tests:
        t.prepare_null()
    fits = [t.fit() for t in tests]
    obs = np.array([b_ / s_ for b_, s_ in fits])
    rng = np.random.default_rng(SEED)
    boot = np.empty((B, len(tests)))
    for r in range(B):
        w = rng.choice(np.array([-1.0, 1.0]), size=NG)
        for j, t in enumerate(tests):
            bb, ss = t.resample(w)
            boot[r, j] = bb / ss
    a = np.abs(obs)
    order = np.argsort(-a)
    p = np.empty(len(tests))
    run_ = 0.0
    for step, j in enumerate(order):
        run_ = max(run_, (1 + np.sum(np.abs(boot[:, order[step:]]).max(axis=1) >= a[j])) / (B + 1.0))
        p[j] = run_
    print('\n' + '=' * 80 + '\n' + title + '\n' + '=' * 80)
    print('   %-40s %10s %7s %9s %8s %8s' % ('test', 'bp per sd', 't', 'p unadj', 'p RW', 'N'))
    for j in order:
        t = tests[j]
        praw = 2 * (1 - 0.5 * (1 + erf(a[j] / sqrt(2))))
        print('   %-40s %10.2f %7.2f %9.3f %8.3f %8s' % (t.label, fits[j][0] * t.scale, obs[j], praw, p[j],
                                                         format(t.n, ',')))
        out.append(dict(family=title, test=t.label, bp_per_sd=fits[j][0] * t.scale, t=obs[j],
                        p_unadj=praw, p_rw=p[j], n=t.n))


romano_wolf(family(False), 'B. REGISTERED SIX-TEST FAMILY, calendar-day leads')
romano_wolf(family(True), 'B. SIX-TEST FAMILY, trading-day leads (disclosed deviation)')
pd.DataFrame(out).to_csv(f'{DATA}/step50_romano_wolf_six.csv', index=False)

# ---------------- C ----------------
print('\nC. ANNOUNCEMENT DATES (H7)')
A = pd.read_csv(f'{DATA}/announcement_dates.csv', dtype=str)
print('   columns:', list(A.columns))
code = next((c for c in A.columns if 'code' in c.lower()), A.columns[0])
dcol = next((c for c in A.columns if 'date' in c.lower()), A.columns[1])
A[dcol] = pd.to_datetime(A[dcol], errors='coerce')
Aw = A[(A[dcol] >= START) & (A[dcol] <= END)]
print('   events in window: %d; firms with any date: %d of %d; firms with >=10 events: %d'
      % (len(Aw), Aw[code].nunique(), base['a_code'].nunique(),
         int((Aw.groupby(code).size() >= 10).sum())))
print('\nDone. Paste everything above back.')
