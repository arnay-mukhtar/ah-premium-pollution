# ============================================================
# Step 48: Romano-Wolf over the falsification family AS REGISTERED
#
# The registration's family is: the two-leg decomposition, future
# pollution, the three-city test (each placebo city entered ALONGSIDE
# Shanghai pollution, on its own instruments) and persistence. Step 28
# ran the placebo cities on their own, without Shanghai, and left the
# A and H legs out. This reruns the family as registered:
#   A leg, H leg, FX, future pollution, Urumqi | Shanghai,
#   Beijing | Shanghai, premium at t+1, premium at t+2
# once with the registered calendar-day leads and once with trading-day
# leads (the disclosed deviation in 8.2).
#
# Bootstrap exactly as step 28: wild restricted, Rademacher weights by
# date, one draw per replication shared by every test, 999 replications,
# same seed. For the joint placebo tests the null imposes a zero city
# coefficient while Shanghai stays in the model.
# Read-only; writes step48_romano_wolf.csv to the data folder.
# ============================================================
import os, numpy as np, pandas as pd
from math import erf, sqrt
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DATA = '/content/drive/MyDrive/bilateral/data'
START, END = '2014-05-13', '2019-12-31'
LAGS, B, SEED = 7, 999, 20261105
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


class Test(object):
    """2SLS test of coefficient j (column j of X), controls partialled out."""

    def __init__(self, label, y, X, Z, W, codes, j=0, scale=1.0):
        X = X.reshape(len(y), -1)
        Wc = np.column_stack([np.ones(len(y)), W])
        M = np.column_stack([y, X, Z])
        R = M - Wc @ np.linalg.lstsq(Wc, M, rcond=None)[0]
        m = X.shape[1]
        self.label, self.j, self.scale = label, j, scale
        self.y, self.X, self.Z = R[:, 0], R[:, 1:1 + m], R[:, 1 + m:]
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
            br = np.linalg.solve(Xh.T @ Xo, Xh.T @ self.y)
            self.fit0 = Xo @ br
        else:
            self.fit0 = np.zeros(self.n)
        self.e0 = self.y - self.fit0
        self.Xhat = self.Z @ (self.Zpinv @ self.X)
        self.V = self.X - self.Xhat

    def resample(self, w_by_date):
        w = w_by_date[self.codes]
        return self.fit(y=self.fit0 + w * self.e0, X=self.Xhat + w[:, None] * self.V)


prem = pd.read_csv(f'{DATA}/ah_premium_clean.csv', parse_dates=['date'],
                   usecols=['date', 'h_code', 'd_prem_raw', 'd_logA', 'd_logH', 'hkd_cny'])
fs = pd.read_csv(f'{DATA}/shanghai_first_stage_fullyear.csv', parse_dates=['date'])
three = pd.read_csv(f'{DATA}/three_city_panel.csv', parse_dates=['date'])
d = prem.merge(fs, on='date', how='inner')
d = d[(d['date'] >= START) & (d['date'] <= END)].sort_values(['h_code', 'date'])
BP = 1e4 * d.groupby('date')['log_pm25'].first().std()
all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(d['date'].unique())))
date_ix = {t: i for i, t in enumerate(all_dates)}
NG, TRADING = len(all_dates), list(all_dates)
print('step 48: %s firm-days, %d days, %d replications' % (format(len(d), ','), NG, B))


def build(frame, endog, inst, ctrl, outcome, label, j=0, scale=BP):
    s = frame.copy()
    s['month'], s['year'], s['dow'] = s['date'].dt.month, s['date'].dt.year, s['date'].dt.dayofweek
    s = s.dropna(subset=endog + inst + ctrl + [outcome])
    fe = pd.get_dummies(s[['month', 'year', 'dow']].astype(str), drop_first=True).astype(float)
    W = np.column_stack([s[ctrl].values, fe.values])
    codes = s['date'].map(date_ix)
    if codes.isna().any():
        raise SystemExit('%s: dates outside the calendar' % label)
    codes = codes.values.astype(int)
    o = np.argsort(codes, kind='stable')
    return Test(label, s[outcome].values[o], s[endog].values[o], s[inst].values[o],
                W[o], codes[o], j=j, scale=scale)


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
    endog = ['log_pm25', '%s_log_pm25' % tag]
    inst = INST + ['%s_%s' % (tag, i) for i in INST]
    ctrl = CONTROLS + ['%s_%s' % (tag, c) for c in CONTROLS]
    sd = J.drop_duplicates('date')['%s_log_pm25' % tag].std()
    return build(J, endog, inst, ctrl, 'd_prem_raw',
                 '%s pollution | Shanghai' % tag.capitalize(), j=1, scale=1e4 * sd)


F_INST = ['f_' + c for c in INST]
common = [
    build(d, ['log_pm25'], INST, CONTROLS, 'd_logA', 'A leg (Shanghai)'),
    build(d, ['log_pm25'], INST, CONTROLS, 'd_logH', 'H leg (Hong Kong)'),
    build(fxd, ['log_pm25'], INST, CONTROLS, 'd_fx', 'FX placebo (HKD/CNY)'),
    joint('urumqi'),
    joint('beijing'),
]


def family(trading):
    tag = ', trading day' if trading else ''
    return common + [
        build(future_frame(trading), ['f_log_pm25'], F_INST, CONTROLS, 'd_prem_raw',
              'future pollution (t+1)' + tag),
        build(lead_frame(1, trading), ['log_pm25'], INST, CONTROLS, 'lead_prem',
              'premium at t+1' + tag),
        build(lead_frame(2, trading), ['log_pm25'], INST, CONTROLS, 'lead_prem',
              'premium at t+2' + tag),
    ]


def ncdf(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


out = []


def romano_wolf(tests, title):
    for t in tests:
        t.prepare_null()
    fits = [t.fit() for t in tests]
    obs = np.array([b / s for b, s in fits])
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
        mx = np.abs(boot[:, order[step:]]).max(axis=1)
        run_ = max(run_, (1 + np.sum(mx >= a[j])) / (B + 1.0))
        p[j] = run_
    print('\n' + '=' * 80 + '\n' + title + '\n' + '=' * 80)
    print('   %-40s %10s %7s %9s %8s %8s' % ('test', 'bp per sd', 't', 'p unadj', 'p RW', 'N'))
    for j in order:
        t = tests[j]
        praw = 2 * (1 - ncdf(a[j]))
        print('   %-40s %10.2f %7.2f %9.3f %8.3f %8s'
              % (t.label, fits[j][0] * t.scale, obs[j], praw, p[j], format(t.n, ',')))
        out.append(dict(family=title, test=t.label, bp_per_sd=fits[j][0] * t.scale,
                        t=obs[j], p_unadj=praw, p_rw=p[j], n=t.n))


romano_wolf(family(False), 'REGISTERED FAMILY, calendar-day leads (as in the registered code)')
romano_wolf(family(True), 'SAME FAMILY, trading-day leads (disclosed deviation)')
pd.DataFrame(out).to_csv(f'{DATA}/step48_romano_wolf.csv', index=False)
print('\nwritten: step48_romano_wolf.csv\nDone. Paste everything above back.')
