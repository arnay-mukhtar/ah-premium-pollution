# Source: paper2_meteorology_build.ipynb, code cell 52 (final version of step 28 as run).
# ============================================================
# Step 28 — Romano-Wolf stepdown over the falsification family,
#           plus a trading-day lead as a disclosed deviation
#
# Step 27 printed unadjusted t-statistics for six falsification tests and
# said the multiplicity correction was computed separately. This is it.
#
# WHY A BOOTSTRAP AND NOT BONFERRONI. The six tests share days, share
# instruments and share an outcome, so they are strongly dependent.
# Bonferroni assumes the worst case and throws away power; Romano-Wolf
# resamples the whole family jointly and so controls the familywise error
# rate at the true dependence. It is also what the registration names.
#
# THE BOOTSTRAP. Wild restricted efficient (Davidson & MacKinnon 2010),
# Rademacher weights drawn by DATE, so the within-day correlation across
# ~50 firms is preserved. The null is imposed before resampling, which is
# what makes the bootstrap distribution a null distribution. ONE set of
# weights is drawn per replication and applied to every test, so the
# cross-test dependence survives into the resampled family. That is the
# whole point: drawing separately per test would silently restore
# independence and give back something close to Bonferroni.
#
# THE DEVIATION. Step 27 shifts leads by CALENDAR days:
#     lead['date'] = lead['date'] - pd.Timedelta(days=k)
# so for k=1 Friday's t+1 is Saturday and Friday drops, and for k=2 both
# Thursday and Friday drop, leaving Monday to Wednesday. The two
# persistence tests therefore run on different, weekday-skewed subsamples.
# The same applies to the future-pollution test. That is the registered
# code and it is reported as registered. This file ALSO reports a
# trading-day lead, labelled as a deviation, so the reader can see whether
# the t+1/t+2 pattern survives a full week.
#
# Nothing here changes the primary specification.
# ============================================================

import os
import numpy as np
import pandas as pd

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DATA = '/content/drive/MyDrive/bilateral/data'
START, END = '2014-05-13', '2019-12-31'
LAGS = 7
B = 999                  # bootstrap replications
SEED = 20261105

INST = ['log_blh', 'u', 'v', 'surface_pressure']
CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']


# ------------------------------------------------------------
# Partialled-out 2SLS. Frisch-Waugh: residualising y, x and Z on the
# controls and fixed effects leaves beta and its Driscoll-Kraay standard
# error unchanged, and turns each bootstrap replication into O(n*k)
# instead of a fresh least-squares solve on a 30-column design.
# ------------------------------------------------------------
def _dk(S, lags):
    om = S.T @ S
    for j in range(1, int(lags) + 1):
        if j >= len(S):
            break
        w = 1.0 - j / (lags + 1.0)
        G = S[j:].T @ S[:-j]
        om += w * (G + G.T)
    return om


class Test(object):
    """One falsification test, pre-residualised and ready to resample."""

    def __init__(self, label, y, x, Z, W, codes):
        Wc = np.column_stack([np.ones(len(y)), W])
        coef, *_ = np.linalg.lstsq(Wc, np.column_stack([y, x, Z]),
                                   rcond=None)
        R = np.column_stack([y, x, Z]) - Wc @ coef
        self.label = label
        self.y, self.x, self.Z = R[:, 0], R[:, 1], R[:, 2:]
        self.codes = codes
        self.G = int(codes.max()) + 1
        self.n = len(y)
        self.k = self.Z.shape[1]
        self.df = self.n - Wc.shape[1] - 1
        self.ZtZi = np.linalg.inv(self.Z.T @ self.Z)
        self.Zpinv = self.ZtZi @ self.Z.T

    def fit(self, y=None, x=None):
        y = self.y if y is None else y
        x = self.x if x is None else x
        pi = self.Zpinv @ x
        xh = self.Z @ pi
        denom = xh @ x
        b = (xh @ y) / denom
        e = y - b * x
        S = np.zeros((self.G, 1))
        np.add.at(S, self.codes, (xh * e)[:, None])
        var = (_dk(S, LAGS)[0, 0] / denom ** 2
               * (self.G / (self.G - 1.0)) * ((self.n - 1.0) / self.df))
        return b, float(np.sqrt(max(var, 1e-300)))

    def prepare_null(self):
        """Impose beta = 0 and store what the resampling needs."""
        # Under H0 the restricted second-stage residual is y itself, since
        # the controls are already partialled out.
        self.e0 = self.y.copy()
        pi = self.Zpinv @ self.x
        self.xhat = self.Z @ pi
        self.v = self.x - self.xhat

    def resample(self, w_by_cluster):
        w = w_by_cluster[self.codes]
        return self.fit(y=w * self.e0, x=self.xhat + w * self.v)


# ------------------------------------------------------------
# Load, exactly as step 27
# ------------------------------------------------------------
prem = pd.read_csv(os.path.join(DATA, 'ah_premium_clean.csv'),
                   parse_dates=['date'],
                   usecols=['date', 'h_code', 'd_prem_raw', 'd_logA',
                            'd_logH', 'hkd_cny'])
fs = pd.read_csv(os.path.join(DATA, 'shanghai_first_stage_fullyear.csv'),
                 parse_dates=['date'])
three = pd.read_csv(os.path.join(DATA, 'three_city_panel.csv'),
                    parse_dates=['date'])

d = prem.merge(fs, on='date', how='inner')
d = d[(d['date'] >= START) & (d['date'] <= END)].copy()
d = d.sort_values(['h_code', 'date'])
sd_pm = d.groupby('date')['log_pm25'].first().std()
BP = 10000.0 * sd_pm

# A single global date index. Every test maps its own dates into this, so
# one draw of Rademacher weights serves the whole family and the tests
# stay dependent in the bootstrap exactly as they are in the data.
# Timestamps throughout, never raw datetime64: a dict keyed by datetime64
# does not match a Timestamp under Series.map and would silently produce
# all-NaN cluster codes.
all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(d['date'].unique())))
date_ix = {t: i for i, t in enumerate(all_dates)}
NG = len(all_dates)
TRADING = list(all_dates)          # trading calendar, for the lead deviation

print('step 28: %s firm-days, %d trading days, %d bootstrap replications'
      % (format(len(d), ','), NG, B))


def build(frame, inst_cols, outcome, label):
    need = list(inst_cols) + CONTROLS + [outcome, 'log_pm25']
    s = frame.dropna(subset=need).copy()
    fe = pd.get_dummies(s[['month', 'year', 'dow']].astype(str),
                        drop_first=True).astype(float)
    W = np.column_stack([s[CONTROLS].values, fe.values])
    # Cluster code is the GLOBAL date index, not a per-test relabelling, so
    # one draw of weights lines up across every test in the family.
    codes = s['date'].map(date_ix)
    if codes.isna().any():
        raise SystemExit('%s: %d rows have dates outside the estimation '
                         'calendar' % (label, int(codes.isna().sum())))
    codes = codes.values.astype(int)
    order = np.argsort(codes, kind='stable')     # DK needs time order
    return Test(label, s[outcome].values[order], s['log_pm25'].values[order],
                s[inst_cols].values[order], W[order], codes[order])


def lead_frame(base, k, trading_day):
    """Outcome at t+k against pollution at t."""
    lead = base[['date', 'h_code', 'd_prem_raw']].copy()
    if trading_day:
        shift = {t: TRADING[i - k] for i, t in enumerate(TRADING) if i >= k}
        lead['date'] = lead['date'].map(shift)
        lead = lead.dropna(subset=['date'])
    else:
        lead['date'] = lead['date'] - pd.Timedelta(days=k)
    lead = lead.rename(columns={'d_prem_raw': 'lead_prem'})
    return (base.drop(columns=['d_prem_raw'])
            .merge(lead, on=['date', 'h_code'], how='inner'))


def future_frame(base, trading_day):
    nxt = fs[['date', 'log_pm25'] + INST].copy()
    if trading_day:
        shift = {t: TRADING[i - 1] for i, t in enumerate(TRADING) if i >= 1}
        nxt['date'] = nxt['date'].map(shift)
        nxt = nxt.dropna(subset=['date'])
    else:
        nxt['date'] = nxt['date'] - pd.Timedelta(days=1)
    nxt = nxt.rename(columns={c: 'f_' + c for c in ['log_pm25'] + INST})
    out = (base[['date', 'h_code', 'd_prem_raw'] + CONTROLS
                + ['month', 'year', 'dow']]
           .merge(nxt, on='date', how='inner'))
    return out.rename(columns={'f_log_pm25': 'log_pm25'})


def city_frame(tag):
    tc = d[['date', 'h_code', 'd_prem_raw']].merge(three, on='date',
                                                   how='inner')
    cols = ['%s_%s' % (tag, c) for c in INST]
    sub = tc.dropna(subset=cols + ['%s_log_pm25' % tag]).copy()
    sub = sub.rename(columns={'%s_log_pm25' % tag: 'log_pm25'})
    for c in CONTROLS:
        sub[c] = sub['%s_%s' % (tag, c)]
    sub['month'] = sub['date'].dt.month
    sub['year'] = sub['date'].dt.year
    sub['dow'] = sub['date'].dt.dayofweek
    return sub, cols


def fx_frame():
    fxd = d.drop_duplicates('date')[['date', 'hkd_cny']].sort_values('date')
    fxd['d_fx'] = np.log(fxd['hkd_cny']).diff()
    return fxd.merge(fs, on='date', how='inner').dropna(subset=['d_fx'])


# ------------------------------------------------------------
# The registered family
# ------------------------------------------------------------
F_INST = ['f_' + c for c in INST]
bj, bj_cols = city_frame('beijing')
ur, ur_cols = city_frame('urumqi')

family = [
    build(future_frame(d, False), F_INST, 'd_prem_raw',
          'future pollution (t+1)'),
    build(bj, bj_cols, 'd_prem_raw', 'beijing pollution'),
    build(ur, ur_cols, 'd_prem_raw', 'urumqi pollution'),
    build(fx_frame(), INST, 'd_fx', 'FX placebo (HKD/CNY)'),
    build(lead_frame(d, 1, False), INST, 'lead_prem', 'premium at t+1'),
    build(lead_frame(d, 2, False), INST, 'lead_prem', 'premium at t+2'),
]

# The deviation: the same leads on trading days instead of calendar days.
deviation = [
    build(future_frame(d, True), F_INST, 'd_prem_raw',
          'future pollution (t+1), trading day'),
    build(lead_frame(d, 1, True), INST, 'lead_prem',
          'premium at t+1, trading day'),
    build(lead_frame(d, 2, True), INST, 'lead_prem',
          'premium at t+2, trading day'),
]


def romano_wolf(tests, label):
    for t in tests:
        t.prepare_null()
    fitted = [t.fit() for t in tests]
    betas = np.array([f[0] for f in fitted])
    obs = np.array([f[0] / f[1] for f in fitted])

    rng = np.random.default_rng(SEED)
    boot = np.empty((B, len(tests)))
    for b in range(B):
        w = rng.choice(np.array([-1.0, 1.0]), size=NG)
        for j, t in enumerate(tests):
            bb, ss = t.resample(w)
            boot[b, j] = bb / ss

    # Stepdown: largest |t| first, maxima taken only over the tests still
    # in contention, p-values forced monotone.
    a = np.abs(obs)
    order = np.argsort(-a)
    p = np.empty(len(tests))
    running = 0.0
    for step, j in enumerate(order):
        rest = order[step:]
        mx = np.abs(boot[:, rest]).max(axis=1)
        raw = (1.0 + np.sum(mx >= a[j])) / (B + 1.0)
        running = max(running, raw)
        p[j] = running

    print('\n' + '=' * 76)
    print(label)
    print('=' * 76)
    print('   %-38s %10s %7s %9s %9s'
          % ('test', 'bp per sd', 't', 'p unadj', 'p  RW'))
    for j in np.argsort(-a):
        t = tests[j]
        praw = 2.0 * (1.0 - _norm_cdf(a[j]))
        print('   %-38s %10.2f %7.2f %9.3f %9.3f'
              % (t.label, betas[j] * BP, obs[j], praw, p[j]))
    print('\n   N by test: ' + ', '.join('%s %s' % (t.label.split(',')[0],
                                                    format(t.n, ','))
                                         for t in tests))
    return obs, p


def _norm_cdf(z):
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


romano_wolf(family, 'REGISTERED FALSIFICATION FAMILY, Romano-Wolf stepdown')
romano_wolf(deviation,
            'DISCLOSED DEVIATION: trading-day leads instead of calendar days')

print("""
HOW TO READ THIS
  Every registered falsification test should carry a large RW p-value. A
  test that survives the correction is a test the design failed, and must
  be reported as such rather than explained away.

  The persistence rows are the ones to watch. If t+1 and t+2 lose
  significance under the correction, as their unadjusted t-statistics near
  1.7 suggest they will, the contemporaneous null stands on its own and
  the suggestive drift is reported as exploratory.

  If the trading-day versions differ materially from the calendar-day
  versions, say so plainly in the paper: the registered code used calendar
  days, the weekday composition differs between the two leads, and the
  honest statement is that the persistence result is fragile to that
  choice.
""")
