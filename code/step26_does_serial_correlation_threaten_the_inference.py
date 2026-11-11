# Source: paper2_meteorology_build.ipynb, code cell 43 (final version of step 26 as run).
# ============================================================
# Step 26 — does serial correlation threaten the inference?
#
# The registration clusters by date. That handles dependence WITHIN a
# day, where ~50 firms share a common shock. It does nothing about
# dependence ACROSS days.
#
# Both the treatment and the instruments are strongly serially
# correlated, because weather is. If the outcome residual also carries
# positive autocorrelation, date-clustered standard errors are biased
# downward and every t-statistic is overstated.
#
# Measured here rather than assumed. Each series is examined on its own;
# the treatment and the outcome are never joined, so this stays inside
# the discipline rule.
#
# What it decides: whether the registered inference is date clustering
# alone, or date clustering plus a HAC/Driscoll-Kraay alternative with
# a stated bandwidth.
# ============================================================

import os
import numpy as np
import pandas as pd
from scipy import stats

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DATA = '/content/drive/MyDrive/bilateral/data'
START, END = '2014-05-13', '2019-12-31'
MAXLAG = 20


def acf(x, nlags):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    denom = x @ x
    return np.array([1.0] + [(x[k:] @ x[:-k]) / denom
                             for k in range(1, nlags + 1)])


def ljung_box(r, n, h):
    """Ljung-Box Q over lags 1..h, with its p-value."""
    q = n * (n + 2) * sum(r[k] ** 2 / (n - k) for k in range(1, h + 1))
    return q, 1 - stats.chi2.cdf(q, h)


def report(name, x):
    n = len(x)
    r = acf(x, MAXLAG)
    se = 1.96 / np.sqrt(n)
    print('\n%s   (n = %d, 95%% band +/- %.4f)' % (name, n, se))
    print('   lag  ' + ''.join('%8d' % k for k in range(1, 11)))
    print('   acf  ' + ''.join('%8.3f' % r[k] for k in range(1, 11)))
    print('   sig  ' + ''.join('%8s' % ('*' if abs(r[k]) > se else '.')
                               for k in range(1, 11)))
    for h in (5, 10, 20):
        q, p = ljung_box(r, n, h)
        print('   Ljung-Box(%2d)  Q = %8.1f   p = %.4g' % (h, q, p))
    # Newey-West rule of thumb, Newey & West (1994)
    nw = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    print('   Newey-West automatic bandwidth: %d lags' % nw)
    # How much would HAC inflate a standard error, roughly?
    infl = np.sqrt(max(1 + 2 * sum((1 - k / (nw + 1)) * r[k]
                                   for k in range(1, nw + 1)), 0.0))
    print('   implied SE inflation factor vs iid: %.3f' % infl)
    return r


print('=' * 70)
print('OUTCOME: daily cross-sectional mean premium change')
print('=' * 70)
P = pd.read_csv(os.path.join(DATA, 'ah_premium_clean.csv'),
                parse_dates=['date'],
                usecols=['date', 'h_code', 'd_prem_raw'])
P = P[(P['date'] >= START) & (P['date'] <= END)].dropna()
day = P.groupby('date')['d_prem_raw'].agg(['mean', 'size'])
day = day[day['size'] >= 10].sort_index()
r_out = report('premium change, daily mean', day['mean'].values)

print('\n' + '=' * 70)
print('TREATMENT AND INSTRUMENT (examined separately, never joined)')
print('=' * 70)
fs = pd.read_csv(os.path.join(DATA, 'shanghai_first_stage_fullyear.csv'),
                 parse_dates=['date'])
fs = fs[(fs['date'] >= START) & (fs['date'] <= END)].sort_values('date')
r_pm = report('log PM2.5', fs['log_pm25'].values)
r_blh = report('log boundary layer height', fs['log_blh'].values)

print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)
se_band = 1.96 / np.sqrt(len(day))
sig_lags = [k for k in range(1, MAXLAG + 1) if abs(r_out[k]) > se_band]
q5, p5 = ljung_box(r_out, len(day), 5)
print('outcome: %d of %d lags outside the band; Ljung-Box(5) p = %.4g'
      % (len(sig_lags), MAXLAG, p5))
if p5 > 0.05 and len(sig_lags) <= 1:
    print("""
  The outcome is close to serially uncorrelated, which is what an
  efficient market implies for a daily price CHANGE. Date clustering
  is then adequate on its own, and the serially correlated treatment
  cannot by itself bias the standard errors.
  REGISTER: date clustering primary, HAC reported as a robustness
  check for completeness.""")
else:
    print("""
  The outcome carries real serial correlation. Date clustering alone
  understates the standard errors by roughly the inflation factor
  printed above, and every t-statistic would be overstated by it.
  REGISTER: a serial-correlation-robust estimator as PRIMARY, either
  Driscoll-Kraay or Newey-West on the daily-collapsed series, with the
  bandwidth fixed in advance at the automatic value printed above.""")

print("""
NOTE ON THE TREATMENT
  log PM2.5 and log BLH will both show strong autocorrelation. That is
  expected and is not itself a problem: a persistent regressor only
  inflates t-statistics when the ERROR is also autocorrelated. The
  outcome's own autocorrelation is the number that decides this.
""")
