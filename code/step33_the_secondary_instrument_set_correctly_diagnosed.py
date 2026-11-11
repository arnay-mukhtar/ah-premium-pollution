# Source: paper2_meteorology_build.ipynb, code cell 62 (final version of step 33 as run).
# ============================================================
# Step 33 — the secondary instrument set, correctly diagnosed
#
# Step 32 reported an effective F of 1636 for the radiosonde instruments.
# That figure is wrong and must not reach the paper. It was computed on
# the FIRM-DAY panel, 55,160 rows, when the treatment varies only by DAY.
# Stacking ~49 firms behind each day's single pollution value inflates F
# by roughly that factor without adding a single independent observation
# about the first stage. The partial R-squared of 0.0552 was the honest
# number all along.
#
# Here the first stage is computed where it belongs, on the daily series,
# for both instrument sets.
#
# The second job separates two things step 32 changed at once. The
# secondary estimate (-21.3 bp) differs from the primary (+0.73 bp), but
# it also runs on 1,122 days instead of 1,311 and on 55,160 firm-days
# instead of 64,422. Running the PRIMARY instruments on the sounding
# subsample says whether the difference is the instruments or the sample.
# Without that, the comparison is uninterpretable.
#
# Nothing here is a new specification. The secondary set is registered;
# the subsample comparison is a diagnostic of it and is labelled as such.
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
SESOI_BP = 25.0

INST = ['log_blh', 'u', 'v', 'surface_pressure']
SEC = ['dT_200_100', 'dT_1000_300']
CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']

import sys
sys.path.insert(0, DATA)
sys.path.insert(0, '/content')
import ar_inference as ari


def _dk(S, lags):
    om = S.T @ S
    for j in range(1, int(lags) + 1):
        if j >= len(S):
            break
        w = 1.0 - j / (lags + 1.0)
        G = S[j:].T @ S[:-j]
        om += w * (G + G.T)
    return om


def tsls(y, x, Z, W, codes, G, lags=LAGS):
    n = len(y)
    Zf = np.column_stack([np.ones(n), Z, W])
    Xf = np.column_stack([x, np.ones(n), W])
    P, *_ = np.linalg.lstsq(Zf, Xf, rcond=None)
    Xh = Zf @ P
    A = Xh.T @ Xf
    b = np.linalg.solve(A, Xh.T @ y)
    e = y - Xf @ b
    S = np.zeros((G, Xh.shape[1]))
    np.add.at(S, codes, Xh * e[:, None])
    Ai = np.linalg.inv(A)
    V = (Ai @ _dk(S, lags) @ Ai.T
         * (G / (G - 1.0)) * ((n - 1.0) / (n - Xf.shape[1])))
    return float(b[0]), float(np.sqrt(V[0, 0]))


def effective_f(y, Z, W):
    """Montiel Olea-Pflueger effective F. Pass DAILY series only."""
    Wc = np.column_stack([np.ones(len(y)), W])
    cf, *_ = np.linalg.lstsq(Wc, np.column_stack([y, Z]), rcond=None)
    R = np.column_stack([y, Z]) - Wc @ cf
    yt, Zt = R[:, 0], R[:, 1:]
    ZtZ = Zt.T @ Zt
    pi = np.linalg.solve(ZtZ, Zt.T @ yt)
    e = yt - Zt @ pi
    n, k = Zt.shape
    br = np.linalg.inv(ZtZ)
    V = (br @ (Zt.T @ (Zt * (e ** 2)[:, None])) @ br
         * (n / max(n - k - Wc.shape[1], 1)))
    return (float(pi @ ZtZ @ pi / np.trace(ZtZ @ V)),
            float(1.0 - (e @ e) / (yt @ yt)), n)


def daily_design(frame, inst_cols):
    """One row per day: the level at which the first stage lives."""
    s = frame.drop_duplicates('date').dropna(
        subset=list(inst_cols) + CONTROLS + ['log_pm25'])
    fe = pd.get_dummies(s[['month', 'year', 'dow']].astype(str),
                        drop_first=True).astype(float)
    W = np.column_stack([s[CONTROLS].values, fe.values])
    return s['log_pm25'].values, s[inst_cols].values, W


def panel_design(frame, inst_cols, outcome):
    need = list(inst_cols) + CONTROLS + [outcome, 'log_pm25']
    s = frame.dropna(subset=need).copy()
    fe = pd.get_dummies(s[['month', 'year', 'dow']].astype(str),
                        drop_first=True).astype(float)
    W = np.column_stack([s[CONTROLS].values, fe.values])
    _, codes = np.unique(s['date'].values, return_inverse=True)
    return (s[outcome].values, s['log_pm25'].values,
            s[inst_cols].values, W, codes, codes.max() + 1, s)


# ------------------------------------------------------------
# Load and build the two samples
# ------------------------------------------------------------
prem = pd.read_csv(os.path.join(DATA, 'ah_premium_clean.csv'),
                   parse_dates=['date'], dtype={'h_code': str},
                   usecols=['date', 'h_code', 'd_prem_raw'])
fs = pd.read_csv(os.path.join(DATA, 'shanghai_first_stage_fullyear.csv'),
                 parse_dates=['date'])
snd = pd.read_csv(os.path.join(DATA, 'sounding_daily.csv'),
                  parse_dates=['date'])
snd = snd[snd['city'].astype(str) == 'shanghai']

full = prem.merge(fs, on='date', how='inner')
full = full[(full['date'] >= START) & (full['date'] <= END)].copy()
sd_pm = full.groupby('date')['log_pm25'].first().std()
BP = 10000.0 * sd_pm

sub = full.drop(columns=[c for c in SEC if c in full.columns])
sub = sub.merge(snd[['date'] + SEC], on='date', how='inner').dropna(
    subset=SEC)

print('full sample     %s firm-days, %d days'
      % (format(len(full), ','), full['date'].nunique()))
print('sounding subset %s firm-days, %d days'
      % (format(len(sub), ','), sub['date'].nunique()))

# ------------------------------------------------------------
# 1. First stages, at the DAY level
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('1. FIRST STAGE, COMPUTED ON DAILY DATA')
print('=' * 74)
print('   %-34s %8s %10s %12s' % ('', 'days', 'F_eff', 'partial R2'))
for frame, inst, lab in [
        (full, INST, 'primary, full sample'),
        (sub, INST, 'primary, sounding subsample'),
        (sub, SEC, 'secondary (radiosonde)')]:
    y_, Z_, W_ = daily_design(frame, inst)
    F, pr2, nd = effective_f(y_, Z_, W_)
    print('   %-34s %8d %10.2f %12.4f' % (lab, nd, F, pr2))

print('''
   Step 32 printed 1636 for the secondary set. That was computed on
   55,160 firm-days for a treatment that takes one value per day, which
   multiplies F by roughly the number of firms. The figures above are
   the ones that belong in the paper.''')

# ------------------------------------------------------------
# 2. Second stage: is the gap the instruments or the sample?
# ------------------------------------------------------------
print('\n' + '=' * 74)
print('2. SECOND STAGE, THREE WAYS')
print('=' * 74)
print('   %-34s %11s %11s %7s %10s'
      % ('', 'beta', 'se', 't', 'bp per sd'))
res = {}
for frame, inst, lab in [
        (full, INST, 'primary, full sample'),
        (sub, INST, 'primary, sounding subsample'),
        (sub, SEC, 'secondary (radiosonde)')]:
    y, x, Z, W, codes, G, _ = panel_design(frame, inst, 'd_prem_raw')
    b, se = tsls(y, x, Z, W, codes, G, LAGS)
    res[lab] = (b, se, y, x, Z, W, codes)
    print('   %-34s %11.6f %11.6f %7.2f %10.2f'
          % (lab, b, se, b / se, b * BP))

print('\n   Anderson-Rubin sets, Driscoll-Kraay(7):')
for lab in res:
    b, se, y, x, Z, W, codes = res[lab]
    out = []
    for a in (0.05, 0.10):
        r = ari.ar_confidence_set(y, x, Z, W, codes, alpha=a, lags=LAGS)
        out.append('UNBOUNDED' if (r['unbounded'] or r.get('empty'))
                   else '[%+.1f, %+.1f]' % (r['lo'] * BP, r['hi'] * BP))
        if a == 0.10:
            last = r
    inside = (not (last['unbounded'] or last.get('empty'))
              and abs(last['lo'] * BP) < SESOI_BP
              and abs(last['hi'] * BP) < SESOI_BP)
    print('   %-34s 95%% %-20s 90%% %-20s  %s'
          % (lab, out[0], out[1],
             'EQUIVALENT' if inside else 'not equivalent'))

print('''
HOW TO READ ROW 2
   The middle row is the key one. It holds the instruments fixed and
   changes only the sample.

   If the primary instruments on the sounding subsample also move
   toward the secondary estimate, the difference is the SAMPLE: the
   1,122 days with a sounding are not a random subset of the 1,311, and
   both results are consistent.

   If the middle row stays close to +0.73 bp, the difference is the
   INSTRUMENTS, and the paper must say that the observed-atmosphere
   instruments give a different point estimate, note that the two
   confidence sets overlap heavily, and leave it there. With a
   first-stage partial R-squared of 0.055 and a set three times wider,
   the secondary estimate is not evidence against the primary result.
   It is a weaker test that cannot distinguish the primary estimate
   from a 20 bp effect, which is itself worth reporting.
''')
