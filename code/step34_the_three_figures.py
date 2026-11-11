# Source: paper2_meteorology_build.ipynb, code cell 67 (final version of step 34 as run).
# ============================================================
# Step 34 — the three figures
#
# FIGURE 1  The first stage. A binned scatter of residualised log PM2.5
#           against residualised log boundary layer height. The reader
#           should see the instrument working without taking the F on
#           trust.
#
# FIGURE 2  The Anderson-Rubin p-value curve against the equivalence
#           bounds. This is the paper's central figure: it shows the
#           accepted region sitting inside the SESOI band, which is the
#           equivalence result as a picture rather than a sentence.
#
# FIGURE 3  The falsification family as a coefficient plot, every test
#           with its AR interval, against the same SESOI band. One look
#           establishes that nothing is anywhere.
#
# DESIGN. Single hue plus greys throughout, so every figure survives
# greyscale printing and every form of colour vision deficiency; no
# figure relies on colour to carry identity. Recessive axes, no grid
# boxes, no chartjunk, direct labels rather than legends wherever a
# legend would be the only thing a legend is for.
#
# Saves 300 dpi PNG and vector PDF of each. Journals want the PDF.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

DATA = '/content/drive/MyDrive/bilateral/data'
OUT = '/content/drive/MyDrive/bilateral/figures'
os.makedirs(OUT, exist_ok=True)
START, END = '2014-05-13', '2019-12-31'
LAGS = 7
SESOI_BP = 25.0

INST = ['log_blh', 'u', 'v', 'surface_pressure']
CONTROLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
            'cloud_cover', 'shortwave_radiation']

import sys
sys.path.insert(0, DATA)
sys.path.insert(0, '/content')
import ar_inference as ari

INK = '#1a1a1a'
ACCENT = '#1f4e79'
MUTED = '#8a8a8a'
BAND = '#e6e6e6'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'axes.edgecolor': MUTED,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.color': INK,
    'ytick.color': INK,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 110,
    'savefig.bbox': 'tight',
})


def save(fig, name):
    for ext in ('png', 'pdf'):
        p = os.path.join(OUT, '%s.%s' % (name, ext))
        fig.savefig(p, dpi=300)
    print('   saved %s.png and %s.pdf' % (name, name))
    plt.close(fig)


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


def design(frame, inst_cols, outcome):
    need = list(inst_cols) + CONTROLS + [outcome, 'log_pm25']
    s = frame.dropna(subset=need).copy()
    fe = pd.get_dummies(s[['month', 'year', 'dow']].astype(str),
                        drop_first=True).astype(float)
    W = np.column_stack([s[CONTROLS].values, fe.values])
    _, codes = np.unique(s['date'].values, return_inverse=True)
    return (s[outcome].values, s['log_pm25'].values,
            s[inst_cols].values, W, codes, codes.max() + 1, s)


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------
prem = pd.read_csv(os.path.join(DATA, 'ah_premium_clean.csv'),
                   parse_dates=['date'], dtype={'h_code': str},
                   usecols=['date', 'h_code', 'd_prem_raw', 'hkd_cny'])
fs = pd.read_csv(os.path.join(DATA, 'shanghai_first_stage_fullyear.csv'),
                 parse_dates=['date'])
three = pd.read_csv(os.path.join(DATA, 'three_city_panel.csv'),
                    parse_dates=['date'])

d = prem.merge(fs, on='date', how='inner')
d = d[(d['date'] >= START) & (d['date'] <= END)].sort_values(
    ['h_code', 'date']).copy()
sd_pm = d.groupby('date')['log_pm25'].first().std()
BP = 10000.0 * sd_pm
print('figures from %s firm-days, %d days'
      % (format(len(d), ','), d['date'].nunique()))

# ------------------------------------------------------------
# FIGURE 1 — the first stage
# ------------------------------------------------------------
print('\nFigure 1: first stage')
day = d.drop_duplicates('date').dropna(
    subset=INST + CONTROLS + ['log_pm25']).copy()
fe = pd.get_dummies(day[['month', 'year', 'dow']].astype(str),
                    drop_first=True).astype(float)
# Partial out the controls, the fixed effects AND the other three
# instruments, so the picture shows log BLH's own contribution rather
# than a raw correlation three other variables also explain.
others = [c for c in INST if c != 'log_blh']
Wc = np.column_stack([np.ones(len(day)), day[CONTROLS].values,
                      day[others].values, fe.values])
cf, *_ = np.linalg.lstsq(Wc, np.column_stack([day['log_pm25'].values,
                                              day['log_blh'].values]),
                         rcond=None)
R = np.column_stack([day['log_pm25'].values, day['log_blh'].values]) - Wc @ cf
yres, xres = R[:, 0], R[:, 1]

nb = 25
edges = np.quantile(xres, np.linspace(0, 1, nb + 1))
edges[-1] += 1e-9
idx = np.clip(np.digitize(xres, edges) - 1, 0, nb - 1)
bx = np.array([xres[idx == b].mean() for b in range(nb)])
by = np.array([yres[idx == b].mean() for b in range(nb)])
slope, intercept = np.polyfit(xres, yres, 1)

fig, ax = plt.subplots(figsize=(4.6, 3.3))
ax.axhline(0, color=MUTED, lw=0.5, zorder=1)
ax.axvline(0, color=MUTED, lw=0.5, zorder=1)
ax.scatter(bx, by, s=26, color=ACCENT, zorder=3,
           edgecolor='white', linewidth=0.6)
xs = np.linspace(xres.min(), xres.max(), 100)
ax.plot(xs, intercept + slope * xs, color=INK, lw=1.4, zorder=2)
ax.set_xlabel('log boundary layer height, residualised')
ax.set_ylabel('log PM2.5, residualised')
ax.text(0.03, 0.06,
        'slope %.3f\n%d trading days, 25 equal-count bins'
        % (slope, len(day)),
        transform=ax.transAxes, fontsize=8, color=INK, va='bottom')
save(fig, 'fig1_first_stage')

# ------------------------------------------------------------
# FIGURE 2 — the AR curve and the equivalence bounds
# ------------------------------------------------------------
print('Figure 2: AR p-value curve')
y, x, Z, W, codes, G, _ = design(d, INST, 'd_prem_raw')
b_hat, se_hat = tsls(y, x, Z, W, codes, G, LAGS)
grid = np.linspace(-45.0 / BP, 45.0 / BP, 601)
pv = ari.ar_pvalues(y, x, Z, W, codes, grid, lags=LAGS)
gbp = grid * BP

fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.axvspan(-SESOI_BP, SESOI_BP, color=BAND, zorder=0)
for e in (-SESOI_BP, SESOI_BP):
    ax.axvline(e, color='#b4b4b4', lw=0.8, ls=(0, (3, 2)), zorder=1)
ax.plot(gbp, pv, color=ACCENT, lw=1.6, zorder=3)
for lvl, lab in [(0.05, '5%'), (0.10, '10%')]:
    ax.axhline(lvl, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=2)
    ax.text(44, lvl + 0.015, lab, fontsize=7.5, color=MUTED, ha='right')
ax.axvline(b_hat * BP, color=INK, lw=1.0, ls=':', zorder=3)

# Headroom above the curve so the AR bracket never meets the x-axis
# labels. The earlier version drew it below the axis, where it landed
# on top of the tick labels.
ax.set_xlim(-45, 45)
ax.set_ylim(0, 1.22)
acc = gbp[pv > 0.05]
if len(acc):
    ax.plot([acc.min(), acc.max()], [1.06, 1.06], color=INK, lw=2.0,
            solid_capstyle='butt', zorder=4)
    for e in (acc.min(), acc.max()):
        ax.plot([e, e], [1.03, 1.09], color=INK, lw=1.0, zorder=4)
    ax.text((acc.min() + acc.max()) / 2, 1.115,
            '95%% Anderson-Rubin set   [%.1f, %.1f] bp'
            % (acc.min(), acc.max()),
            ha='center', va='bottom', fontsize=8, color=INK)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel('effect of a 1 SD rise in log PM2.5, basis points')
ax.set_ylabel('Anderson-Rubin p-value')
ax.text(-SESOI_BP + 1.6, 0.55, 'equivalence bound, ±25 bp',
        rotation=90, va='center', ha='left', fontsize=7.5,
        color='#5a5a5a')
save(fig, 'fig2_ar_curve')

# ------------------------------------------------------------
# FIGURE 3 — the falsification family
# ------------------------------------------------------------
print('Figure 3: falsification family (this one takes a few minutes)')
tests = [('Premium (primary)', d, INST, 'd_prem_raw')]

nxt = fs[['date', 'log_pm25'] + INST].copy()
nxt['date'] = nxt['date'] - pd.Timedelta(days=1)
nxt = nxt.rename(columns={c: 'f_' + c for c in ['log_pm25'] + INST})
dfut = (d[['date', 'h_code', 'd_prem_raw'] + CONTROLS
          + ['month', 'year', 'dow']]
        .merge(nxt, on='date', how='inner')
        .rename(columns={'f_log_pm25': 'log_pm25'}))
tests.append(('Future pollution', dfut, ['f_' + c for c in INST],
              'd_prem_raw'))

tc = d[['date', 'h_code', 'd_prem_raw']].merge(three, on='date', how='inner')
for tag, lab in [('beijing', 'Beijing'), ('urumqi', 'Ürümqi')]:
    cols = ['%s_%s' % (tag, c) for c in INST]
    sub = tc.dropna(subset=cols + ['%s_log_pm25' % tag]).copy()
    sub = sub.rename(columns={'%s_log_pm25' % tag: 'log_pm25'})
    for c in CONTROLS:
        sub[c] = sub['%s_%s' % (tag, c)]
    sub['month'] = sub['date'].dt.month
    sub['year'] = sub['date'].dt.year
    sub['dow'] = sub['date'].dt.dayofweek
    tests.append((lab + ' pollution', sub, cols, 'd_prem_raw'))

# CALENDAR-day leads, matching the registered specification in step 27
# and Table 8.1. An earlier version shifted by trading days, which is
# the disclosed deviation of Section 8.2: that gives -10.29 and -5.47
# where the registered family gives -14.92 and -15.08, so the figure
# would have contradicted the table it sits beside.
for k in (1, 2):
    lead = d[['date', 'h_code', 'd_prem_raw']].copy()
    lead['date'] = lead['date'] - pd.Timedelta(days=k)
    lead = lead.rename(columns={'d_prem_raw': 'lead_prem'})
    dl = d.drop(columns=['d_prem_raw']).merge(lead, on=['date', 'h_code'],
                                              how='inner')
    tests.append(('Premium at t+%d' % k, dl, INST, 'lead_prem'))

rows = []
for lab, frame, inst, outcome in tests:
    yy, xx, ZZ, WW, cc, GG, _ = design(frame, inst, outcome)
    bb, ss = tsls(yy, xx, ZZ, WW, cc, GG, LAGS)
    r = ari.ar_confidence_set(yy, xx, ZZ, WW, cc, alpha=0.05, lags=LAGS)
    lo = np.nan if (r['unbounded'] or r.get('empty')) else r['lo'] * BP
    hi = np.nan if (r['unbounded'] or r.get('empty')) else r['hi'] * BP
    rows.append((lab, bb * BP, lo, hi))
    print('   %-22s %7.2f bp   AR95 [%.1f, %.1f]'
          % (lab, bb * BP, lo, hi))

rows = rows[::-1]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.axvspan(-SESOI_BP, SESOI_BP, color=BAND, zorder=0)
ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
for i, (lab, bb, lo, hi) in enumerate(rows):
    primary = lab.endswith('(primary)')
    col = INK if primary else ACCENT
    if np.isfinite(lo):
        ax.plot([lo, hi], [i, i], color=col, lw=1.6 if primary else 1.2,
                solid_capstyle='butt', zorder=2)
    ax.plot([bb], [i], 'o', ms=6 if primary else 5, color=col,
            markeredgecolor='white', markeredgewidth=0.7, zorder=3)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel('effect of a 1 SD rise in log PM2.5, basis points')
# Limits from the data rather than hardcoded, so a wide AR set is not
# silently clipped and a narrow family does not sit in empty space.
fin = [v for _, _, lo, hi in rows for v in (lo, hi) if np.isfinite(v)]
span = max(max(abs(min(fin)), abs(max(fin))), SESOI_BP + 5) * 1.12
ax.set_xlim(-span, span)
ax.set_ylim(-0.7, len(rows) - 0.1)
ax.text(SESOI_BP + span * 0.03, len(rows) - 0.55, 'equivalence bound',
        fontsize=8, color='#5a5a5a')
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)
save(fig, 'fig3_falsification')

print('\nall three written to %s' % OUT)
print('Download the three PNGs and attach them here.')
