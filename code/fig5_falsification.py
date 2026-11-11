# Figure 5 (falsification family), regenerated from the reported estimates:
# primary AR set; Wald 95% intervals from Driscoll-Kraay SEs (se = |bp / t|) for the tests.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
INK, ACCENT, MUTED, BAND = '#1a1a1a', '#1f4e79', '#8a8a8a', '#e6e6e6'
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'savefig.bbox': 'tight'})
w = lambda b, t: (b, b - 1.96 * abs(b / t), b + 1.96 * abs(b / t))
rows = [('Premium (primary)', 0.73, -15.52, 18.07), ('Future pollution', *w(-2.42, -0.34)),
        ('Beijing, with Shanghai', *w(4.29, 0.67)), ('Ürümqi, with Shanghai', *w(-12.62, -0.88)),
        ('Premium at t+1', *w(-14.92, -1.73)), ('Premium at t+2', *w(-15.08, -1.72))][::-1]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.axvspan(-25, 25, color=BAND, zorder=0); ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
for i, (lab, b, lo, hi) in enumerate(rows):
    p = lab.endswith('(primary)'); c = INK if p else ACCENT
    ax.plot([lo, hi], [i, i], color=c, lw=1.6 if p else 1.2, solid_capstyle='butt', zorder=2)
    ax.plot([b], [i], 'o', ms=6 if p else 5, color=c, markeredgecolor='white', markeredgewidth=0.7, zorder=3)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel('effect of a 1 SD rise in log PM2.5, basis points')
span = max(abs(v) for r in rows for v in r[2:]) * 1.12
ax.set_xlim(-span, span); ax.set_ylim(-0.7, len(rows) - 0.1)
ax.text(25 + span * 0.03, len(rows) - 0.55, 'equivalence bound', fontsize=8, color='#5a5a5a')
ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
for ext in ('pdf', 'png'):
    fig.savefig('falsification.' + ext, dpi=300)
