import importlib.util
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

OUT = r'<OUTPUT_DIR>'

p2 = pd.read_csv(r'<PROJECT_DIR>\part2\part2_quantile_completeness_perpt.csv')
series = {
    'SR':                   (p2['sr'], p2['grade']),
    'Spike':                (p2['spike'], p2['grade']),
    'Ripple':               (p2['ripple'], p2['grade']),
    'Broadband HFO':        (p2['wb'], p2['grade']),
    'FR':                   (p2['fr'], p2['grade']),
}

spec = importlib.util.spec_from_file_location(
    'm', r'<PROJECT_DIR>\part11\_sr_part11_three_markers.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
data = m.load()
pts = list(data.keys())
for label, key, q in [('Spike without ripple', 'spike_only', 85),
                      ('Ripple without spike', 'ripple_only', 85),
                      ('Spike-gamma', 'spike_gamma', 70)]:
    LOO = m.loo_completeness(data, pts, key, q)
    have = [p for p in pts if p in LOO and not np.isnan(LOO[p])]
    series[label] = (pd.Series([LOO[p] for p in have], index=have),
                     pd.Series([data[p]['grade'] for p in have], index=have))

COL = {'SR': '#B2182B', 'Spike': '#3A79C0', 'Ripple': '#2E9E5B',
       'Broadband HFO': '#66B2C8', 'FR': '#8C8C8C',
       'Spike without ripple': '#D98E73', 'Ripple without spike': '#8E5BA6',
       'Spike-gamma': '#E09F3E'}

GRID = np.linspace(0, 1, 101)
rng = np.random.default_rng(20260820)

def roc_grid(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return np.interp(GRID, fpr, tpr)

fig, ax = plt.subplots(figsize=(6.0, 5.4))
aucs, curves, bands = {}, {}, {}
for name, (c, g) in series.items():
    c = np.asarray(c, float); g = np.asarray(g, int); ok = ~np.isnan(c)
    y, s = g[ok], c[ok]
    aucs[name] = roc_auc_score(y, s)
    curves[name] = roc_grid(y, s)

    tprs = np.empty((1000, len(GRID)))
    for b in range(1000):
        idx = rng.integers(0, len(y), len(y))
        yy, ss = y[idx], s[idx]
        if len(np.unique(yy)) < 2:
            tprs[b] = np.nan
            continue
        fpr, tpr, _ = roc_curve(yy, ss)
        tprs[b] = np.interp(GRID, fpr, tpr)
    bands[name] = np.nanpercentile(tprs, [2.5, 97.5], axis=0)

EXP = {'SR': 0.837, 'Spike': 0.728, 'Ripple': 0.727, 'Broadband HFO': 0.659,
       'FR': 0.617, 'Spike without ripple': 0.758, 'Ripple without spike': 0.691,
       'Spike-gamma': 0.759}
for n_, e in EXP.items():
    assert abs(aucs[n_] - e) < 5e-4, (n_, aucs[n_])

for name in sorted(aucs, key=aucs.get):
    lw = 2.4 if name == 'SR' else 1.5
    ax.fill_between(GRID, bands[name][0], bands[name][1],
                    color=COL[name], alpha=0.10 if name != 'SR' else 0.15,
                    linewidth=0, zorder=1)
    ax.plot(GRID, curves[name], color=COL[name], linewidth=lw,
            zorder=3 if name == 'SR' else 2, solid_capstyle='round')

ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1.2, zorder=1)

items = sorted(((aucs[n], n) for n in aucs), reverse=True)
y0, dy = 0.055, 0.049
for i, (a, n) in enumerate(items):
    ax.text(0.97, y0 + i * dy, f'{n} (AUC = {a:.3f})',
            transform=ax.transAxes, fontsize=10, color=COL[n],
            va='bottom', ha='right',
            fontweight='bold' if n == 'SR' else 'normal',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=1.2),
            zorder=5)
for a, n in items:
    print(f'{n:<22} AUC = {a:.3f}  CI [{bands[n][0][60]:.2f}, {bands[n][1][60]:.2f}] @fpr0.6')

ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
ax.set_xlabel('1 − Specificity', fontsize=13)
ax.set_ylabel('Sensitivity', fontsize=13)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2); ax.spines['bottom'].set_linewidth(2)
ax.tick_params(width=2, length=5, labelsize=12)
ax.set_box_aspect(1)
fig.subplots_adjust(left=0.13, right=0.97, top=0.96, bottom=0.12)
fig.savefig(OUT + r'\fig3_roc8.png', dpi=300)
fig.savefig(OUT + r'\fig3_roc8.pdf')
print('saved fig3_roc8')
