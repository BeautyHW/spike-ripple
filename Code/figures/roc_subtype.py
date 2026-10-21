import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

OUT = r'<OUTPUT_DIR>'

df = pd.read_csv(r'<PROJECT_DIR>\part7\part7_loo_perpatient.csv')

series = {
    'Single-spike SR': (df['single_loo'], df['grade']),
    'Poly-spike SR':   (df['poly_loo'],   df['grade']),
}

COL = {'Single-spike SR': '#B2182B', 'Poly-spike SR': '#3A79C0'}
LW  = {'Single-spike SR': 2.4,       'Poly-spike SR': 1.8}

GRID = np.linspace(0, 1, 101)
rng = np.random.default_rng(20260820)

def roc_grid(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return np.interp(GRID, fpr, tpr)

aucs, curves, bands = {}, {}, {}
for name, (c, g) in series.items():
    c = np.asarray(c, float); g = np.asarray(g, int); ok = ~np.isnan(c)
    y, s = g[ok], c[ok]
    n = len(y)
    print(f'{name}: N={n}, Engel I={int(y.sum())}, Engel II-IV={int(n-y.sum())}')

    aucs[name] = roc_auc_score(y, s)
    curves[name] = roc_grid(y, s)

    tprs = np.empty((1000, len(GRID)))
    boot_aucs = np.empty(1000)
    for b in range(1000):
        idx = rng.integers(0, n, n)
        yy, ss = y[idx], s[idx]
        if len(np.unique(yy)) < 2:
            tprs[b] = np.nan; boot_aucs[b] = np.nan; continue
        fpr_b, tpr_b, _ = roc_curve(yy, ss)
        tprs[b] = np.interp(GRID, fpr_b, tpr_b)
        boot_aucs[b] = roc_auc_score(yy, ss)

    bands[name] = np.nanpercentile(tprs, [2.5, 97.5], axis=0)
    bv = boot_aucs[~np.isnan(boot_aucs)]
    ci_lo, ci_hi = np.percentile(bv, 2.5), np.percentile(bv, 97.5)
    print(f'  AUC = {aucs[name]:.4f}  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]')

fig, ax = plt.subplots(figsize=(5.0, 4.8))

ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1.2, zorder=1)

for name in ['Single-spike SR', 'Poly-spike SR']:
    ax.fill_between(GRID, bands[name][0], bands[name][1],
                    color=COL[name], alpha=0.12, linewidth=0, zorder=1)
    ax.plot(GRID, curves[name], color=COL[name], linewidth=LW[name],
            zorder=3, solid_capstyle='round')

items = sorted(((aucs[n], n) for n in aucs), reverse=True)
y0, dy = 0.055, 0.049
for i, (a, n) in enumerate(items):
    ax.text(0.97, y0 + i * dy, f'{n} (AUC = {a:.3f})',
            transform=ax.transAxes, fontsize=10, color=COL[n],
            va='bottom', ha='right',
            fontweight='bold' if n == 'Single-spike SR' else 'normal',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=1.2),
            zorder=5)

ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
ax.set_xlabel('1 − Specificity', fontsize=13)
ax.set_ylabel('Sensitivity', fontsize=13)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2); ax.spines['bottom'].set_linewidth(2)
ax.tick_params(width=2, length=5, labelsize=12)
ax.set_box_aspect(1)
fig.subplots_adjust(left=0.13, right=0.97, top=0.96, bottom=0.12)

fig.savefig(OUT + r'\fig_roc_single_poly.png', dpi=300)
fig.savefig(OUT + r'\fig_roc_single_poly.pdf')
print('\nsaved fig_roc_single_poly')
