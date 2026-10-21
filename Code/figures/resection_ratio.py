import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle, Patch
from scipy.stats import mannwhitneyu

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

OUT = r'<OUTPUT_DIR>'
COL_ENGEL1, COL_ENGEL24 = '#C27B7B', '#3A79C0'
BOX_W, DX = 0.30, 0.8

d = pd.read_csv(r'<PROJECT_DIR>\part2\part2_quantile_completeness_perpt.csv')
n3 = pd.read_csv(OUT + r'\_fig_new3_perpatient.csv')
d = d.merge(n3[['patient', 'swr_loo', 'rws_loo', 'sg_loo']], on='patient', how='left')
BIOMK = [('SR', 'sr'), ('Spike', 'spike'), ('Ripple', 'ripple'),
         ('Broadband HFO', 'wb'), ('FR', 'fr'),
         ('Spike without\nripple', 'swr_loo'), ('Ripple without\nspike', 'rws_loo'),
         ('Spike-gamma', 'sg_loo')]

_CHK = {'swr_loo': (0.515, 0.0067), 'rws_loo': (0.381, 0.024), 'sg_loo': (0.519, 0.0020)}
for _k, (_r, _p) in _CHK.items():
    c = np.asarray(d[_k], float); g = np.asarray(d['grade'], int); ok = ~np.isnan(c)
    _u, _pu = mannwhitneyu(c[ok & (g == 1)], c[ok & (g == 0)],
                            alternative='two-sided', method='asymptotic')
    _rr = 2 * _u / ((g[ok] == 1).sum() * (g[ok] == 0).sum()) - 1
    assert abs(_rr - _r) < 5e-3 and abs(_pu - _p) < 5e-4, (_k, _rr, _pu)

def stars(p):
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))

def draw_gradient_box(ax, x, w, q1, q3, color, edge_color='#666666', edge_lw=2.5):
    rgb = to_rgb(color)
    light = tuple(min(1.0, c + (1.0 - c) * 0.80) for c in rgb)
    dark = tuple(max(0.0, c * 0.74) for c in rgb)
    cmap = LinearSegmentedColormap.from_list('g', [light, dark, light])
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, extent=[x - w / 2, x + w / 2, q1, q3], aspect='auto',
              cmap=cmap, vmin=0, vmax=1, zorder=1, interpolation='bilinear')
    ax.add_patch(Rectangle((x - w / 2, q1), w, q3 - q1, fill=False,
                           edgecolor=edge_color, linewidth=edge_lw, zorder=3))

fig, ax = plt.subplots(figsize=(11.2, 5.6))
for i, (name, key) in enumerate(BIOMK):
    gx = i + DX
    c = np.asarray(d[key], float); g = np.asarray(d['grade'], int)
    ok = ~np.isnan(c)
    c1 = c[ok & (g == 1)]; c2 = c[ok & (g == 0)]
    rng = np.random.default_rng(i)
    for x_off, cc, col in [(-0.19, c1, COL_ENGEL1), (+0.19, c2, COL_ENGEL24)]:
        jit = rng.uniform(-0.09, 0.09, len(cc))
        ax.scatter(gx + x_off + jit, cc, s=22, color=col, alpha=0.55, edgecolor='none', zorder=2)
        if len(cc) == 0:
            continue
        q1, med, q3 = np.percentile(cc, [25, 50, 75])
        iqr = q3 - q1
        lo = cc[cc >= q1 - 1.5 * iqr].min(); hi = cc[cc <= q3 + 1.5 * iqr].max()
        x = gx + x_off
        draw_gradient_box(ax, x, BOX_W, q1, q3, col)
        ax.plot([x - BOX_W / 2, x + BOX_W / 2], [med, med], color='black', lw=1.5, zorder=4)
        ax.plot([x, x], [q3, hi], color='#808080', lw=1.4, zorder=3)
        ax.plot([x, x], [q1, lo], color='#808080', lw=1.4, zorder=3)
        ax.plot([x - BOX_W / 4, x + BOX_W / 4], [hi, hi], color='#808080', lw=1.4, zorder=3)
        ax.plot([x - BOX_W / 4, x + BOX_W / 4], [lo, lo], color='#808080', lw=1.4, zorder=3)
    u, p = mannwhitneyu(c1, c2, alternative='two-sided', method='asymptotic')
    r = 2 * (u / (len(c1) * len(c2))) - 1
    y = np.nanmax(c) + 0.06
    ax.plot([gx - 0.19, gx - 0.19, gx + 0.19, gx + 0.19],
            [y, y + 0.012, y + 0.012, y], color='#808080', lw=1.4, clip_on=False)
    ax.text(gx, y + 0.018, f'{stars(p)}\nr = {r:.2f}', ha='center', va='bottom', fontsize=12)
    print(f'{name:<15} n1={len(c1)} med1={np.median(c1):.3f}  n0={len(c2)} med0={np.median(c2):.3f}  '
          f'MWU p={p:.3g}  r={r:.3f}')

ax.set_xticks([i + DX for i in range(len(BIOMK))])
ax.set_xticklabels([b[0] for b in BIOMK], fontsize=12)
ax.set_ylabel('Dense-region resection ratio', fontsize=14)
ax.set_ylim(-0.05, 1.30)
ax.set_xlim(0.35, DX + len(BIOMK) - 1 + 0.85)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2); ax.spines['bottom'].set_linewidth(2)
ax.tick_params(width=2, length=5, labelsize=12)
ax.set_box_aspect(0.5)
ax.legend(handles=[Patch(facecolor=COL_ENGEL1, edgecolor='#666666', label='Engel I (n=29)'),
                   Patch(facecolor=COL_ENGEL24, edgecolor='#666666', label='Engel II-IV (n=21)')],
          fontsize=12, loc='lower right', bbox_to_anchor=(1.0, 1.005), ncol=2, frameon=False)
fig.tight_layout()
fig.savefig(OUT + r'\fig4_biomarkers.png', dpi=300)
fig.savefig(OUT + r'\fig4_biomarkers.pdf')
print('saved fig4_biomarkers')
