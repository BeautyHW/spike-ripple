import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle
from scipy.stats import wilcoxon

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

OUT = r'<OUTPUT_DIR>'
COL_R, COL_N = '#C27B7B', '#D55E00'
YL = 30.0
YB = 31.6
BRK = 1.0
MK = [('SR', 'sr'), ('Spike', 'spike'), ('Ripple', 'ripple'),
      ('Broadband\nHFO', 'wb'), ('FR', 'fr'),
      ('Spike w/o\nripple', 'swr'), ('Ripple w/o\nspike', 'rws'),
      ('Spike-gamma', 'sg')]

d = pd.read_csv(OUT + r'\_fig5_patient_medians.csv')
n3 = pd.read_csv(OUT + r'\_fig_new3_perpatient.csv')
n3 = n3.rename(columns={pre + '_ezmed': pre + '_ez' for pre in ('swr', 'rws', 'sg')}
               ).rename(columns={pre + '_nezmed': pre + '_nez' for pre in ('swr', 'rws', 'sg')})
d = d.merge(n3[['patient', 'swr_ez', 'swr_nez', 'rws_ez', 'rws_nez',
                'sg_ez', 'sg_nez']], on='patient', how='left')
rng = np.random.default_rng(5)

def stars(p):
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))

def grad_bar(ax, x, w, h, color):
    
    rgb = to_rgb(color)
    light = tuple(min(1.0, c + (1.0 - c) * 0.55) for c in rgb)
    cmap = LinearSegmentedColormap.from_list('g', [light, rgb])
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, extent=[x - w / 2, x + w / 2, 0, h], aspect='auto',
              cmap=cmap, vmin=0, vmax=1, zorder=2, interpolation='bilinear', alpha=0.85)
    ax.add_patch(Rectangle((x - w / 2, 0), w, h, fill=False,
                           edgecolor='#666666', linewidth=1.6, zorder=3))

fig = plt.figure(figsize=(15.6, 7.8))
gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0], hspace=0.07, wspace=0.20,
                      left=0.06, right=0.985, top=0.88, bottom=0.12)
for col, (g, title) in enumerate([(1, 'Engel I (n=29)'), (0, 'Engel II-IV (n=21)')]):
    axT = fig.add_subplot(gs[0, col])
    axB = fig.add_subplot(gs[1, col], sharex=axT)
    sub = d[d.grade == g]
    for i, (_, k) in enumerate(MK):
        ez = np.asarray(sub[k + '_ez'], float)
        nz = np.asarray(sub[k + '_nez'], float)
        w = 0.34
        for xoff, v, col_c in [(-w / 2 - 0.02, ez, COL_R), (w / 2 + 0.02, nz, COL_N)]:
            for ax in (axT, axB):
                grad_bar(ax, i + xoff, w, np.median(v), col_c)
            jit = rng.uniform(-w * 0.30, w * 0.30, len(v))
            inside = v[(v <= YL) & ~np.isnan(v)]
            for ax in (axT, axB):
                ax.scatter(i + xoff + jit[:len(inside)], inside, s=15, color=col_c,
                           alpha=0.5, edgecolor='none', zorder=4)
            for kk in range(int((v > YL).sum())):
                axT.scatter(i + xoff + [-0.07, 0.0, 0.07][kk % 3], YL * (0.97 - 0.012 * (kk // 3)),
                            s=24, marker='^', facecolor='none', edgecolor=col_c,
                            linewidth=1.2, zorder=5)
        ok = ~(np.isnan(ez) | np.isnan(nz))
        p = wilcoxon(ez[ok], nz[ok], zero_method='wilcox').pvalue
        print(f'{"Engel I " if g else "II-IV    "} {k:<7} EZ {np.median(ez[ok]):>6.2f}  '
              f'NEZ {np.median(nz[ok]):>5.2f}  p={p:.2g}  {stars(p)}  off>{YL}: {(ez[ok]>YL).sum()}/{(nz[ok]>YL).sum()}')

        axT.plot([i - w / 2 - 0.02, i - w / 2 - 0.02, i + w / 2 + 0.02, i + w / 2 + 0.02],
                 [YB - 0.5, YB, YB, YB - 0.5], color='black', lw=1.2, zorder=6)
        axT.text(i, YB + 0.4, stars(p), ha='center', va='bottom', fontsize=12,
                 fontweight='bold', zorder=6)

    axT.set_ylim(BRK, 34.5); axB.set_ylim(0, BRK)
    axT.set_yticks([2, 5, 10, 15, 20, 25, 30]); axB.set_yticks([0, 0.25, 0.5, 0.75, 1])
    axT.tick_params(labelbottom=False)
    for ax, t in [(axT, title), (axB, None)]:
        ax.set_xlim(-0.65, 7.65)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2); ax.spines['bottom'].set_linewidth(2)
        ax.tick_params(width=2, length=5, labelsize=11)
    axB.set_xticks(range(8))
    axB.set_xticklabels([m[0] for m in MK], fontsize=11.5)
    axT.set_title(title, fontsize=13.5, pad=8)

    dd = 0.012
    for ax, yb in [(axT, 0), (axB, 1)]:
        ax.plot([0, 1], [yb, yb], transform=ax.transAxes,
                marker=[(-1, -dd), (1, dd)], markersize=11, linestyle='none',
                color='k', mec='k', mew=1.1, clip_on=False)

fig.text(0.022, 0.5, 'Median channel rate (events/min)', rotation=90,
         va='center', fontsize=13)
fig.legend(handles=[Rectangle((0, 0), 1, 1, fc=COL_R, ec='#666666'),
                    Rectangle((0, 0), 1, 1, fc=COL_N, ec='#666666')],
           labels=['Resected region', 'Non-resected region'],
           loc='upper center', ncol=2, fontsize=12.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.text(0.5, 0.015, 'Stars: resected vs non-resected, paired Wilcoxon signed-rank test (patient level);  '
                     'y-axis broken at 1 /min',
         ha='center', fontsize=10.5, style='italic', color='#444444')
fig.savefig(OUT + r'\fig5_rate_patient.png', dpi=300)
fig.savefig(OUT + r'\fig5_rate_patient.pdf')
print('saved fig5_rate_patient')
