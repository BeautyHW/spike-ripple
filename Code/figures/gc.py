import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

OUT = r'<OUTPUT_DIR>'
C1, C0 = '#D55E00', '#0072B2'
Y_SR, Y_CL = 0.45, 0.08
ECL = 0.415

E1 = pd.read_csv(r'<PROJECT_DIR>\part5\sr_gc_engel1.csv')
IIV = pd.read_csv(r'<PROJECT_DIR>\part5\sr_gc_iiv.csv')
E1S = pd.read_csv(r'<PROJECT_DIR>\part5\sr_gc_source_engel1.csv')
IIVS = pd.read_csv(r'<PROJECT_DIR>\part5\sr_gc_source_iiv.csv')

def stars(p):
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))

def grad_box(ax, x, w, q1, q3, color, ylim):
    rgb = to_rgb(color)
    light = tuple(min(1.0, c + (1.0 - c) * 0.80) for c in rgb)
    dark = tuple(max(0.0, c * 0.74) for c in rgb)
    cmap = LinearSegmentedColormap.from_list('g', [light, dark, light])
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, extent=[x - w / 2, x + w / 2, max(q1, -ylim), min(q3, ylim)],
              aspect='auto', cmap=cmap, vmin=0, vmax=1, zorder=1, interpolation='bilinear')
    ax.add_patch(Rectangle((x - w / 2, max(q1, -ylim)), w,
                           min(q3, ylim) - max(q1, -ylim), fill=False,
                           edgecolor='#666666', linewidth=2.2, zorder=3))

def one_box(ax, x, w, v, col, rng, ylim, ystar_frac=0.86):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    jit = rng.uniform(-w * 0.28, w * 0.28, len(v))
    inside = v[np.abs(v) <= ylim]
    ax.scatter(x + jit[:len(inside)], inside, s=15, color=col, alpha=0.6,
               edgecolor='none', zorder=2)
    epos = ylim * 0.92; n_out = 0

    for k in range(int((v > ylim).sum())):
        ax.scatter(x + [-0.13, 0.0, 0.13][k % 3], ylim * (0.92 - 0.13 * (k // 3)),
                   s=26, marker='^', facecolor='none', edgecolor=col, linewidth=1.2, zorder=5)
        n_out += 1
    for k in range(int((v < -ylim).sum())):
        ax.scatter(x + [-0.13, 0.0, 0.13][k % 3], -ylim * (0.92 - 0.13 * (k // 3)),
                   s=26, marker='v', facecolor='none', edgecolor=col, linewidth=1.2, zorder=5)
        n_out += 1
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo = v[v >= q1 - 1.5 * iqr].min(); hi = v[v <= q3 + 1.5 * iqr].max()
    grad_box(ax, x, w, q1, q3, col, ylim)
    ax.plot([x - w / 2, x + w / 2], [med, med], color='black', lw=1.4, zorder=4)
    ax.plot([x, x], [min(q3, ylim), min(hi, ylim)], color='#808080', lw=1.3, zorder=3)
    ax.plot([x, x], [max(q1, -ylim), max(lo, -ylim)], color='#808080', lw=1.3, zorder=3)
    p = wilcoxon(v).pvalue if np.any(v != 0) else 1.0
    ax.text(x, ylim * ystar_frac, stars(p), ha='center', va='center', fontsize=11,
            fontweight='bold', clip_on=False)
    return med, p, len(v), n_out

def style(ax, ylim, yticks=None):
    ax.axhline(0, color='grey', linestyle='--', linewidth=1.1, zorder=1)
    ax.set_ylim(-ylim, ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2); ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(width=2, length=5, labelsize=10.5)

def slope(ax, sr, cl, rng):
    for gi, (col, s, c) in enumerate([(C1, sr[0], cl[0]), (C0, sr[1], cl[1])]):
        s = np.asarray(s, float); c = np.asarray(c, float)
        for sv, cv in zip(s, c):
            sv2 = np.clip(sv, -ECL, ECL); cv2 = np.clip(cv, -ECL, ECL)
            ax.plot([0, 1], [sv2, cv2], color=col, alpha=0.4, lw=0.9, zorder=2)
            for xx, vv, raw in [(0, sv2, sv), (1, cv2, cv)]:
                if abs(raw) > Y_SR:
                    ax.scatter(xx, vv, s=26, marker='^' if raw > 0 else 'v',
                               facecolor='none', edgecolor=col, linewidth=1.2, zorder=4)
                else:
                    ax.scatter(xx, vv, s=15, color=col, alpha=0.7, zorder=3)
        arr = np.column_stack([s, c]); ok = ~np.isnan(arr).any(axis=1); arr = arr[ok]
        meds = np.empty((1000, 2))
        for b in range(1000):
            idx = rng.integers(0, len(arr), len(arr))
            meds[b] = np.median(arr[idx], axis=0)
        lo = np.clip(np.percentile(meds, 2.5, axis=0), -ECL, ECL)
        hi = np.clip(np.percentile(meds, 97.5, axis=0), -ECL, ECL)
        ax.fill_between([0, 1], lo, hi, color=col, alpha=0.18, zorder=1)
        ms, mc = np.clip(np.median(arr, axis=0), -ECL, ECL)
        ax.plot([0, 1], [ms, mc], color=col, lw=3.0, zorder=5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['SR\nwindows', 'Clean\ninterictal'], fontsize=10.5)
    ax.set_xlim(-0.3, 1.3)
    style(ax, Y_SR)

fig = plt.figure(figsize=(12.2, 15.8))
outer = fig.add_gridspec(3, 2, width_ratios=[1.55, 1.0], height_ratios=[1, 1, 1.05],
                         hspace=0.30, wspace=0.24, left=0.06, right=0.975,
                         top=0.96, bottom=0.095)

LINKS = [('Source → dense non-source', 'src_dense'),
         ('Dense non-source → other', 'dense_neither'),
         ('Source → other', 'src_neither')]

for row, (lab, title, ksr, kcl, dfs) in enumerate([
        ('A', 'Dense region level (dense → non-dense)',
         'sr_net_50_150', 'clean_net_50_150', (E1, IIV)),
        ('B', 'Source channel level (source → non-source)',
         'sr_src_rest_50_150', 'clean_src_rest_50_150', (E1S, IIVS))]):
    gl = outer[row, 0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.35)
    axS = fig.add_subplot(gl[0, 0]); axC = fig.add_subplot(gl[0, 1])
    axR = fig.add_subplot(outer[row, 1])
    rng = np.random.default_rng(row)
    for gi, col in enumerate([C1, C0]):
        med, p, n, no = one_box(axS, gi, 0.34, dfs[gi][ksr], col, rng, Y_SR)
        print(f'{lab} SR  {"E1 " if gi==0 else "IIV"} n={n} med={med:+.3f} p={p:.2g} {stars(p)} off={no}')
        med, p, n, no = one_box(axC, gi, 0.34, dfs[gi][kcl], col, rng, Y_CL)
        print(f'{lab} CLN {"E1 " if gi==0 else "IIV"} n={n} med={med:+.3f} p={p:.2g} {stars(p)} off={no}')
    style(axS, Y_SR); style(axC, Y_CL, yticks=[-0.05, 0, 0.05])
    axS.set_xlim(-0.55, 1.55); axC.set_xlim(-0.55, 1.55)
    axS.set_xticks([0, 1]); axS.set_xticklabels(['Engel I', 'Engel II-IV'], fontsize=10.5)
    axC.set_xticks([0, 1]); axC.set_xticklabels(['Engel I', 'Engel II-IV'], fontsize=10.5)
    axS.set_ylabel('Net Granger causality', fontsize=12)
    axS.set_title(f'{lab}.  {title} — SR windows', fontsize=12.5, loc='left', pad=6)
    axC.set_title('Clean interictal', fontsize=11.5, loc='center', pad=6)
    slope(axR, (dfs[0][ksr], dfs[1][ksr]), (dfs[0][kcl], dfs[1][kcl]),
          np.random.default_rng(10 + row))
    axR.set_title(f"{lab}'.  Patient level", fontsize=12.5, loc='left', pad=6)
    axR.set_ylabel('Net Granger causality', fontsize=12)

gl = outer[2, 0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.32)
rng = np.random.default_rng(2)
axCS = fig.add_subplot(gl[0, 0]); axCC = fig.add_subplot(gl[0, 1])
for axx, cond, ylim in [(axCS, 'sr', Y_SR), (axCC, 'clean', Y_CL)]:
    for li, (lname, key) in enumerate(LINKS):
        for gi, col in enumerate([C1, C0]):
            x = li * 1.5 + gi * 0.5
            df = E1S if gi == 0 else IIVS
            med, p, n, no = one_box(axx, x, 0.36, df[f'{cond}_{key}_50_150'], col, rng, ylim)
            print(f'C {key} {cond.upper():4s} {"E1 " if gi==0 else "IIV"} n={n} med={med:+.3f} '
                  f'p={p:.2g} {stars(p)} off={no}')
    style(axx, ylim, yticks=[-0.05, 0, 0.05] if ylim == Y_CL else None)
    axx.set_xlim(-0.65, 4.15)
    axx.set_xticks([li * 1.5 + 0.25 for li in range(3)])
    axx.set_xticklabels(['Source →\ndense non-src', 'Dense non-src\n→ other', 'Source →\nother'],
                        fontsize=10)
axCS.set_title('C.  Stepwise driving level — SR windows', fontsize=12.5, loc='left', pad=6)
axCC.set_title('Clean interictal', fontsize=11.5, loc='center', pad=6)
axCS.set_ylabel('Net Granger causality', fontsize=12)

axP = fig.add_subplot(outer[2, 1])
MK = [('src_dense', 'o', -0.24), ('dense_neither', 's', 0.0), ('src_neither', '^', 0.24)]
frames = [(E1S, C1, 'Engel I'), (IIVS, C0, 'Engel II-IV')]
x = 0
xt, xt_lab = [], []
for fi, (df, col, gname) in enumerate(frames):
    cols = [f'sr_{k}_50_150' for k, _, _ in MK]
    sub = df[cols].copy()
    order = sub.median(axis=1, skipna=True).sort_values(ascending=False).index
    for xi, idx in enumerate(order):
        for k, m, dx in MK:
            v = df.loc[idx, f'sr_{k}_50_150']
            if pd.isna(v):
                continue
            if abs(v) > Y_SR:
                axP.scatter(x + dx, np.sign(v) * ECL, s=22, marker=m, facecolor='none',
                            edgecolor=col, linewidth=1.1, zorder=4)
            else:
                axP.scatter(x + dx, v, s=16, marker=m, color=col, alpha=0.75, zorder=3)
        x += 1
    xt.append(x - len(order) / 2); xt_lab.append(f'{gname}\n(n={len(order)})')
    if fi == 0:
        sep = x; x += 1.2
axP.axhline(0, color='grey', linestyle='--', linewidth=1.1, zorder=1)
axP.axvline(sep + 0.6, color='#BBBBBB', linestyle=':', linewidth=1.0, zorder=1)
axP.set_xticks(xt); axP.set_xticklabels(xt_lab, fontsize=10.5)
axP.set_xlim(-1, x)
style(axP, Y_SR)
axP.set_title("C'.  Patient level — three links", fontsize=12.5, loc='left', pad=6)
axP.set_ylabel('Net Granger causality (SR windows)', fontsize=12)
fig.legend(handles=[Patch(facecolor=C1, edgecolor='#666666', label='Engel I'),
                    Patch(facecolor=C0, edgecolor='#666666', label='Engel II-IV'),
                    Line2D([], [], marker='o', color='#555555', linestyle='none',
                           label='Src → dense non-src'),
                    Line2D([], [], marker='s', color='#555555', linestyle='none',
                           label='Dense non-src → other'),
                    Line2D([], [], marker='^', color='#555555', linestyle='none',
                           label='Src → other')],
           fontsize=11, loc='lower center', frameon=False, ncol=5,
           bbox_to_anchor=(0.5, 0.005))

fig.savefig(OUT + r'\fig6_gc_v4.png', dpi=300)
fig.savefig(OUT + r'\fig6_gc_v4.pdf')
print('saved fig6_gc_v4')
