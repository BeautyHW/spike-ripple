import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle, Patch
from scipy.stats import wilcoxon

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

SR_DIR = r"<DATA_DIR>\last\<TIMESTAMPS_DIR>"
FR_ROOT = r"<DATA_DIR>\last\<TIMESTAMPS_DIR>"
V2 = r"<DATA_DIR>\last\<TIMESTAMPS_DIR>\<TIMESTAMPS_DIR>"
LABEL_I = r"<DATA_DIR>\<DATA_DIR>\labels"
COOC_WIN = 0.035
OUT = r'<OUTPUT_DIR>'


def sr_count(spike_pos, rs, re_):
    sp = np.sort(np.asarray(spike_pos, float)); rs = np.asarray(rs, float); re_ = np.asarray(re_, float)
    if len(rs) == 0 or len(sp) == 0:
        return 0
    rmid = (rs + re_) / 2.0
    idx = np.searchsorted(sp, rmid)
    best = np.full(len(rmid), np.inf)
    for off in (0, -1):
        j = idx + off; valid = (j >= 0) & (j < len(sp)); jc = np.clip(j, 0, len(sp) - 1)
        d = np.abs(rmid - sp[jc]); d = np.where(valid, d, np.inf)
        best = np.minimum(best, d)
    return int((best <= COOC_WIN).sum())

rows = []
for pt in sorted(os.listdir(SR_DIR)):
    sf = os.path.join(SR_DIR, f'{pt}_sr.npz')
    if not os.path.isfile(sf):
        continue
    zs = np.load(sf, allow_pickle=True)
    ch = list(zs['ch_names']); n = len(ch)
    ez = None
    if pt.startswith(("P", "c", "d", "h")):
        lf = os.path.join(LABEL_I, f'{pt}_labels.csv')
        if os.path.isfile(lf):
            dl = pd.read_csv(lf, encoding='utf-8-sig')
            em = dict(zip(dl['channel_name'], dl['EZ_NEZ']))
            ez = np.array([int(em.get(c, 0)) for c in ch])
    if ez is None:
        ez = np.asarray(zs['ez_clin'], int)
    grade = 1 if pt.startswith(("P", "c", "d", "h")) else 0
    cnt = {'spike': [], 'ripple': [], 'sr': []}
    for ci in range(n):
        cnt['spike'].append(len(np.asarray(zs['spike_pos'][ci], float)))
        cnt['ripple'].append(len(np.asarray(zs['ripple_start'][ci], float)))
        cnt['sr'].append(sr_count(np.asarray(zs['spike_pos'][ci], float),
                                  np.asarray(zs['ripple_start'][ci], float),
                                  np.asarray(zs['ripple_end'][ci], float)))
    frf = os.path.join(FR_ROOT, f'{pt}_fr.npz') if pt.startswith(('P','c','d','h')) else os.path.join(V2, f'{pt}_fr.npz')
    zf = np.load(frf, allow_pickle=True)
    cnt['fr'] = [len(np.asarray(zf['fr_start'][ci], float)) for ci in range(n)]
    cnt['wb'] = [len(np.asarray(zf['wbhfo_start'][ci], float)) for ci in range(n)]
    rec = {'patient': pt, 'grade': grade}
    for k, c in cnt.items():
        c = np.asarray(c, float)
        rec[k] = c[ez == 1].sum() / c.sum() if c.sum() > 0 else np.nan
    rows.append(rec)

d = pd.DataFrame(rows)
n3 = pd.read_csv(OUT + r'\_fig_new3_perpatient.csv')
d = d.merge(n3[['patient', 'swr_rf', 'rws_rf', 'sg_rf']], on='patient', how='left')
d.to_csv(OUT + r'\_fig7_resect_frac.csv', index=False)

MK = [('SR', 'sr'), ('Spike', 'spike'), ('Ripple', 'ripple'),
      ('Broadband\nHFO', 'wb'), ('FR', 'fr'),
      ('Spike w/o\nripple', 'swr_rf'), ('Ripple w/o\nspike', 'rws_rf'),
      ('Spike-gamma', 'sg_rf')]
FIGMED = {1: [.645, .609, .575, .619, .603, .615, .528, np.nan],
          0: [.394, .297, .289, .316, .345, np.nan, np.nan, np.nan]}

def stars(p):
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))

print('check (group median vs manuscript)vs vs SR paired Wilcoxon:')
for g in (1, 0):
    sub = d[d.grade == g]
    for (name, k), ref in zip(MK, FIGMED[g]):
        med = np.nanmedian(sub[k])
        note = ''
        if k != 'sr':
            common = sub[['patient', 'sr', k]].dropna()
            p = wilcoxon(common['sr'], common[k], zero_method='wilcox').pvalue
            note = f'  vs SR p={p:.4g} {stars(p)} (n={len(common)})'
        refstr = f'(manuscript {ref:.3f})' if np.isfinite(ref) else '(current cache)'
        print(f'  {"Engel I" if g else "II-IV  "} {name:<15} med={med:.3f} {refstr}{note}')

COL1, COL0 = '#C27B7B', '#3A79C0'
YB = 1.03
rng = np.random.default_rng(7)

def grad_bar(ax, x, w, h, color):
    rgb = to_rgb(color)
    light = tuple(min(1.0, c + (1.0 - c) * 0.55) for c in rgb)
    cmap = LinearSegmentedColormap.from_list('g', [light, rgb])
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, extent=[x - w / 2, x + w / 2, 0, h], aspect='auto',
              cmap=cmap, vmin=0, vmax=1, zorder=2, interpolation='bilinear', alpha=0.85)
    ax.add_patch(Rectangle((x - w / 2, 0), w, h, fill=False,
                           edgecolor='#666666', linewidth=1.6, zorder=3))

fig, ax = plt.subplots(figsize=(12.4, 6.0))
sub1, sub0 = d[d.grade == 1], d[d.grade == 0]
for i, (name, k) in enumerate(MK):
    for xoff, sub, col in [(-0.20, sub1, COL1), (+0.20, sub0, COL0)]:
        v = np.asarray(sub[k], float)
        v_ = v[~np.isnan(v)]
        w = 0.32
        x = i + xoff
        grad_bar(ax, x, w, np.median(v_), col)
        jit = rng.uniform(-w * 0.28, w * 0.28, len(v_))
        ax.scatter(x + jit, v_, s=15, color=col, alpha=0.5, edgecolor='none', zorder=4)

        if k != 'sr':
            common = sub[['patient', 'sr', k]].dropna()
            p = wilcoxon(common['sr'], common[k], zero_method='wilcox').pvalue
            ax.plot([x - w / 2 - 0.02, x - w / 2 - 0.02, x + w / 2 + 0.02, x + w / 2 + 0.02],
                    [YB - 0.012, YB, YB, YB - 0.012], color='black', lw=1.1, zorder=6)
            ax.text(x, YB + 0.008, stars(p), ha='center', va='bottom', fontsize=11.5,
                    fontweight='bold', zorder=6)

ax.set_ylim(0, 1.14)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlim(-0.65, 7.65)
ax.set_xticks(range(8))
ax.set_xticklabels([m[0] for m in MK], fontsize=12)
ax.set_ylabel('Fraction of events in resected region', fontsize=13)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2); ax.spines['bottom'].set_linewidth(2)
ax.tick_params(width=2, length=5, labelsize=11)
ax.legend(handles=[Patch(facecolor=COL1, edgecolor='#666666', label='Engel I (n=29)'),
                   Patch(facecolor=COL0, edgecolor='#666666', label='Engel II-IV (n=21)')],
          fontsize=12, loc='lower right', bbox_to_anchor=(1.0, 1.04), ncol=2, frameon=False)
fig.text(0.5, 0.015, 'Stars: each biomarker vs SR, paired Wilcoxon signed-rank test '
                     '(left/pink: Engel I; right/blue: Engel II-IV)',
         ha='center', fontsize=10.5, style='italic', color='#444444')
fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.13)
fig.savefig(OUT + r'\fig7_resect_frac.png', dpi=300)
fig.savefig(OUT + r'\fig7_resect_frac.pdf')
print('saved fig7_resect_frac')
