import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

C_FWD   = '#3A79C0'
C_REV   = '#C27B7B'
C_LINE  = '#B3B3B3'
C_MED   = '#4675B1'
C_DARK  = '#595757'

path = r'<PROJECT_DIR>\part10\part10_mechanism.csv'
rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
ok = [r for r in rows
      if r['A_exc_out'] not in ('',) and r['A_exc_dn'] not in ('',)
      and r['n_ev_out'] not in ('', '0')]
rev = np.array([float(r['A_exc_dn']) for r in ok])
fwd = np.array([float(r['A_exc_out']) for r in ok])

assert len(ok) == 49, len(ok)
assert int((fwd > rev).sum()) == 47
assert abs(np.median(fwd) - 0.465) < 5e-4, np.median(fwd)
assert abs(np.median(rev) - 0.076) < 5e-4, np.median(rev)
print('n=%d  fwd>rev: %d/49  med_fwd=%.3f  med_rev=%.3f'
      % (len(ok), int((fwd > rev).sum()), np.median(fwd), np.median(rev)))

fig, ax = plt.subplots(figsize=(5.2, 5.2))

X0, X1 = 0.0, 1.0

for r, f in zip(rev, fwd):
    ax.plot([X0, X1], [r, f], color=C_LINE, lw=0.7, alpha=0.75,
            zorder=1, solid_capstyle='round')

ax.axhline(0, color=C_DARK, lw=1.0, ls=(0, (4, 3)), zorder=1.5)

ax.plot([X0], [np.median(rev)], marker='D', ms=8, color=C_MED, zorder=5)
ax.plot([X1], [np.median(fwd)], marker='D', ms=8, color=C_MED, zorder=5)

ax.text(X0 - 0.06, np.median(rev), 'Median +0.076', ha='right', va='center',
        fontsize=12, color=C_MED)
ax.text(X1 + 0.06, np.median(fwd), 'Median +0.465', ha='left', va='center',
        fontsize=12, color=C_MED)

ax.text(0.985, 0.965, '47/49 patients: forward > reverse',
        transform=ax.transAxes, ha='right', va='top', fontsize=12, color=C_DARK)

ax.set_xlim(-0.72, 1.72)
ax.set_ylim(-0.06, 0.95)
ax.set_xticks([X0, X1])
ax.set_xticklabels(['Reverse\n(dense follows outside)',
                    'Forward\n(outside follows dense)'], fontsize=11)
ax.set_ylabel('Excess following rate', fontsize=12)
ax.tick_params(axis='y', labelsize=11, width=3, length=6)
ax.tick_params(axis='x', width=3, length=6)
for s in ('left', 'bottom'):
    ax.spines[s].set_linewidth(3)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

fig.tight_layout()

def gradient_ball(color, n=72, gamma=1.2):
    
    rgb = tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - c, yy - c) / c
    a = np.clip(1.0 - r * r, 0.0, 1.0) ** gamma
    rgba = np.zeros((n, n, 4), float)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = rgb
    rgba[..., 3] = a
    return rgba

R_PT = np.sqrt(22) / 2.0
fig.canvas.draw()
inv = ax.transData.inverted()
p0 = inv.transform((0, 0))
p1 = inv.transform((fig.dpi / 72.0, fig.dpi / 72.0))
rx = R_PT * abs(p1[0] - p0[0])
ry = R_PT * abs(p1[1] - p0[1])
ax.set_autoscale_on(False)
for xs_, ys_, col, z in [(np.full_like(rev, X0), rev, C_REV, 2),
                         (np.full_like(fwd, X1), fwd, C_FWD, 3)]:
    ball = gradient_ball(col)
    for x_, y_ in zip(xs_, ys_):
        ax.imshow(ball, extent=(x_ - rx, x_ + rx, y_ - ry, y_ + ry),
                  interpolation='bilinear', origin='lower', zorder=z,
                  aspect='auto')

ax.scatter(np.full_like(rev, X0), rev, s=22, facecolors='none',
           edgecolors='black', linewidths=0.25, zorder=4)
ax.scatter(np.full_like(fwd, X1), fwd, s=22, facecolors='none',
           edgecolors='black', linewidths=0.25, zorder=4)
ax.set_xlim(-0.72, 1.72)
ax.set_ylim(-0.06, 0.95)

out = r'<OUTPUT_DIR>\fig4c_time_asym'
fig.savefig(out + '.png', dpi=600)
fig.savefig(out + '.pdf')
print('saved:', out + '.png / .pdf')
