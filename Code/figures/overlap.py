import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

plt.rcParams['font.family'] = ['Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

OUT = r'<OUTPUT_DIR>'

C_DENSE  = '#F4D35E'
C_SRC    = '#2E8B57'
C_OVER   = '#8DB87C'
C_GOOD   = '#C27B7B'
C_POOR   = '#3A79C0'
EC_DENSE = '#8B7D1E'
EC_SRC   = '#1B5E20'
C_TEXT   = '#2a2a2a'

def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 6.0), dpi=300,
                                   gridspec_kw={'width_ratios': [1.45, 1.0],
                                                'wspace': 0.02})

    axA.set_xlim(-3.2, 3.2)
    axA.set_ylim(-2.2, 2.5)
    axA.set_aspect('equal')
    axA.axis('off')

    rx, ry = 1.45, 1.20
    sep = 1.35
    cx1, cx2 = -sep, +sep

    axA.add_patch(Ellipse((cx1, 0), 2*rx, 2*ry,
                           fc=C_DENSE, ec='none', alpha=0.40, zorder=1))
    axA.add_patch(Ellipse((cx2, 0), 2*rx, 2*ry,
                           fc=C_SRC, ec='none', alpha=0.40, zorder=1))

    axA.add_patch(Ellipse((cx1, 0), 2*rx, 2*ry,
                           fc='none', ec=EC_DENSE, lw=2.0, zorder=4))
    axA.add_patch(Ellipse((cx2, 0), 2*rx, 2*ry,
                           fc='none', ec=EC_SRC, lw=2.0, zorder=4))

    axA.text(0, 0.15, '82.1%', ha='center', va='center',
             fontsize=20, color=EC_SRC, fontweight='bold', zorder=5)
    axA.text(0, -0.20, 'source ∩ dense', ha='center', va='center',
             fontsize=11, color='#2E5E3E', zorder=5)

    axA.text(cx1 - 0.55, -0.25, '17.9%', ha='center', va='center',
             fontsize=13, color=EC_DENSE, fontweight='bold', zorder=5)
    axA.text(cx1 - 0.55, -0.55, 'non-source\nin dense', ha='center', va='center',
             fontsize=9, color=EC_DENSE, zorder=5)

    axA.text(cx2 + 0.50, 0.25, '9.2%', ha='center', va='center',
             fontsize=13, color=EC_SRC, fontweight='bold', zorder=5)
    axA.text(cx2 + 0.50, -0.05, 'source outside\ndense zone', ha='center', va='center',
             fontsize=9, color=EC_SRC, zorder=5)

    axA.text(cx1, ry + 0.12, 'SR Dense Zone', ha='center', va='bottom',
             fontsize=14, color=EC_DENSE, fontweight='bold', zorder=5)
    axA.text(cx2, ry + 0.12, 'Source Channels', ha='center', va='bottom',
             fontsize=14, color=EC_SRC, fontweight='bold', zorder=5)

    axA.annotate('75.0% of source channels\nlie within dense zone',
                 xy=(0.0, -0.45), xytext=(-2.65, -1.35),
                 fontsize=11, color='#2E5E3E', fontweight='bold',
                 ha='center', va='center',
                 arrowprops=dict(arrowstyle='->', color='#2E5E3E',
                                 lw=1.8, connectionstyle='arc3,rad=0.20'),
                 zorder=5)

    axA.text(0, -ry - 0.28, 'Jaccard = 0.588 ± 0.213',
             ha='center', va='top', fontsize=14, color=C_TEXT,
             fontweight='bold', zorder=5)

    axA.text(0, -ry - 0.60,
             '$P = 7.1 \\times 10^{-15}$  (Wilcoxon signed-rank, patient level)',
             ha='center', va='top', fontsize=10, color='#555555',
             style='italic', zorder=5)

    axA.text(-3.05, 2.35, 'A', fontsize=20, fontweight='bold',
             color=C_TEXT, va='top', zorder=5)

    axB.set_xlim(-2.2, 2.2)
    axB.set_ylim(-2.2, 2.5)
    axB.set_aspect('equal')
    axB.axis('off')

    mr, mry = 0.72, 0.55
    msep = 0.45

    my1 = 0.85
    mx1 = 0.0
    axB.add_patch(Ellipse((mx1 - msep, my1), 2*mr, 2*mry,
                           fc=C_DENSE, ec='none', alpha=0.35, zorder=1))
    axB.add_patch(Ellipse((mx1 + msep, my1), 2*mr, 2*mry,
                           fc=C_GOOD, ec='none', alpha=0.35, zorder=1))
    axB.add_patch(Ellipse((mx1 - msep, my1), 2*mr, 2*mry,
                           fc='none', ec=EC_DENSE, lw=1.2, zorder=3))
    axB.add_patch(Ellipse((mx1 + msep, my1), 2*mr, 2*mry,
                           fc='none', ec='#8B4F4F', lw=1.2, zorder=3))
    axB.text(mx1, my1, '76.7%', ha='center', va='center',
             fontsize=13, fontweight='bold', color=C_GOOD, zorder=5)
    axB.text(mx1 - msep, my1 - mry - 0.08, 'Dense', ha='center', va='top',
             fontsize=8, color=EC_DENSE, zorder=5)
    axB.text(mx1 + msep, my1 - mry - 0.08, 'Source', ha='center', va='top',
             fontsize=8, color='#8B4F4F', zorder=5)
    axB.text(mx1, my1 + mry + 0.10, 'Engel I  (n = 29)', ha='center', va='bottom',
             fontsize=11, color=C_GOOD, fontweight='bold', zorder=5)
    axB.text(mx1, my1 - mry - 0.35, 'Jaccard = 0.592 ± 0.222',
             ha='center', va='top', fontsize=9.5, color=C_GOOD, zorder=5)

    my2 = -0.75
    mx2 = 0.0
    axB.add_patch(Ellipse((mx2 - msep, my2), 2*mr, 2*mry,
                           fc=C_DENSE, ec='none', alpha=0.35, zorder=1))
    axB.add_patch(Ellipse((mx2 + msep, my2), 2*mr, 2*mry,
                           fc=C_POOR, ec='none', alpha=0.35, zorder=1))
    axB.add_patch(Ellipse((mx2 - msep, my2), 2*mr, 2*mry,
                           fc='none', ec=EC_DENSE, lw=1.2, zorder=3))
    axB.add_patch(Ellipse((mx2 + msep, my2), 2*mr, 2*mry,
                           fc='none', ec='#2244AA', lw=1.2, zorder=3))
    axB.text(mx2, my2, '67.6%', ha='center', va='center',
             fontsize=13, fontweight='bold', color=C_POOR, zorder=5)
    axB.text(mx2 - msep, my2 - mry - 0.08, 'Dense', ha='center', va='top',
             fontsize=8, color=EC_DENSE, zorder=5)
    axB.text(mx2 + msep, my2 - mry - 0.08, 'Source', ha='center', va='top',
             fontsize=8, color='#2244AA', zorder=5)
    axB.text(mx2, my2 + mry + 0.10, 'Engel II\u2013IV  (n = 21)', ha='center', va='bottom',
             fontsize=11, color=C_POOR, fontweight='bold', zorder=5)
    axB.text(mx2, my2 - mry - 0.35, 'Jaccard = 0.583 ± 0.205',
             ha='center', va='top', fontsize=9.5, color=C_POOR, zorder=5)

    axB.text(0, -2.05,
             'Jaccard:  $P = 0.74$\n'
             'Dense-source %:  $P = 0.23$\n'
             'No significant difference between groups',
             ha='center', va='top', fontsize=10, color='#555555',
             style='italic', zorder=5)

    axB.text(-2.05, 2.35, 'B', fontsize=20, fontweight='bold',
             color=C_TEXT, va='top', zorder=5)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)

    png = os.path.join(OUT, 'fig9_mechanism_overlap.png')
    fig.savefig(png, dpi=300)
    fig.savefig(png.replace('.png', '.pdf'))
    plt.close(fig)
    print('saved %s' % png)

if __name__ == '__main__':
    main()
