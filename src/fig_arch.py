"""Architecture schematic of the three switches (report/figures/ri_arch.pdf). No GPU, no data."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BP, CM, OR, GR = '#2a6fb0', '#d9822b', '#3a9a5b', '#555555'
fig, ax = plt.subplots(figsize=(7.6, 3.6))
ax.set_xlim(0, 15.6), ax.set_ylim(-0.3, 6.4), ax.axis('off')


def box(x, y, w, h, t, c=GR, fs=6.6, fill=True, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=c + '22' if fill else 'white', ec=c, lw=1.1, ls=ls))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, color='black')


def arrow(a, b, c=GR, ls='-'):
    ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '-|>', 'color': c, 'lw': 1.1, 'ls': ls, 'shrinkA': 0, 'shrinkB': 0})


y0 = 2.5  # main stream
for x, w, t in [(0.1, 1.4, 'tokens\n$x_t$'), (1.9, 2.2, 'embedding E\n+ position'), (4.5, 1.5, 'block 0'), (8.6, 1.5, 'block 1'), (10.6, 1.3, 'final\nLN')]:
    box(x, y0, w, 0.9, t)
for a, b in [(1.5, 1.9), (4.1, 4.5), (6.0, 6.9), (7.8, 8.6), (10.1, 10.6)]:
    arrow((a, y0 + .45), (b, y0 + .45))
ax.text(6.4, y0 + .45, '$h_t$', ha='center', va='center', fontsize=9)
ax.plot([6.4, 6.4], [y0 - .35, y0 + 1.35], color=GR, lw=0, marker='')
ax.text(6.4, y0 - 0.5, 'SAE fitted here', ha='center', fontsize=6.8, color=OR, style='italic')
box(12.4, y0, 3.0, 0.9, 'plain: $h\\,E^{\\top}$\nBackpack: $(\\sum\\alpha S)\\,E^{\\top}$', fs=6.6)
arrow((11.9, y0 + .45), (12.4, y0 + .45))
ax.text(13.9, y0 + 1.15, 'logits', ha='center', fontsize=7.5)

# concept path (CoCoMix)
box(4.0, 4.9, 2.4, 0.85, 'concept head\n$z_t = M h_t$ (512)', CM)
box(7.0, 4.9, 3.4, 0.85, 'top-4 of $z_t$, then compress\n$\\hat c_t = W\\,\\mathrm{TopK}(z_t)$', CM)
arrow((6.4, y0 + .9), (5.2, 4.9), CM)
arrow((6.4, 5.32), (7.0, 5.32), CM)
arrow((8.5, 4.9), (7.35, y0 + .6), CM)
ax.text(8.5, 3.95, '$\\hat c_t$ inserted\nnext to $h_t$', fontsize=6.8, color=CM, va='center')
ax.text(0.1, 5.6, 'CoCoMix', color=CM, fontsize=9, weight='bold')
ax.text(0.1, 5.2, 'label loss: $z_t$ against\n4 teacher concepts', color=CM, fontsize=6.6, va='top')

# Backpack path
box(2.1, 0.0, 3.6, 0.9, 'sense network, token only:\n$S_1(x_t),\\dots,S_4(x_t)$', BP)
arrow((0.85, y0), (2.1, 0.55), BP)
box(8.2, 0.0, 3.6, 0.9, 'weights $\\alpha_{\\ell tj}$: one attention head\nper sense, on the final states', BP)
arrow((11.25, y0), (10.5, 0.9), BP)
ax.add_patch(plt.Circle((12.75, 0.45), 0.45, fc=BP + '22', ec=BP, lw=1.1))
ax.text(12.75, 0.45, '$\\sum\\alpha S$', ha='center', va='center', fontsize=8)
arrow((5.7, 0.45), (8.2, 0.45), BP)
ax.text(6.95, 0.62, '$S_\\ell(x_j)$,\nall $j\\leq t$', ha='center', fontsize=6.3, color=BP)
arrow((11.8, 0.45), (12.3, 0.45), BP)
arrow((12.75, 0.9), (13.4, y0), BP)
ax.text(0.1, 0.1, 'Backpack', color=BP, fontsize=9, weight='bold')

# penalty
for (x, y, w, h) in [(2.0, -0.1, 3.8, 1.1), (6.9, 4.8, 3.6, 1.05)]:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.14', fc='none', ec=OR, lw=1.3, ls=(0, (3, 2))))
ax.text(5.0, 6.15, 'dashed green: penalty $\\Omega$ pushes the senses of a word, and the columns of $W$, apart', color=OR, fontsize=6.6, ha='left')
fig.savefig('../report/figures/ri_arch.pdf', bbox_inches='tight')
