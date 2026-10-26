"""Receiver and observer accuracy by k, rendered from saved verified summaries."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/receiver-observer-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

output = Path(__file__).resolve().parent
data = json.loads((output / 'receiver_observer_by_k_data.json').read_text())
colors = {0:'#555555', 1:'#e69f00', 2:'#0072b2', 4:'#009e73', 8:'#cc79a7'}
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.titlesize':13,
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'svg.hashsalt':'receiver-observer-by-k', 'savefig.facecolor':'white'})
fig, axes = plt.subplots(1, 5, figsize=(15, 3.8), sharex=True, sharey=True)
for ax, (k, color) in zip(axes, colors.items()):
    rows = data[str(k)]['rows']
    assert [r['cycle'] for r in rows] == list(range(0, 201, 10))
    assert all(r['k'] == k and r['university_n'] == 2000 for r in rows)
    for r in rows:
        assert abs(r['C_R'] - r['counts']['receiver_correct']/2000) < 1e-10
        assert abs(r['C_O'] - r['counts']['observer_correct']/2000) < 1e-10
    x = [r['cycle'] for r in rows]
    ax.axhline(50, color='#adb5bd', linestyle=':', linewidth=1, zorder=0)
    ax.plot(x, [100*r['C_R'] for r in rows], color=color, linewidth=2.1,
            marker='o', markersize=3.4, zorder=3)
    ax.plot(x, [100*r['C_O'] for r in rows], color=color, linewidth=1.8,
            linestyle='--', marker='s', markersize=3.7, markerfacecolor='white',
            markeredgewidth=1.1, zorder=4)
    ax.set_title(f'k = {k}', color=color, fontweight='bold', pad=10)
    ax.set_xlim(-3, 203); ax.set_ylim(45, 102)
    ax.set_xticks([0, 50, 100, 150, 200]); ax.set_yticks([50, 60, 70, 80, 90, 100])
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    ax.spines['bottom'].set_color('#b5bbc3'); ax.spines['left'].set_color('#b5bbc3')
    ax.tick_params(labelsize=9)

axes[0].set_ylabel('Accuracy (%)', labelpad=8)
fig.suptitle('Receiver and Observer Accuracy Across Training', fontsize=17, fontweight='semibold', y=.98)
handles = [Line2D([], [], color='#333333', lw=2.1, marker='o', markersize=4, label='Receiver'),
           Line2D([], [], color='#333333', lw=1.8, ls='--', marker='s', markerfacecolor='white', markersize=4, label='Observer')]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.90), ncol=2, frameon=False)
fig.supxlabel('Training cycles', y=.10, fontsize=11)
fig.text(.5,.025,'Same colors as the sweep figure · 2,000 university examples per evaluation · k = 0: frozen observer',
         ha='center', fontsize=9, color='#596273')
fig.subplots_adjust(left=.055, right=.99, bottom=.23, top=.72, wspace=.15)
for ext in ['png', 'svg']:
    fig.savefig(output / f'receiver_observer_by_k.{ext}', dpi=180)
plt.close(fig)
print(output/'receiver_observer_by_k.png')
