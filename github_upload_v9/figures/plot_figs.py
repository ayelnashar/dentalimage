"""Figs. 2 and 5 drawn from results/figures_data.json, written by notebook 10. Usage: plot_figs.py <json> <outdir>"""
import json, sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import setup, TEXT_W, INK, GREY, LIGHT, DV, BC, DENTEX_C, EXT_C, panel_label
setup()
D = json.load(open(sys.argv[1])); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
NAMES = {'dental_vitl': 'Dental ViT-L', 'biomedclip': 'BiomedCLIP'}
STY = {'dental_vitl': DV, 'biomedclip': BC}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def ece(y, p, nb=10):
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for lo, hi in zip(b[:-1], b[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum():
            e += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return e


# ---------------------------------------------------------------- Fig. 2 reliability diagrams with bin counts
def fig2():
    y = np.array(D['fig2']['y'], float); p = np.array(D['fig2']['p'], float); cls = D['classes']
    fig = plt.figure(figsize=(TEXT_W, 2.75))
    gs = fig.add_gridspec(2, 4, height_ratios=[3.1, 1], hspace=0.12, wspace=0.36, left=0.075, right=0.975, top=0.86,
                          bottom=0.15)
    edges = np.linspace(0, 1, 11)
    for c in range(4):
        ax = fig.add_subplot(gs[0, c]); axh = fig.add_subplot(gs[1, c], sharex=ax)
        yc, pc = y[:, c], p[:, c]
        xs, ob, lo_, hi_, n_ = [], [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pc > lo) & (pc <= hi); n = int(m.sum()); n_.append(n)
            if n == 0:
                continue
            k = int(yc[m].sum()); l, h = wilson(k, n)
            xs.append(pc[m].mean()); ob.append(k / n); lo_.append(k / n - l); hi_.append(h - k / n)
        ax.plot([0, 1], [0, 1], ls=':', color=GREY, lw=0.8)
        ax.errorbar(xs, ob, yerr=[lo_, hi_], fmt='o-', color=DV['color'], ms=3.2, lw=1.0, elinewidth=0.7, capsize=1.6)
        ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02); ax.set_yticks([0, 0.5, 1])
        ax.set_title(f'{cls[c]}\nECE {ece(yc, pc):.3f}', fontsize=8, linespacing=1.3)
        plt.setp(ax.get_xticklabels(), visible=False)
        if c == 0:
            ax.set_ylabel('Observed frequency')
        axh.bar(edges[:-1], n_, width=0.1, align='edge', color=LIGHT, edgecolor=GREY, lw=0.5)
        axh.set_yscale('log'); axh.set_ylim(0.8, 1000); axh.set_yticks([1, 10, 100]); axh.set_yticklabels(['1', '10', '100'])
        axh.set_xticks([0, 0.5, 1]); axh.set_xlabel('Predicted probability')
        if c == 0:
            axh.set_ylabel('Teeth')
    fig.savefig(f'{OUT}/fig2.png'); plt.close(fig)


# ---------------------------------------------------------------- Fig. 3 delta sweep
def fig3():
    sw = D['fig3']['delta_sweep']
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.35))
    for j, (key, lab) in enumerate([('auc', 'Macro-AUC'), ('ece', 'Macro-ECE')]):
        ax = axes[j]
        for n in ('dental_vitl', 'biomedclip'):
            ds = sorted(sw[n], key=float); x = [float(d) for d in ds]
            ax.plot(x, [sw[n][d][f'env_{key}'] for d in ds], color=STY[n]['color'], marker=STY[n]['marker'],
                    ls=STY[n]['ls'], label=NAMES[n])
            ax.axhline(sw[n][ds[0]][f'base_{key}'], color=STY[n]['color'], ls=':', lw=0.9)
        ax.axvline(1.5, color=LIGHT, lw=0.8, zorder=0)
        ax.set_xlim(0.1, 3.15); ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        ax.set_xlabel('Clipping bound $\\delta$'); ax.set_ylabel(lab)
        panel_label(ax, '(a)' if j == 0 else '(b)', x=-0.2)
    h = [Line2D([], [], **{k: STY[n][k] for k in ('color', 'marker', 'ls')}) for n in ('dental_vitl', 'biomedclip')]
    h.append(Line2D([], [], color=GREY, ls=':', lw=0.9))
    fig.legend(h, ['Dental ViT-L, bounded update', 'BiomedCLIP, bounded update', 'Reference probe (same colour)'],
               loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.8, bottom=0.18, wspace=0.32)
    fig.savefig(f'{OUT}/fig3.png'); plt.close(fig)


# ---------------------------------------------------------------- Fig. 4 retention curves
def fig4():
    cv = D['fig4']['curves']
    fig, ax = plt.subplots(figsize=(4.3, 2.55))
    for n in ('dental_vitl', 'biomedclip'):
        ks = sorted(cv[n], key=float)
        ax.plot([float(k) for k in ks], [cv[n][k] for k in ks], color=STY[n]['color'], marker=STY[n]['marker'],
                ls=STY[n]['ls'], label=NAMES[n], ms=3.5)
    ax.set_xlim(0.27, 1.03); ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel('Fraction of DENTEX test crops retained'); ax.set_ylabel('Element-wise accuracy on retained crops')
    ax.legend(loc='upper right')
    fig.subplots_adjust(left=0.17, right=0.98, top=0.96, bottom=0.17)
    fig.savefig(f'{OUT}/fig4.png'); plt.close(fig)


# ---------------------------------------------------------------- Fig. 5 distance distributions
def fig5():
    f5 = D['fig5']
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.45))
    for j, n in enumerate(('dental_vitl', 'biomedclip')):
        ax = axes[j]; r = f5[n]
        din, dout = np.array(r['dentex_test']), np.array(r['tufts'])
        lo, hi = np.percentile(np.concatenate([din, dout]), [1, 99]); bins = np.linspace(lo, hi, 40)
        ax.hist(dout, bins=bins, density=True, color=EXT_C, alpha=0.55, lw=0)
        ax.hist(din, bins=bins, density=True, color=DENTEX_C, alpha=0.55, lw=0)
        ax.hist(dout, bins=bins, density=True, histtype='step', color=EXT_C, lw=0.8)
        ax.hist(din, bins=bins, density=True, histtype='step', color=DENTEX_C, lw=0.8)
        ax.axvline(r['threshold_val_p95'], color=INK, lw=1.0)
        ax.axvline(r['threshold_test_p95'], color=INK, lw=1.0, ls='--')
        ax.set_xlabel('Mahalanobis distance'); ax.set_ylabel('Density' if j == 0 else '')
        ax.set_title(NAMES[n], fontsize=8.5)
        panel_label(ax, '(a)' if j == 0 else '(b)', x=-0.16)
    h = [plt.Rectangle((0, 0), 1, 1, color=DENTEX_C, alpha=0.55), plt.Rectangle((0, 0), 1, 1, color=EXT_C, alpha=0.55),
         Line2D([], [], color=INK, lw=1.0), Line2D([], [], color=INK, lw=1.0, ls='--')]
    fig.legend(h, ['DENTEX test crops', 'Tufts crops', 'Source-validation threshold', '95% DENTEX test retention'],
               loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.8, bottom=0.17, wspace=0.22)
    fig.savefig(f'{OUT}/fig5.png'); plt.close(fig)


for f in (fig2, fig5):
    f(); print('saved', f.__name__)
