"""Shared figure style for the JMI manuscript: Arial-metric sans serif, 8-9 pt at print size,
thin lines, colour-blind-safe colours that also separate in greyscale (marker and line style differ)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TEXT_W = 6.3          # inches, the manuscript text width
DPI = 600

INK = '#1a1a1a'
GREY = '#6b6b6b'
LIGHT = '#d9d9d9'
# Okabe-Ito blue and vermillion
DV = dict(color='#0072B2', marker='o', ls='-', label='Dental ViT-L')
BC = dict(color='#D55E00', marker='s', ls='--', label='BiomedCLIP')
DENTEX_C = '#4d4d4d'
EXT_C = '#E69F00'


def setup():
    plt.rcParams.update({
        'font.family': 'Liberation Sans',
        'font.size': 8,
        'axes.titlesize': 8.5,
        'axes.labelsize': 8,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 7.5,
        'axes.linewidth': 0.6,
        'axes.edgecolor': INK,
        'axes.labelcolor': INK,
        'text.color': INK,
        'xtick.color': INK,
        'ytick.color': INK,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 1.1,
        'lines.markersize': 4,
        'legend.frameon': False,
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Liberation Sans',
        'mathtext.it': 'Liberation Sans:italic',
        'mathtext.bf': 'Liberation Sans:bold',
        'savefig.dpi': DPI,
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white',
    })


def panel_label(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')
