"""Fig. 1: schematic of the frozen-encoder prediction layer. Usage: fig1.py <out.png>"""
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from figstyle import setup, INK, GREY, TEXT_W
setup()
OUT = sys.argv[1]

W, H = TEXT_W, 2.95
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
LW = 0.7
STEP = 0.175


def box(x0, yc, w, h, title, lines, fill='white', italic_last=False):
    ax.add_patch(Rectangle((x0, yc - h / 2), w, h, facecolor=fill, edgecolor=INK, lw=LW))
    n = len(lines) + 1
    top = yc + (n - 1) * STEP / 2
    ax.text(x0 + w / 2, top, title, ha='center', va='center', fontsize=8.5, fontweight='bold')
    for i, s in enumerate(lines):
        last = italic_last and i == len(lines) - 1
        ax.text(x0 + w / 2, top - (i + 1) * STEP, s, ha='center', va='center', fontsize=7.5 if last else 8,
                style='italic' if last else 'normal', color=GREY if last else INK)


def arrow(p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=7, lw=LW, color=INK, shrinkA=0, shrinkB=0))


def line(pts):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=INK, lw=LW, solid_capstyle='butt')


# column 1: frozen encoder
ex0, ew, eh, ey = 0.04, 1.36, 0.9, 1.47
box(ex0, ey, ew, eh, 'Frozen encoder', ['Input: tooth crop', 'Output: embedding $\\mathbf{x}$', 'Weights unchanged'],
    fill='#eeeeee')
# column 2: probe, residual head, distance
cx0, cw, ch = 1.84, 1.80, 0.64
ys = {'probe': 2.45, 'res': 1.47, 'maha': 0.5}
box(cx0, ys['probe'], cw, ch, 'Reference probe',
    ['$b_k(\\mathbf{x}) = z_k(\\mathbf{x})\\,/\\,T$', 'Logistic probe, temperature $T$'])
box(cx0, ys['res'], cw, ch, 'Residual head',
    ['$c_k = \\mathrm{clip}(r_k(\\mathbf{x}),\\,{-\\delta},\\,\\delta)$', 'Shallow trained head'])
box(cx0, ys['maha'], cw, ch, 'Mahalanobis distance',
    ['$m(\\mathbf{x})$, regularised covariance', 'Fitted on source training data'])
xs = 1.62
line([(ex0 + ew, ey), (xs, ey)])
line([(xs, ys['maha']), (xs, ys['probe'])])
for k in ys:
    arrow((xs, ys[k]), (cx0, ys[k]))
ax.text((ex0 + ew + xs) / 2, ey + 0.04, '$\\mathbf{x}$', fontsize=8.5, ha='center', va='bottom')
# sum node
ox0, ow, oh, oy = 4.3, 1.96, 1.1, 2.0
sx, sy, r = 3.97, oy, 0.11
ax.add_patch(Circle((sx, sy), r, facecolor='white', edgecolor=INK, lw=LW))
ax.plot([sx - 0.055, sx + 0.055], [sy, sy], color=INK, lw=LW)
ax.plot([sx, sx], [sy - 0.055, sy + 0.055], color=INK, lw=LW)
line([(cx0 + cw, ys['probe']), (sx, ys['probe'])]); arrow((sx, ys['probe']), (sx, sy + r))
line([(cx0 + cw, ys['res']), (sx, ys['res'])]); arrow((sx, ys['res']), (sx, sy - r))
ax.text(sx - 0.06, ys['probe'] + 0.035, '$b_k$', fontsize=8, ha='right', va='bottom')
ax.text(sx - 0.06, ys['res'] + 0.035, '$c_k$', fontsize=8, ha='right', va='bottom')
# column 3: bounded output and gate
box(ox0, oy, ow, oh, 'Bounded output',
    ['$f_k = b_k + c_k$', '$|f_k - b_k| \\leq \\delta$', '$p_k = \\sigma(f_k)$', 'Calibration is empirical'],
    italic_last=True)
arrow((sx + r, sy), (ox0, oy))
gh, gy = 0.86, 0.5
box(ox0, gy, ow, gh, 'Gate',
    ['$m(\\mathbf{x}) \\leq \\tau$: display $p_k$', '$m(\\mathbf{x}) > \\tau$: defer to the clinician',
     '$\\tau$: 95th percentile, source validation'])
arrow((cx0 + cw, ys['maha']), (ox0, gy))
gx = ox0 + ow / 2
arrow((gx, oy - oh / 2), (gx, gy + gh / 2))
ax.text(gx + 0.06, (oy - oh / 2 + gy + gh / 2) / 2, '$p_k$', fontsize=8, ha='left', va='center')
fig.savefig(OUT, dpi=600)
print('saved', OUT)
