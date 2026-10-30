"""Figs. 6 and 7 of the manuscript: crop the black frame of each radiograph panel and stack (a) above (b) with plain panel labels."""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
BOLD = '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'


def content_box(imgs, thr=40, margin=8):
    boxes = []
    for im in imgs:
        g = np.asarray(im.convert('RGB')).astype(int).max(axis=2)
        ys, xs = np.where(g > thr)
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    x0 = max(0, min(b[0] for b in boxes) - margin); y0 = max(0, min(b[1] for b in boxes) - margin)
    x1 = min(imgs[0].width, max(b[2] for b in boxes) + margin + 1); y1 = min(imgs[0].height, max(b[3] for b in boxes) + margin + 1)
    return x0, y0, x1, y1


def stack(imgs, out, label_px=34, gap=26):
    box = content_box(imgs)
    crops = [im.convert('RGB').crop(box) for im in imgs]
    f = ImageFont.truetype(BOLD, label_px)
    lab_h = int(label_px * 1.45)
    W = crops[0].width
    H = sum(lab_h + c.height for c in crops) + gap * (len(crops) - 1)
    canvas = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(canvas)
    y = 0
    for lab, c in zip(['(a)', '(b)'], crops):
        d.text((0, y), lab, fill=(26, 26, 26), font=f)
        y += lab_h
        canvas.paste(c, (0, y)); y += c.height + gap
    canvas.save(out, dpi=(300, 300))
    print(out, canvas.size, 'crop box', box)


if __name__ == '__main__':
    # usage: compose.py <fig6_raw.png> <fig6_enveloped.png> <fig7_two_panel.png>
    # Fig. 6 panels come from notebook 09; the two-panel Fig. 7 image comes from notebook 07 (panels stacked, 976 px each)
    stack([Image.open(sys.argv[1]), Image.open(sys.argv[2])], 'fig6.png')
    f7 = Image.open(sys.argv[3]).convert('RGB')
    stack([f7.crop((15, 60, 1991, 1036)), f7.crop((15, 1096, 1991, 2072))], 'fig7.png')
