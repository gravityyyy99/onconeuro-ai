"""Makes every picture used by README.md into docs/img/.  Run from the project root:

    python tools/make_readme_graphics.py            # SVGs only (fast)
    python tools/make_readme_graphics.py --gifs     # also the two rotating 3D GIFs (needs PyTorch, grad-cam, matplotlib)

Numbers come from evaluation.json, so the charts always match the real results.
"""
import json
import math
import os
import sys
from xml.sax.saxutils import escape

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "docs", "img")
os.makedirs(OUT, exist_ok=True)
EV = json.load(open(os.path.join(ROOT, "evaluation.json")))
LABELS = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumor", "pituitary": "Pituitary"}

# self-contained dark cards so they look right on GitHub's light AND dark themes
BG, CARD, LINE = "#1f1e1d", "#2a2927", "#3d3c39"
INK, BODY, MUTED = "#f5f4ee", "#d6d4cb", "#a3a198"
ACC, ACC2, SAGE = "#d97757", "#e8a58c", "#9bb27a"
SERIF = "Georgia, 'Times New Roman', serif"
SANS = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
C30, S30 = math.cos(math.radians(30)), math.sin(math.radians(30))


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{escape(title)}"><rect width="{w}" height="{h}" rx="20" fill="{BG}"/>{body}</svg>')


def text(x, y, s, size=14, fill=INK, weight=400, anchor="start", family=SANS, extra=""):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}" '
            f'font-family="{family}" {extra}>{escape(str(s))}</text>')


def save(name, content):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(content)
    print("wrote", name, len(content) // 1024, "KB")


def iso_fn(u, ox, oy):
    def iso(x, y, z):
        return ox + (x - y) * C30 * u, oy + (x + y) * S30 * u - z
    return iso


def poly(iso, pts, fill, extra=""):
    p = " ".join(f"{iso(*q)[0]:.1f},{iso(*q)[1]:.1f}" for q in pts)
    return f'<polygon points="{p}" fill="{fill}" stroke="#000" stroke-opacity=".25" stroke-width="1" {extra}/>'


def iso_box(iso, x, y, z, dx, dy, dz, top, left, right, extra=""):
    """A 3D block. `left` is the face towards +y, `right` the face towards +x."""
    return (poly(iso, [(x, y + dy, z), (x + dx, y + dy, z), (x + dx, y + dy, z + dz), (x, y + dy, z + dz)], left, extra)
            + poly(iso, [(x + dx, y, z), (x + dx, y + dy, z), (x + dx, y + dy, z + dz), (x + dx, y, z + dz)], right, extra)
            + poly(iso, [(x, y, z + dz), (x + dx, y, z + dz), (x + dx, y + dy, z + dz), (x, y + dy, z + dz)], top, extra))


# ------------------------------------------------------------------ banner
def banner():
    rng = np.random.default_rng(7)
    layers = [4, 6, 6, 4]
    xs = [770, 880, 990, 1100]
    nodes = []
    for L, x in zip(layers, xs):
        gap = 230 / (L - 1)
        nodes.append([(x, 55 + i * gap) for i in range(L)])
    b = ""
    for a, c in zip(nodes[:-1], nodes[1:]):
        for (x1, y1) in a:
            for (x2, y2) in c:
                b += f'<line x1="{x1}" y1="{y1:.0f}" x2="{x2}" y2="{y2:.0f}" stroke="{LINE}" stroke-width="1"/>'
    k = 0
    for li, layer in enumerate(nodes):
        for (x, y) in layer:
            d = (li * 0.5 + (k % 3) * 0.35)
            b += (f'<circle cx="{x}" cy="{y:.0f}" r="9" fill="{ACC}"><animate attributeName="opacity" values="0.25;1;0.25" '
                  f'dur="3s" begin="{d:.2f}s" repeatCount="indefinite"/><animate attributeName="r" values="7;10;7" dur="3s" '
                  f'begin="{d:.2f}s" repeatCount="indefinite"/></circle>')
            k += 1
    # scan frame with scanning line
    b += f'<rect x="60" y="150" width="0" height="0"/>'
    b += text(60, 130, "OncoNeuro AI", 66, INK, 500, family=SERIF)
    b += text(62, 175, "A brain MRI classifier that shows its work", 24, ACC2)
    b += text(62, 215, "ResNet-18 transfer learning  +  Grad-CAM heatmaps  +  a multi-page Streamlit site", 16, MUTED)
    chips = [("Python", 62), ("PyTorch", 160), ("Grad-CAM", 262), ("Streamlit", 370)]
    for name, x in chips:
        w = 18 + 9 * len(name)
        b += f'<rect x="{x}" y="240" width="{w}" height="30" rx="15" fill="{CARD}" stroke="{LINE}"/>'
        b += text(x + w / 2, 260, name, 13, BODY, 600, "middle")
    b += (f'<rect x="62" y="288" width="372" height="30" rx="15" fill="none" stroke="{ACC}" stroke-width="1.5"/>'
          + text(248, 308, "Learning project  -  not a medical tool", 13, ACC, 600, "middle"))
    b += (f'<rect x="0" y="0" width="1200" height="4" fill="{ACC}"><animate attributeName="width" values="0;1200;0" dur="6s" '
          f'repeatCount="indefinite"/></rect>')
    save("banner.svg", svg(1200, 340, b, "OncoNeuro AI banner with an animated neural network"))


# ------------------------------------------------------------------ 3D pipeline
def pipeline():
    u, ox, oy = 118, 60, 330

    def ob(x, y, z):          # oblique 3D: x right, y into the screen, z up
        return ox + x * u + y * u * 0.5, oy - z - y * u * 0.34

    def box3(x, dx, dy, dz, top, front, side):
        P = lambda q: " ".join(f"{ob(*c)[0]:.1f},{ob(*c)[1]:.1f}" for c in q)
        st = 'stroke="#000" stroke-opacity=".3" stroke-width="1"'
        return (f'<polygon points="{P([(x + dx, 0, 0), (x + dx, dy, 0), (x + dx, dy, dz), (x + dx, 0, dz)])}" fill="{side}" {st}/>'
                f'<polygon points="{P([(x, dy, dz), (x + dx, dy, dz), (x + dx, 0, dz), (x, 0, dz)])}" fill="{top}" {st}/>'
                f'<polygon points="{P([(x, 0, 0), (x + dx, 0, 0), (x + dx, 0, dz), (x, 0, dz)])}" fill="{front}" {st}/>')

    b = text(40, 48, "The journey of one scan, in 3D", 28, INK, 500, family=SERIF)
    b += text(40, 74, "Taller blocks hold more detail. As the scan moves right, the network squeezes it down to meaning.", 13, MUTED)
    fl = [(-0.3, -0.2, 0), (8.6, -0.2, 0), (8.6, 1.4, 0), (-0.3, 1.4, 0)]
    b += f'<polygon points="{" ".join(f"{ob(*c)[0]:.1f},{ob(*c)[1]:.1f}" for c in fl)}" fill="#252423"/>'
    centers = []

    def label(xc, l1, l2, dy=0.0, c1=INK, c2=MUTED):
        nonlocal b
        b += text(ob(xc, 0, 0)[0], oy + 30, l1, 14, c1, 700, "middle") + text(ob(xc, 0, 0)[0], oy + 48, l2, 11.5, c2, 400, "middle")

    # 1 scan, 2 prepare (flat slabs)
    b += box3(0.0, 0.95, 0.95, 14, "#9a978f", "#6a6862", "#4d4b46"); centers.append(ob(0.47, 0, 40))
    label(0.47, "1  Scan", "one MRI picture")
    b += box3(1.3, 0.7, 0.7, 14, "#b4b1a8", "#77756e", "#5a5853"); centers.append(ob(1.65, 0, 40))
    label(1.65, "2  Prepare", "gray, 224 x 224")
    # 3 ResNet-18 stack (frozen)
    x = 2.6
    specs = [(0.30, 1.10, 160), (0.40, 1.00, 138), (0.50, 0.88, 116), (0.60, 0.76, 94), (0.70, 0.64, 76)]
    for dx, dy, dz in specs:
        b += box3(x, dx, dy, dz, "#8fa7c6", "#6f87a6", "#4b6079"); centers.append(ob(x + dx / 2, 0, dz / 2))
        x += dx + 0.12
    xm = 2.6 + (x - 0.12 - 2.6) / 2
    b += text(ob(xm, 0, 0)[0], oy + 30, "3  ResNet-18  (FROZEN)", 14, INK, 700, "middle")
    b += text(ob(xm, 0, 0)[0], oy + 48, "11.2 million numbers, already trained", 11.5, "#8fa7c6", 600, "middle")
    b += text(ob(xm, 0, 0)[0], oy + 64, "on over a million everyday photos", 11.5, MUTED, 400, "middle")
    # 4 new layer (trained)
    xn = x + 0.35
    b += box3(xn, 0.5, 0.6, 62, ACC2, ACC, "#a8492b"); centers.append(ob(xn + 0.25, 0, 31))
    label(xn + 0.25, "4  New last layer", "TRAINED: 2,052 numbers", c2=ACC2)
    # 5 answer bars
    xa = xn + 1.05
    hs = [118, 42, 24, 70]
    names = ["Gli", "Men", "Non", "Pit"]
    for i, h in enumerate(hs):
        b += box3(xa + i * 0.42, 0.3, 0.3, h, "#b9c9a0", SAGE if i == 0 else "#7d8f69", "#566646")
        b += text(ob(xa + i * 0.42 + 0.15, 0, 0)[0], oy + 18, names[i], 10.5, MUTED, 400, "middle")
    centers.append(ob(xa + 0.6, 0, 60))
    b += text(ob(xa + 0.6, 0, 0)[0], oy + 40, "5  Answer", 14, INK, 700, "middle")
    b += text(ob(xa + 0.6, 0, 0)[0], oy + 58, "4 scores + heatmap", 11.5, MUTED, 400, "middle")
    path = "M " + " L ".join(f"{c[0]:.0f},{c[1] - 10:.0f}" for c in centers)
    b += (f'<circle r="9" fill="{ACC}" stroke="#fff" stroke-opacity=".7" stroke-width="2"><animateMotion dur="8s" repeatCount="indefinite" path="{path}"/>'
          f'<animate attributeName="opacity" values="0;1;1;1;0" dur="8s" repeatCount="indefinite"/></circle>')
    b += f'<rect x="40" y="448" width="14" height="14" rx="3" fill="#6f87a6"/>' + text(62, 460, "frozen (never changes)", 13, BODY)
    b += f'<rect x="270" y="448" width="14" height="14" rx="3" fill="{ACC}"/>' + text(292, 460, "trained by me", 13, BODY)
    b += f'<rect x="440" y="448" width="14" height="14" rx="3" fill="{SAGE}"/>' + text(462, 460, "what you see in the app", 13, BODY)
    save("pipeline-3d.svg", svg(1200, 490, b, "3D pipeline from MRI scan to answer"))


# ------------------------------------------------------------------ 3D results bars
def results():
    u, ox, oy, H = 165, 95, 470, 300
    classes = EV["classes"]

    def ob(x, y, z):
        return ox + x * u + y * u * 0.5, oy - z - y * u * 0.34

    def P(q):
        return " ".join(f"{ob(*c)[0]:.1f},{ob(*c)[1]:.1f}" for c in q)

    acc = EV["accuracy"] + 1e-9
    b = text(40, 48, "How well did it do? In 3D", 28, INK, 500, family=SERIF)
    b += text(40, 74, f"{acc:.1%} right overall on {EV['test_size']:,} scans it had never seen. A tall bar means it found more of the real cases.", 13, MUTED)
    x0, x1, dpt = -0.15, 4.35, 0.9
    b += f'<polygon points="{P([(x0, 0, 0), (x1, 0, 0), (x1, dpt, 0), (x0, dpt, 0)])}" fill="#262523"/>'
    b += f'<polygon points="{P([(x0, dpt, 0), (x1, dpt, 0), (x1, dpt, H), (x0, dpt, H)])}" fill="#232221"/>'
    for g in range(0, 5):
        z = g * H / 4
        a1, a2 = ob(x0, dpt, z), ob(x1, dpt, z)
        b += f'<line x1="{a1[0]:.0f}" y1="{a1[1]:.0f}" x2="{a2[0]:.0f}" y2="{a2[1]:.0f}" stroke="{LINE}" stroke-width="1"/>'
        b += text(a1[0] - 8, a1[1] + 4, f"{g * 25}%", 11.5, MUTED, 400, "end")
    for i, c in enumerate(classes):
        r, pr = EV["per_class"][c]["recall"], EV["per_class"][c]["precision"]
        for j, (val, top, front, side) in enumerate([(r, ACC2, ACC, "#a8492b"), (pr, "#b9c9a0", SAGE, "#667a4e")]):
            h = val * H
            bx, dx, dy = i * 1.1 + j * 0.42 + 0.1, 0.38, 0.5
            faces = {
                "side": lambda hh: [(bx + dx, 0, 0), (bx + dx, dy, 0), (bx + dx, dy, hh), (bx + dx, 0, hh)],
                "top": lambda hh: [(bx, dy, hh), (bx + dx, dy, hh), (bx + dx, 0, hh), (bx, 0, hh)],
                "front": lambda hh: [(bx, 0, 0), (bx + dx, 0, 0), (bx + dx, 0, hh), (bx, 0, hh)],
            }
            for face, col in (("side", side), ("top", top), ("front", front)):
                fr, to = P(faces[face](0.01)), P(faces[face](h))
                b += (f'<polygon points="{to}" fill="{col}" stroke="#000" stroke-opacity=".3"><animate attributeName="points" '
                      f'from="{fr}" to="{to}" dur="1.3s" begin="{0.18 * i + 0.08 * j:.2f}s" fill="freeze" calcMode="spline" '
                      f'keyTimes="0;1" keySplines="0.2 0.8 0.2 1"/></polygon>')
            tx, ty = ob(bx + dx / 2, 0, h)
            b += (f'<text x="{tx:.0f}" y="{ty - 12:.0f}" font-size="14" fill="{INK}" font-weight="700" text-anchor="middle" '
                  f'font-family="{SANS}" opacity="0">{val:.0%}<animate attributeName="opacity" from="0" to="1" '
                  f'begin="{1.1 + 0.18 * i:.2f}s" dur="0.4s" fill="freeze"/></text>')
        b += text(ob(i * 1.1 + 0.46, 0, 0)[0], oy + 30, LABELS[c], 14.5, INK, 600, "middle")
    lx = 840
    b += f'<rect x="{lx}" y="150" width="16" height="16" rx="3" fill="{ACC}"/>' + text(lx + 26, 163, "Found (recall)", 14.5, INK, 600)
    b += text(lx + 26, 182, "of all REAL cases, how many it caught", 12.5, MUTED)
    b += f'<rect x="{lx}" y="208" width="16" height="16" rx="3" fill="{SAGE}"/>' + text(lx + 26, 221, "Correct when it speaks (precision)", 14.5, INK, 600)
    b += text(lx + 26, 240, "when it names a group, how often it is right", 12.5, MUTED)
    gl = EV["per_class"]["glioma"]["recall"]
    b += f'<rect x="{lx}" y="285" width="320" height="140" rx="14" fill="{CARD}" stroke="{ACC}" stroke-width="1.5"/>'
    b += text(lx + 20, 316, "The honest part", 17, ACC, 700)
    for k, line in enumerate([f"It finds only {gl:.0%} of real gliomas.", "Many get called meningioma instead.",
                              "Missing a tumor is the costly mistake,", "so this is NOT a medical tool."]):
        b += text(lx + 20, 344 + k * 22, line, 14, BODY)
    save("results-3d.svg", svg(1200, 540, b, "3D bar chart of recall and precision per tumor group"))


# ------------------------------------------------------------------ confusion matrix
def confusion():
    cm = np.array(EV["confusion"])
    classes = EV["classes"]
    cw, ch, ox, oy = 150, 78, 250, 120
    b = text(40, 48, "Where it was right, and where it slipped", 28, INK, 500, family=SERIF)
    b += text(40, 74, "Rows: what the scan really was.  Columns: what the model said.  Diagonal = correct.", 13, MUTED)
    for j, c in enumerate(classes):
        b += text(ox + j * cw + cw / 2, oy - 14, LABELS[c], 14, INK, 600, "middle")
    for i, c in enumerate(classes):
        b += text(ox - 14, oy + i * ch + ch / 2 + 5, LABELS[c], 14, INK, 600, "end")
        tot = cm[i].sum()
        for j in range(4):
            v, share = int(cm[i, j]), cm[i, j] / tot
            diag = i == j
            col = ACC if diag else "#a8492b"
            op = 0.25 + 0.75 * share if diag else 0.12 + 1.5 * share
            b += (f'<rect x="{ox + j * cw + 3}" y="{oy + i * ch + 3}" width="{cw - 6}" height="{ch - 6}" rx="12" fill="{col}" '
                  f'fill-opacity="{min(op, 1):.2f}" stroke="{LINE}"><animate attributeName="fill-opacity" from="0" to="{min(op, 1):.2f}" '
                  f'dur="0.8s" begin="{(i + j) * 0.08:.2f}s" fill="freeze"/></rect>')
            b += text(ox + j * cw + cw / 2, oy + i * ch + ch / 2 + 2, v, 24, INK, 700 if diag else 500, "middle", SERIF)
            b += text(ox + j * cw + cw / 2, oy + i * ch + ch / 2 + 22, f"{share:.0%}", 12, BODY, 400, "middle")
    b += text(ox + 2 * cw, oy + 4 * ch + 40, "What the model said", 13, MUTED, 400, "middle")
    b += text(40, oy + 2 * ch, "Real", 13, MUTED)
    save("confusion.svg", svg(1000, oy + 4 * ch + 60, b, "Confusion matrix"))


# ------------------------------------------------------------------ frozen vs trained
def frozen():
    total = 11_178_564
    trained = 512 * 4 + 4
    b = text(40, 48, "Why this trains in minutes, not days", 28, INK, 500, family=SERIF)
    b += text(40, 74, "Almost everything the network knows was learned by someone else, on over a million photos. I only trained the very last step.", 13, MUTED)
    b += f'<rect x="40" y="110" width="1020" height="64" rx="14" fill="#4b6079"/>'
    b += text(60, 148, f"{total - trained:,} numbers: FROZEN (kept exactly as ImageNet taught them)", 18, INK, 600)
    b += f'<rect x="1055" y="110" width="5" height="64" rx="2" fill="{ACC}"/>'
    b += f'<line x1="1057" y1="180" x2="1057" y2="214" stroke="{ACC}" stroke-width="2"/>'
    b += f'<rect x="700" y="214" width="360" height="86" rx="14" fill="{CARD}" stroke="{ACC}" stroke-width="1.5"/>'
    b += text(720, 244, f"{trained:,} numbers: TRAINED", 20, ACC, 700)
    b += text(720, 270, f"{trained / total:.3%} of the whole network.", 14, BODY)
    b += text(720, 290, "512 features x 4 groups + 4 offsets", 13, MUTED)
    b += text(40, 250, "Like hiring an expert photographer", 18, INK, 600, family=SERIF)
    b += text(40, 276, "and teaching them only to recognize four kinds of picture,", 14, BODY)
    b += text(40, 296, "instead of teaching a baby to see from zero.", 14, BODY)
    save("frozen-vs-trained.svg", svg(1100, 330, b, "Frozen versus trained numbers"))


# ------------------------------------------------------------------ rotating 3D GIFs
def font(size, serif=False):
    from PIL import ImageFont
    for name in (("georgia.ttf", "times.ttf") if serif else ("segoeui.ttf", "arial.ttf")):
        for base in ("C:/Windows/Fonts/", ""):
            try:
                return ImageFont.truetype(base + name, size)
            except OSError:
                pass
    return ImageFont.load_default()


def mri_cloud(frames=56, size=520, ss=2):
    """A rotating 3D point cloud of a real sample scan: brighter pixels float higher."""
    from PIL import Image, ImageDraw
    img = Image.open(os.path.join(ROOT, "assets", "samples", "meningioma.jpg")).convert("L")
    g = 110
    arr = np.asarray(img.resize((g, g), Image.LANCZOS)).astype(float) / 255
    jj, ii = np.nonzero(arr > 0.10)
    L = arr[jj, ii] ** 0.7
    X = (ii / (g - 1) - .5) * 2.8
    Y = (.5 - jj / (g - 1)) * 2.8
    Z = (L - .4) * 1.0
    accent = np.array([217, 119, 87.0])
    tint = np.clip(0.35 * L[:, None] * 255 + 0.65 * accent * (0.35 + L[:, None] * 0.9), 0, 255)
    W = size * ss
    bg = np.array([[34, 33, 31]] * 1, dtype=np.uint8)
    out = []
    cx, cy, D = W / 2, W * .50, 7.5
    f = 120 * ss * D
    for t in range(frames):
        ph = t / frames
        th = 2 * math.pi * ph
        im = Image.new("RGB", (W, W), (31, 30, 29))
        d = ImageDraw.Draw(im, "RGBA")
        for R in (170, 215):
            r = R * ss * (1 + .04 * math.sin(2 * math.pi * ph))
            d.ellipse([cx - r, W * .88 - r * .16, cx + r, W * .88 + r * .16], outline=(217, 119, 87, 170), width=2 * ss)
        xr = X * math.cos(th) + Z * math.sin(th)
        zr = -X * math.sin(th) + Z * math.cos(th)
        s = f / (D - zr)
        sx, sy = cx + xr * s, cy - Y * s
        r = (.30 + .6 * L) * .021 * s
        for k in np.argsort(zr):
            d.ellipse([sx[k] - r[k], sy[k] - r[k], sx[k] + r[k], sy[k] + r[k]], fill=tuple(int(c) for c in tint[k]))
        d.text((22 * ss, 18 * ss), "One MRI slice, lifted into 3D", font=font(19 * ss, True), fill=(245, 244, 238))
        d.text((22 * ss, 46 * ss), "brighter pixel = higher point", font=font(13 * ss), fill=(163, 161, 152))
        out.append(im.resize((size, size), Image.LANCZOS))
    pal = [f.quantize(colors=96, method=Image.MEDIANCUT, dither=Image.NONE) for f in out]
    p = os.path.join(OUT, "mri-3d.gif")
    pal[0].save(p, save_all=True, append_images=pal[1:], duration=60, loop=0, optimize=True, disposal=1)
    print("wrote mri-3d.gif", os.path.getsize(p) // 1024, "KB")


def gradcam_surface(frames=48, size=520):
    """The Grad-CAM heatmap as a rotating 3D mountain range over the scan (real model output)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch
    from pytorch_grad_cam import GradCAM
    import model as M

    net, classes = M.load_model()
    img = Image.open(os.path.join(ROOT, "assets", "samples", "pituitary.jpg"))
    x = M.transform()(img.convert("RGB"))
    for p in net.parameters():
        p.requires_grad = True
    with GradCAM(model=net, target_layers=[net.layer4[-1]]) as cam:
        heat = cam(input_tensor=x.unsqueeze(0), targets=None)[0]
    _, overlay, _ = M.analyze(net, classes, img)
    n = 56
    hs = np.asarray(Image.fromarray((heat * 255).astype("uint8")).resize((n, n), Image.BILINEAR)) / 255.0
    cs = np.asarray(Image.fromarray(overlay).resize((n, n), Image.BILINEAR)) / 255.0
    gx, gy = np.meshgrid(np.linspace(-1, 1, n), np.linspace(1, -1, n))
    frames_out = []
    for t in range(frames):
        fig = plt.figure(figsize=(5.2, 5.2), dpi=100, facecolor="#1f1e1d")
        ax = fig.add_subplot(111, projection="3d", facecolor="#1f1e1d")
        ax.plot_surface(gx, gy, hs * 0.9, facecolors=cs, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        ax.set_zlim(0, 1.2)
        ax.view_init(elev=38, azim=t * 360 / frames)
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 0.45))
        fig.text(0.04, 0.95, "Grad-CAM as a 3D landscape", color="#f5f4ee", fontsize=13, family="serif")
        fig.text(0.04, 0.915, "taller = pushed harder toward the answer", color="#a3a198", fontsize=9)
        fig.canvas.draw()
        frames_out.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).copy())
        plt.close(fig)
    pal = [f.quantize(colors=96, method=Image.MEDIANCUT, dither=Image.NONE) for f in frames_out]
    p = os.path.join(OUT, "gradcam-3d.gif")
    pal[0].save(p, save_all=True, append_images=pal[1:], duration=70, loop=0, optimize=True, disposal=1)
    print("wrote gradcam-3d.gif", os.path.getsize(p) // 1024, "KB")


if __name__ == "__main__":
    banner()
    pipeline()
    results()
    confusion()
    frozen()
    if "--gifs" in sys.argv:
        mri_cloud()
        gradcam_surface()
