"""Makes docs/img/tour.gif: an animated tour built from the real screenshots in docs/screenshots/.
Run from the project root:  python tools/make_tour.py
"""
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = [("01-welcome-popup.jpg", "A welcome message for every visitor: learning project, not a medical tool"),
         ("02-result-dark.jpg", "Pick a scan: see where the model looked and how sure it is"),
         ("03-result-light.jpg", "Light and dark mode, on every screen size"),
         ("04-journey.jpg", "Our journey: how it was built, step by step"),
         ("05-about-3d.jpg", "About me: a 3D hologram that reacts as you scroll"),
         ("06-documentation.jpg", "An 11-chapter documentation book, in plain words")]


def font(size, serif=False):
    for name in (("georgia.ttf", "times.ttf") if serif else ("segoeui.ttf", "arial.ttf")):
        for base in ("C:/Windows/Fonts/", ""):
            try:
                return ImageFont.truetype(base + name, size)
            except OSError:
                pass
    return ImageFont.load_default()


def main(width=800):
    frames, durs, prev = [], [], None
    for n, (fn, cap) in enumerate(SHOTS, 1):
        im = Image.open(os.path.join(ROOT, "docs", "screenshots", fn)).convert("RGB")
        im = im.resize((width, int(width * 500 / 800)), Image.LANCZOS)
        bar = Image.new("RGBA", im.size, (0, 0, 0, 0))
        bd = ImageDraw.Draw(bar)
        bh = 46
        bd.rectangle([0, im.height - bh, im.width, im.height], fill=(20, 19, 18, 225))
        bd.text((16, im.height - bh + 11), f"{n} / {len(SHOTS)}", font=font(18, True), fill=(217, 119, 87, 255))
        bd.text((82, im.height - bh + 13), cap, font=font(16), fill=(245, 244, 238, 255))
        cur = Image.alpha_composite(im.convert("RGBA"), bar).convert("RGB")
        if prev is not None:
            for a in (.25, .5, .75):
                frames.append(Image.blend(prev, cur, a))
                durs.append(60)
        frames.append(cur)
        durs.append(2200)
        prev = cur
    pal = [f.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.NONE) for f in frames]
    out = os.path.join(ROOT, "docs", "img", "tour.gif")
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True, disposal=1)
    print("wrote tour.gif", os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    main()
