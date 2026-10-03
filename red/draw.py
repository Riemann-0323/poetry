"""Vector overlay library for 《红色》: every literal image the poem names, drawn with anti-aliased OpenCV into
premultiplied uint8 layers (over) and an additive glow layer."""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import *

GOLD = (0.86, 0.66, 0.30)
WHITE = (1.0, 0.97, 0.94)
RED = (0.95, 0.06, 0.12)
DARK = (0.03, 0.0, 0.01)
JADE = (0.45, 0.85, 0.72)
SH = 4                      # sub-pixel bits for cv2


def sm(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def sst(a, b, t):
    return sm((t - a) / (b - a))


class Layer:
    def __init__(self):
        self.over = np.zeros((H, W, 4), np.uint8)
        self.glow = np.zeros((H, W, 4), np.uint8)
        self.used = False
        self.gused = False

    @staticmethod
    def pm(col, a):
        a = max(0.0, min(1.0, a))
        return (col[0] * a * 255, col[1] * a * 255, col[2] * a * 255, a * 255)

    @staticmethod
    def pts(p):
        return [np.round(np.asarray(q, np.float64) * (1 << SH)).astype(np.int32).reshape(-1, 1, 2) for q in p]

    def line(self, p, col, a=1.0, th=2, glow=0.0, closed=False):
        if a <= 0.003:
            return
        P = self.pts([p])
        cv2.polylines(self.over, P, closed, self.pm(col, a), max(1, int(th)), cv2.LINE_AA, SH)
        self.used = True
        if glow > 0:
            g = tuple(min(255, c * glow * a * 255) for c in col) + (255,)
            cv2.polylines(self.glow, P, closed, g, max(1, int(th) + 2), cv2.LINE_AA, SH)
            self.gused = True

    def lines(self, plist, col, a=1.0, th=2, glow=0.0):
        if a <= 0.003 or not plist:
            return
        P = self.pts(plist)
        cv2.polylines(self.over, P, False, self.pm(col, a), max(1, int(th)), cv2.LINE_AA, SH)
        self.used = True
        if glow > 0:
            g = tuple(min(255, c * glow * a * 255) for c in col) + (255,)
            cv2.polylines(self.glow, P, False, g, max(1, int(th) + 2), cv2.LINE_AA, SH)
            self.gused = True

    def fill(self, p, col, a=1.0, glow=0.0):
        if a <= 0.003:
            return
        P = self.pts([p])
        cv2.fillPoly(self.over, P, self.pm(col, a), cv2.LINE_AA, SH)
        self.used = True
        if glow > 0:
            g = tuple(min(255, c * glow * a * 255) for c in col) + (255,)
            cv2.fillPoly(self.glow, P, g, cv2.LINE_AA, SH)
            self.gused = True

    def circle(self, c, r, col, a=1.0, th=-1, glow=0.0):
        if a <= 0.003 or r <= 0:
            return
        cc = (int(round(c[0] * 16)), int(round(c[1] * 16)))
        cv2.circle(self.over, cc, int(round(r * 16)), self.pm(col, a), th if th < 0 else max(1, int(th)), cv2.LINE_AA, SH)
        self.used = True
        if glow > 0:
            g = tuple(min(255, x * glow * a * 255) for x in col) + (255,)
            cv2.circle(self.glow, cc, int(round(r * 16)), g, th if th < 0 else max(1, int(th) + 2), cv2.LINE_AA, SH)
            self.gused = True

    def alpha_blit(self, al, x, y, col, a=1.0, glow=0.0):
        """composite an alpha mask (float 0..1) at integer position"""
        h, w = al.shape
        x, y = int(round(x)), int(round(y))
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x0 >= x1 or y0 >= y1 or a <= 0.003:
            return
        sub = al[y0 - y:y1 - y, x0 - x:x1 - x, None] * a
        reg = self.over[y0:y1, x0:x1].astype(np.float32)
        src = np.array([col[0] * 255, col[1] * 255, col[2] * 255, 255], np.float32)
        reg = reg * (1 - sub) + src * sub
        self.over[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)
        self.used = True
        if glow > 0:
            g = self.glow[y0:y1, x0:x1].astype(np.float32)
            g[..., :3] += sub * np.array(col, np.float32) * 255 * glow
            self.glow[y0:y1, x0:x1] = np.clip(g, 0, 255).astype(np.uint8)
            self.gused = True


# ---------------------------------------------------------------- fonts / strings for the overlay
FONT_FILES = dict(serif="NotoSerifSC-Black.otf", serif_sb="NotoSerifSC-SemiBold.otf", sans="NotoSansSC-Black.otf",
                  anton="Anton.ttf", mono="JetBrainsMono-Regular.ttf", cinzel="Cinzel.ttf", frak="Unifraktur.ttf",
                  kai="wenkai.ttf", kr="NotoSansKR.otf", deva="NotoSansDeva.ttf", arab="NotoSansArabic.ttf",
                  hebr="NotoSansHebrew.ttf", egy="NotoSansEgypt.ttf", lightserif="NotoSerifSC.otf")
_FC, _SC = {}, {}


def font(key, size):
    k = (key, size)
    if k not in _FC:
        _FC[k] = ImageFont.truetype(f"{FONT_DIR}/{FONT_FILES[key]}", size)
    return _FC[k]


def string_mask(s, key, size, stroke=0):
    k = (s, key, size, stroke)
    if k not in _SC:
        f = font(key, size)
        l, t, r, b = f.getbbox(s, stroke_width=stroke)
        pad = 4 + stroke
        im = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
        ImageDraw.Draw(im).text((pad - l, pad - t), s, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
        a = np.asarray(im, np.float32) / 255
        _SC[k] = a
        if len(_SC) > 4000:
            _SC.clear()
    return _SC[k]


def text(L, s, key, size, x, y, col, a=1.0, anchor="lt", glow=0.0, stroke=0, rot=0.0, scale=1.0):
    al = string_mask(s, key, size, stroke)
    if scale != 1.0:
        al = cv2.resize(al, (max(1, int(al.shape[1] * scale)), max(1, int(al.shape[0] * scale))), interpolation=cv2.INTER_LINEAR)
    if rot:
        h, w = al.shape
        d = int(math.hypot(w, h)) + 2
        canvas = np.zeros((d, d), np.float32)
        canvas[(d - h) // 2:(d - h) // 2 + h, (d - w) // 2:(d - w) // 2 + w] = al
        M = cv2.getRotationMatrix2D((d / 2, d / 2), -math.degrees(rot), 1.0)
        al = cv2.warpAffine(canvas, M, (d, d), flags=cv2.INTER_LINEAR)
    h, w = al.shape
    ox = {"l": 0, "c": w / 2, "r": w}[anchor[0]]
    oy = {"t": 0, "c": h / 2, "b": h}[anchor[1]]
    L.alpha_blit(al, x - ox, y - oy, col, a, glow)
    return w, h


# ================================================================= S1
_rw = np.random.default_rng(1)
WILLOW = [(x, _rw.uniform(380, 820), _rw.uniform(0, 6.28), _rw.uniform(-.3, .3)) for x in np.linspace(-60, W + 60, 30)]


def willow(L, t, col=GOLD, a=1.0, sway=1.0, leaves=True):
    plist, leafs = [], []
    for i, (x0, ln, ph, cu) in enumerate(WILLOW):
        s = np.linspace(0, 1, 40)
        sw = sway * (40 * s ** 2 * math.sin(1.1 * t + ph) + 14 * s ** 3 * math.sin(2.3 * t + ph * 2))
        xs = x0 + cu * 120 * s ** 2 + sw
        ys = -30 + s * ln
        plist.append(np.stack([xs, ys], 1))
        if leaves:
            for k in range(6, 40, 3):
                side = 1 if k % 2 else -1
                dx, dy = side * 10, 16
                leafs.append(np.array([[xs[k], ys[k]], [xs[k] + dx, ys[k] + dy]]))
    L.lines(plist, col, a * 0.9, 1, glow=0.35)
    if leafs:
        L.lines(leafs, col, a * 0.7, 2, glow=0.2)


_rs = np.random.default_rng(3)
SPIDERS = [dict(y0=_rs.uniform(150, 950), y1=_rs.uniform(150, 950), sp=_rs.uniform(0.07, 0.13), off=_rs.uniform(0, 1),
                sz=_rs.uniform(0.7, 1.4), dirn=1 if i % 2 else -1) for i in range(8)]


def spiders(L, t, lt, a=1.0):
    for k, s in enumerate(SPIDERS):
        u = (s["off"] + lt * s["sp"]) % 1.3 - 0.15
        x = -100 + u * (W + 200)
        if s["dirn"] < 0:
            x = W - x
        y = s["y0"] + (s["y1"] - s["y0"]) * u + 20 * math.sin(lt * 2 + k)
        ang = math.atan2((s["y1"] - s["y0"]), (W + 200) * s["dirn"])
        sz = s["sz"]
        ca, sa = math.cos(ang), math.sin(ang)
        def R(px, py):
            return (x + (px * ca - py * sa) * sz, y + (px * sa + py * ca) * sz)
        legs = []
        for side in (-1, 1):
            for j in range(4):
                base = (-6 + j * 5, side * 6)
                phase = lt * 9 + j * 1.6 + (0 if side > 0 else math.pi)
                fa = (-0.9 + j * 0.6) + 0.25 * math.sin(phase)
                foot = (base[0] + math.cos(fa) * 46 * (1 if j < 2 else -1) * -1, side * (40 + 6 * math.cos(phase)))
                knee = ((base[0] + foot[0]) / 2 + (j - 1.5) * 4, side * 32 + side * 6 * max(0, math.sin(phase)))
                legs.append(np.array([R(*base), R(*knee), R(*foot)]))
        L.lines(legs, DARK, 0.95 * a, 2)
        body = cv2.ellipse2Poly((0, 0), (16, 11), 0, 0, 360, 20)
        abd = cv2.ellipse2Poly((22, 0), (22, 16), 0, 0, 360, 20)
        L.fill(np.array([R(px, py) for px, py in body]), DARK, 0.97 * a)
        L.fill(np.array([R(px, py) for px, py in abd]), DARK, 0.97 * a)
        L.line(np.array([R(px, py) for px, py in abd[3:9]]), RED, 0.6 * a, 1)
        for e in (-4, 4):
            L.circle(R(-12, e), 1.6 * sz, (1, 0.1, 0.1), a, glow=1.5)


def dwarves(L, t, lt, a=1.0):
    n = 13
    for i in range(n):
        x = 160 + i * (W - 320) / (n - 1)
        y = 1050
        ph = (t / (2 * BEAT) + i * 0.08) % 1.0
        bow = 1.1 * math.sin(math.pi * min(1, ph * 1.6)) ** 2
        hx, hy = x + math.sin(bow) * 44, y - math.cos(bow) * 44
        L.line(np.array([[x - 8, y], [x, y - 14], [x + 8, y]]), DARK, a, 4)
        L.line(np.array([[x, y - 14], [hx, hy]]), DARK, a, 9)
        L.circle((hx + math.sin(bow) * 12, hy - math.cos(bow) * 12), 9, DARK, a)
        L.circle((hx + math.sin(bow) * 12, hy - math.cos(bow) * 12 - 9), 3, GOLD, a * 0.8, glow=0.6)


_ra = np.random.default_rng(4)
ARMS = [(x, _ra.uniform(150, 420), _ra.uniform(0, 6.28)) for x in np.linspace(40, W - 40, 46)]


def arms(L, t, lt, a=1.0):
    grow = sst(0, 2.5, lt)
    pl = []
    hands = []
    for x, h, ph in ARMS:
        hh = h * (0.5 + 0.5 * grow) * (0.9 + 0.1 * math.sin(t * 1.3 + ph))
        tip = (x + 30 * math.sin(t * 0.8 + ph), H + 10 - hh)
        mid = (x + 12 * math.sin(t + ph), H + 10 - hh * 0.5)
        pl.append(np.array([[x, H + 10], mid, tip]))
        for f in range(5):
            fa = -math.pi / 2 + (f - 2) * 0.28 + 0.1 * math.sin(t * 3 + ph + f)
            hands.append(np.array([tip, (tip[0] + math.cos(fa) * 26, tip[1] + math.sin(fa) * 26)]))
    L.lines(pl, DARK, a, 9)
    L.lines(hands, DARK, a, 3)


_rsh = np.random.default_rng(6)
SHARDS = []
for i in range(11):
    pts = [(0, -_rsh.uniform(40, 120)), (_rsh.uniform(12, 30), _rsh.uniform(-10, 30)), (-_rsh.uniform(12, 30), _rsh.uniform(0, 40))]
    SHARDS.append(dict(p=np.array(pts, float), y=_rsh.uniform(150, 930), sp=_rsh.uniform(700, 1400), rot=_rsh.uniform(-4, 4),
                       d=_rsh.uniform(0, 1.2)))


def spear_shards(L, t, lt, a=1.0):
    for s in SHARDS:
        tt = lt - s["d"]
        if tt < 0:
            continue
        x = -150 + tt * s["sp"]
        if x > W + 200:
            continue
        ang = s["rot"] * tt
        c, si = math.cos(ang), math.sin(ang)
        P = s["p"] @ np.array([[c, si], [-si, c]]) * 1.3 + [x, s["y"]]
        trail = np.array([[x - 260, s["y"]], [x - 20, s["y"]]])
        L.line(trail, (1, 0.8, 0.8), 0.25 * a, 1, glow=0.5)
        L.fill(P, (0.55, 0.55, 0.6), a)
        L.line(P, WHITE, a, 1, glow=0.8, closed=True)


def bolt(seed, x0, y0, x1, y1, depth=7):
    r = np.random.default_rng(seed)
    pts = [np.array([x0, y0], float), np.array([x1, y1], float)]
    disp = 160.0
    for _ in range(depth):
        new = [pts[0]]
        for p, q in zip(pts[:-1], pts[1:]):
            m = (p + q) / 2
            d = q - p
            n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-6)
            new += [m + n * r.normal(0, disp), q]
        pts = new
        disp *= 0.55
    return np.array(pts)


BOLTS = [bolt(s, 900 + 200 * (s - 2), -20, 760 + 300 * (s - 2), 1100) for s in range(3)]
BRANCH = [bolt(10 + s, BOLTS[0][40 * s + 20][0], BOLTS[0][40 * s + 20][1], BOLTS[0][40 * s + 20][0] + 260 * (1 - 2 * (s % 2)),
               BOLTS[0][40 * s + 20][1] + 300, 5) for s in range(3)]


def lightning(L, t, t0):
    dt = t - t0
    if dt < 0 or dt > 0.45:
        return
    on = [0, 0.05, 0.11, 0.2, 0.3]
    vis = any(o <= dt < o + 0.045 for o in on)
    if not vis:
        return
    k = int(dt / 0.11) % 3
    L.line(BOLTS[k], (0.85, 0.8, 1.0), 1, 3, glow=3.0)
    L.lines([b for b in BRANCH], (0.8, 0.75, 1.0), 0.8, 1, glow=2.0)


# ================================================================= S2
def certificate(L, t, lt, a=1.0):
    x0, y0, w, h = 1240, 680, 580, 370
    k = sst(0.2, 0.9, lt) * a
    if k <= 0:
        return
    L.fill(np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]]), (0.02, 0, 0.01), 0.72 * k)
    L.line(np.array([[x0 + 10, y0 + 10], [x0 + w - 10, y0 + 10], [x0 + w - 10, y0 + h - 10], [x0 + 10, y0 + h - 10]]), GOLD, k, 1, closed=True)
    text(L, "GEMOLOGICAL REPORT", "cinzel", 30, x0 + 34, y0 + 34, GOLD, k)
    rows = [("SPECIES", "Spinel  尖晶石"), ("VARIETY", "'Pigeon's Blood'  鸽血红"), ("ORIGIN", "International"),
            ("WEIGHT", "1.00 ct"), ("COMMENT", "Not a ruby.")]
    for i, (a1, b1) in enumerate(rows):
        kk = sst(0.5 + 0.25 * i, 0.7 + 0.25 * i, lt) * a
        text(L, a1, "mono", 22, x0 + 34, y0 + 100 + i * 46, (0.75, 0.6, 0.5), kk)
        text(L, b1, "serif_sb" if any(ord(c) > 255 for c in b1) else "mono", 24, x0 + 200, y0 + 98 + i * 46,
             WHITE if a1 != "COMMENT" else (1, 0.3, 0.3), kk)
    st = sst(2.6, 2.75, lt) * a
    if st > 0:
        text(L, "NOT RUBY", "anton", 64, x0 + w - 70, y0 + h - 70, (0.95, 0.1, 0.15), st * 0.85, anchor="rb", rot=-0.18,
             scale=1 + 0.5 * (1 - sst(2.6, 2.7, lt)))


def kneeler(L, t, lt, t_kneel, a=1.0, gem=True):
    """gold line-figure; it tries to kneel but the spinel on its knee will not let it"""
    cx, gy = 1320, 980
    tk = t - t_kneel
    if tk < 0:
        bend = 0.0
    else:
        bend = 0.8 * sm(tk / 0.35) - 0.55 * sm((tk - 0.35) / 0.12) * math.exp(-max(tk - 0.47, 0) * 3) * 1.0
        bend = max(0.0, bend) * (1 + 0.15 * math.sin(tk * 25) * math.exp(-tk * 3))
    hip = np.array([cx, gy - 420 + 150 * bend])
    knee = np.array([cx + 120 * math.sin(bend * 1.2), gy - 220 + 100 * bend])
    ankle = np.array([cx - 10, gy])
    toe = ankle + [70, 0]
    sh = hip + [-40 * bend, -260]
    head = sh + [-30 * bend, -80]
    hand = sh + [140 - 60 * bend, 140 + 40 * bend]
    leg2k = knee + [-60, 20]
    body = [np.array([ankle, knee, hip]), np.array([toe, ankle]), np.array([hip, sh]), np.array([sh, (sh + hand) / 2 + [20, 30], hand]),
            np.array([[cx - 60, gy], leg2k, hip])]
    L.lines(body, GOLD, a, 3, glow=0.9)
    L.circle(head, 46, GOLD, a, th=3, glow=0.9)
    if gem:
        g = knee + [16, -6]
        s = 26
        P = np.array([g + [0, -s], g + [s, 0], g + [0, s], g + [-s, 0]])
        L.fill(P, (0.85, 0.02, 0.1), a, glow=1.4)
        L.line(P, (1, 0.6, 0.6), a, 1, glow=1.0, closed=True)
        L.line(np.array([g + [-s, 0], g + [s, 0]]), (1, 0.5, 0.5), a * 0.8, 1)
        tw = 0.5 + 0.5 * math.sin(t * 7)
        L.line(np.array([g + [-40 * tw, -30], g + [40 * tw, -30]]), WHITE, a * tw, 1, glow=2)
    return head


def halo(L, t, center, t_break, a=1.0):
    c = np.array(center, float) + [0, -135]
    dt = t - t_break
    c = c + [0, 40]
    if dt < 0:
        ring = cv2.ellipse2Poly((0, 0), (78, 20), -8, 0, 360, 6).astype(float) + c
        L.line(ring, GOLD, a, 5, glow=1.8, closed=True)
        return
    for i in range(7):
        ang0 = i * 2 * math.pi / 7
        arc = cv2.ellipse2Poly((0, 0), (78, 20), -8, int(math.degrees(ang0)), int(math.degrees(ang0 + 0.8)), 8).astype(float)
        d = np.array([math.cos(ang0 + 0.4), math.sin(ang0 + 0.4)])
        off = c + d * (dt * 600 + 60 * dt * dt * 80)
        rr = dt * (3 if i % 2 else -3)
        R = np.array([[math.cos(rr), -math.sin(rr)], [math.sin(rr), math.cos(rr)]])
        L.line(arc @ R.T + off, GOLD, a * max(0, 1 - dt / 1.2), 4, glow=1.5)


def chat_texture():
    """the handkerchief: a night of messages, printed in red thread"""
    im = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(f"{FONT_DIR}/wenkai.ttf", 34)
    fs = ImageFont.truetype(f"{FONT_DIR}/JetBrainsMono-Regular.ttf", 18)
    msgs = [("23:41", "L", "在吗"), ("23:41", "R", "在"), ("23:52", "L", "听说他们……"), ("00:07", "L", "你别告诉别人"),
            ("00:13", "R", "嗯"), ("01:30", "L", "[语音 0:59]"), ("02:46", "L", "对方撤回了一条消息"), ("03:15", "R", "你还在吗"),
            ("04:58", "L", "天亮了"), ("05:52", "R", "晚安")]
    y = 70
    ink = (150, 12, 28, 235)
    for ts, side, m in msgs:
        if "撤回" in m:
            d.text((512, y + 6), m, font=fs, fill=(150, 12, 28, 160), anchor="mt")
            y += 70
            continue
        d.text((512, y - 26), ts, font=fs, fill=(150, 12, 28, 140), anchor="mt")
        w = d.textlength(m, font=f)
        if side == "L":
            box = [110, y, 110 + w + 44, y + 58]
        else:
            box = [914 - w - 44, y, 914, y + 58]
        d.rounded_rectangle(box, 14, outline=ink, width=3)
        d.text((box[0] + 22, y + 9), m, font=f, fill=ink)
        y += 92
    return np.asarray(im, np.float32) / 255


# ================================================================= S3
BABEL_WORDS = [("红", "serif"), ("red", "cinzel"), ("rouge", "cinzel"), ("rot", "cinzel"), ("rosso", "cinzel"),
               ("rojo", "cinzel"), ("vermelho", "cinzel"), ("красный", "serif_sb"), ("κόκκινο", "serif_sb"),
               ("赤", "serif"), ("빨강", "kr"), ("أحمر", "arab"), ("אדום", "hebr"), ("लाल", "deva"), ("merah", "cinzel"),
               ("czerwony", "serif_sb"), ("punainen", "cinzel"), ("rubrum", "cinzel"), ("𓂋", "egy")]


def babel_words(L, t, lt, a=1.0):
    k = sst(0.2, 1.2, lt) * a
    for i, (w, fk) in enumerate(BABEL_WORDS):
        ring = i % 5
        ang = t * (0.35 + 0.08 * ring) * (1 if ring % 2 else -1) + i * 2.4
        cy = 820 - ring * 120
        rx = 620 - ring * 70
        x = 960 + math.cos(ang) * rx
        y = cy + math.sin(ang) * 40
        depth = math.sin(ang)
        col = WHITE if depth > 0 else (0.9, 0.4, 0.35)
        text(L, w, fk, int(34 + 18 * (depth + 1) / 2), x, y, col, k * (0.35 + 0.65 * (depth + 1) / 2), anchor="cc",
             glow=0.4 if depth > 0 else 0)


def pages(L, t, lt, a=1.0):
    cx, cy, pw, ph = 1300, 580, 440, 600
    rate = 3 + 22 * sst(0, 3.5, lt)
    n = int(lt * rate)
    for j in range(3):
        ph_ = (lt * rate - n) + j / 3
        ang = (ph_ % 1.0) * math.pi
        sx = math.cos(ang)
        x1 = cx + pw * sx
        lift = math.sin(ang) * 40
        P = np.array([[cx, cy - ph / 2], [x1, cy - ph / 2 - lift], [x1, cy + ph / 2 + lift * 0.2], [cx, cy + ph / 2]])
        shade = 0.55 + 0.45 * abs(sx)
        L.fill(P, (0.92 * shade, 0.88 * shade, 0.85 * shade), a * 0.92)
        rows = []
        for r in range(16):
            yy = cy - ph / 2 + 60 + r * 34
            rows.append(np.array([[cx + 30 * np.sign(sx) if sx else cx, yy], [cx + (pw - 40) * sx, yy - lift * (1 - r / 16) * 0.3]]))
        L.lines(rows, (0.35, 0.3, 0.3), a * 0.6 * abs(sx), 2)
    L.fill(np.array([[cx - pw, cy - ph / 2], [cx, cy - ph / 2], [cx, cy + ph / 2], [cx - pw, cy + ph / 2]]), (0.88, 0.84, 0.8), a * 0.9)
    rows = [np.array([[cx - pw + 40, cy - ph / 2 + 60 + r * 34], [cx - 40, cy - ph / 2 + 60 + r * 34]]) for r in range(16)]
    L.lines(rows, (0.35, 0.3, 0.3), a * 0.55, 2)
    page = int(min(3000, 1 + (2999 * (math.exp(lt * 1.25) - 1) / (math.exp(4.8) - 1))))
    text(L, f"p. {page:04d} / 3000", "anton", 96, 1840, 120, WHITE, a, anchor="rt", glow=0.4)
    text(L, "Dissertation · Babel University of Linguistic Magic", "cinzel", 26, 1840, 240, GOLD, a * 0.85, anchor="rt")


def wineglass(L, t, lt, a=1.0):
    cx, top, bowl_h, R = 1420, 280, 300, 170
    ys = np.linspace(0, 1, 30)
    left = np.stack([cx - R * np.sin(ys * math.pi * 0.62 + 0.25) * 1.05, top + ys * bowl_h], 1)
    right = left.copy()
    right[:, 0] = 2 * cx - left[:, 0]
    level = 0.45
    wob = 6 * math.sin(t * 3)
    fy = top + level * bowl_h
    wine = [p for p in left if p[1] >= fy] + [p for p in right[::-1] if p[1] >= fy]
    if len(wine) > 2:
        wine = np.array(wine)
        wine[0, 1] += wob
        wine[-1, 1] -= wob
        L.fill(wine, (0.6, 0.0, 0.08), a * 0.9, glow=0.3)
    stem = np.array([[cx, top + bowl_h], [cx, top + bowl_h + 230]])
    base = cv2.ellipse2Poly((cx, top + bowl_h + 236), (110, 18), 0, 0, 360, 10)
    L.lines([left, right, stem], WHITE, a, 3, glow=0.7)
    L.line(base, WHITE, a, 3, glow=0.7, closed=True)
    rr = np.random.default_rng(8)
    for i in range(26):
        ph = (lt * rr.uniform(0.3, 0.8) + rr.random()) % 1.0
        bx = cx + rr.uniform(-R * 0.6, R * 0.6) + 6 * math.sin(lt * 4 + i)
        by = top + bowl_h - 10 - ph * (bowl_h * (1 - level) - 10)
        L.circle((bx, by), 2 + 3 * rr.random(), (1, 0.75, 0.7), a * 0.8, th=1, glow=0.6)
    for i in range(14):
        ang = i * 0.45 + lt
        L.circle((cx + math.cos(ang) * R * 0.75 * (i / 14), fy - 4 + 3 * math.sin(lt * 6 + i)), 4 + 2 * math.sin(lt * 5 + i),
                 (1, 0.85, 0.8), a * 0.7, th=1, glow=0.8)


def chips(L, t, t_push, a=1.0):
    k = sst(t_push, t_push + 0.35, t)
    stacks = [(-1, 900, 760), (-1, 1040, 880), (1, 1800, 760), (1, 1700, 900)]
    for side, x, y in stacks:
        xx = x + (1420 - x) * 0.8 * k
        for j in range(9):
            yy = y - j * 14 - (900 - y) * 0 + (1 - k) * 0
            col = (0.75, 0.03, 0.08) if j % 2 == 0 else (0.92, 0.9, 0.86)
            e = cv2.ellipse2Poly((int(xx), int(yy)), (62, 20), 0, 0, 360, 10)
            L.fill(e, col, a)
            L.line(e, (0.1, 0, 0), a * 0.8, 1, closed=True)
            for q in range(6):
                ang = q * 60 + 15
                px, py = xx + 62 * math.cos(math.radians(ang)), yy + 20 * math.sin(math.radians(ang))
                L.circle((px, py), 4, (1, 1, 1) if j % 2 == 0 else (0.75, 0.03, 0.08), a * 0.9)


# ================================================================= S4
def roulette(L, t, lt, a=1.0, cx=1330, cy=560, R=400):
    spin = t * 2.4
    order = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26]
    n = 37
    for i, num in enumerate(order):
        a0 = spin + i * 2 * math.pi / n
        a1 = a0 + 2 * math.pi / n
        ang = np.linspace(a0, a1, 6)
        outer = np.stack([cx + R * np.cos(ang), cy + R * np.sin(ang)], 1)
        inner = np.stack([cx + R * 0.72 * np.cos(ang[::-1]), cy + R * 0.72 * np.sin(ang[::-1])], 1)
        col = (0.05, 0.4, 0.15) if num == 0 else ((0.78, 0.03, 0.08) if i % 2 else (0.04, 0.02, 0.02))
        L.fill(np.concatenate([outer, inner]), col, a)
        am = (a0 + a1) / 2
        text(L, str(num), "anton", 26, cx + R * 0.86 * math.cos(am), cy + R * 0.86 * math.sin(am), WHITE, a, anchor="cc",
             rot=am + math.pi / 2)
    L.circle((cx, cy), R, GOLD, a, th=4, glow=0.6)
    L.circle((cx, cy), R * 0.72, GOLD, a, th=2)
    L.circle((cx, cy), R * 0.42, (0.2, 0.08, 0.04), a)
    L.circle((cx, cy), R * 0.42, GOLD, a, th=3, glow=0.5)
    for q in range(4):
        aq = -spin * 0.6 + q * math.pi / 2
        L.line(np.array([[cx, cy], [cx + R * 0.4 * math.cos(aq), cy + R * 0.4 * math.sin(aq)]]), GOLD, a, 6, glow=0.5)
    L.circle((cx, cy), 26, GOLD, a, glow=0.8)
    br = R * (1.08 - 0.25 * sst(1.5, 4.5, lt))
    ba = -t * 5.5 * (1 - 0.6 * sst(1.5, 5, lt))
    L.circle((cx + br * math.cos(ba), cy + br * math.sin(ba)), 13, WHITE, a, glow=1.5)


def counters(L, t, lt, a=1.0):
    v = int(1000 + 9999999 * (math.exp(lt * 0.9) - 1) / (math.exp(5) - 1))
    text(L, "积分 POINTS", "sans", 30, 100, 760, GOLD, a)
    text(L, f"{min(v, 9999999):07d}", "anton", 120, 100, 800, WHITE, a, glow=0.4)
    beats = int(lt / BEAT)
    for i in range(max(0, beats - 2), beats + 1):
        dt = lt - i * BEAT
        if 0 <= dt < 0.6:
            sign = "+" if (i * 7) % 3 else "−"
            text(L, f"{sign}{(i * 397) % 900 + 100}", "anton", 54, 560 + (i % 3) * 60, 790 - dt * 120,
                 (0.4, 1, 0.6) if sign == "+" else (1, 0.2, 0.2), a * (1 - dt / 0.6))


def rose_curve(cx, cy, R, rot, k=5 / 4, n=900):
    th = np.linspace(0, 8 * math.pi, n)
    r = R * np.cos(k * th)
    return np.stack([cx + r * np.cos(th + rot), cy + r * np.sin(th + rot)], 1)


def price_rose(L, t, lt, a=1.0):
    cx, cy = 1330, 430
    L.line(rose_curve(cx, cy, 190, t * 0.2), (0.95, 0.15, 0.22), a, 2, glow=1.2)
    stem = np.array([[cx, cy + 190], [cx - 20, cy + 380], [cx + 10, cy + 600]])
    L.line(stem, (0.3, 0.55, 0.25), a, 5)
    L.line(np.array([[cx - 14, cy + 330], [cx - 90, cy + 290], [cx - 20, cy + 360]]), (0.3, 0.55, 0.25), a, 4)
    sw = 0.15 * math.sin(t * 2.5)
    tx, ty = cx + 40, cy + 420
    L.line(np.array([[cx - 10, cy + 400], [tx + 30, ty + 20]]), WHITE, a, 1)
    c, s = math.cos(sw), math.sin(sw)
    tag = np.array([[0, 0], [260, -40], [300, 60], [40, 100]], float) @ np.array([[c, s], [-s, c]]) + [tx + 30, ty + 20]
    L.fill(tag, (0.95, 0.92, 0.86), a)
    L.circle(tuple(tag[0] + (tag[2] - tag[0]) * 0.12), 8, (0.2, 0.0, 0.02), a)
    scr = "¥ " + "".join("0123456789?∞"[int((t * 37 + i * 7) % 12)] for i in range(5)) if lt < 3.2 else "¥ NaN"
    text(L, scr, "anton", 64, tag[:, 0].mean() + 10, tag[:, 1].mean(), (0.7, 0.0, 0.08), a, anchor="cc", rot=sw - 0.15)


def draw_icon(L, kind, cx, cy, s, a):
    if kind == "tear":
        th = np.linspace(0, 2 * math.pi, 40)
        x = np.sin(th) * (1 - np.cos(th)) * 0.5 * s
        y = -np.cos(th) * s
        L.fill(np.stack([cx + x, cy + y * 0.9], 1), (0.75, 0.85, 1.0), a, glow=0.8)
    elif kind == "kiss":
        th = np.linspace(0, math.pi, 20)
        top = np.stack([cx + np.cos(th) * s, cy - np.abs(np.sin(th * 2)) * s * 0.35], 1)
        bot = np.stack([cx + np.cos(th[::-1]) * s, cy + np.sin(th[::-1]) * s * 0.45], 1)
        L.fill(np.concatenate([top, bot]), (0.9, 0.05, 0.15), a, glow=0.9)
        L.line(np.array([[cx - s, cy], [cx, cy + 4], [cx + s, cy]]), (0.3, 0, 0.03), a, 2)
    else:
        th = np.linspace(0, 2 * math.pi, 60)
        x = 16 * np.sin(th) ** 3
        y = -(13 * np.cos(th) - 5 * np.cos(2 * th) - 2 * np.cos(3 * th) - np.cos(4 * th))
        P = np.stack([x, y], 1) * s / 17
        left = P[(P[:, 0] <= 0)]
        right = P[(P[:, 0] >= 0)]
        crack = np.array([[0, -s * 0.45], [-s * 0.12, -s * 0.1], [s * 0.1, s * 0.15], [0, s * 0.9]])
        L.fill(np.concatenate([left, crack[::-1]]) + [cx - 10, cy], (0.85, 0.02, 0.1), a, glow=0.7)
        L.fill(np.concatenate([right, crack]) + [cx + 10, cy], (0.85, 0.02, 0.1), a, glow=0.7)


def cards3(L, t, times, a=1.0):
    spec = [("tear", "第一滴泪", "VALUE  ∞"), ("kiss", "第一次拥吻", "VALUE  0"), ("heart", "第一次心碎", "VALUE  NaN")]
    for i, (kind, name, val) in enumerate(spec):
        dt = t - times[i]
        if dt < 0:
            continue
        sx = sm(dt / 0.18)
        cx, cy, w, h = 1000 + i * 340, 600, 290, 420
        x0 = cx - w / 2 * sx
        x1 = cx + w / 2 * sx
        L.fill(np.array([[x0, cy - h / 2], [x1, cy - h / 2], [x1, cy + h / 2], [x0, cy + h / 2]]), (0.12, 0.0, 0.02), a)
        L.line(np.array([[x0 + 10, cy - h / 2 + 10], [x1 - 10, cy - h / 2 + 10], [x1 - 10, cy + h / 2 - 10], [x0 + 10, cy + h / 2 - 10]]),
               GOLD, a, 2, glow=0.6, closed=True)
        if sx > 0.9:
            draw_icon(L, kind, cx, cy - 50, 70, a)
            text(L, name, "serif", 34, cx, cy + 90, WHITE, a, anchor="cc")
            text(L, val, "anton", 40, cx, cy + 150, GOLD, a, anchor="cc", glow=0.5)
            text(L, f"No.{i + 1:03d}", "mono", 18, x0 + 24, cy - h / 2 + 24, GOLD, a * 0.8)


def anubis_scale(L, t, lt, a=1.0):
    cx, top = 1330, 260
    tilt = 0.32 * math.sin(lt * 2.2) * math.exp(-lt * 0.25) + 0.08 * math.sin(t * 1.3)
    L.line(np.array([[cx, top], [cx, 900]]), GOLD, a, 6, glow=0.6)
    L.line(np.array([[cx - 160, 900], [cx + 160, 900]]), GOLD, a, 8, glow=0.6)
    text(L, "𓁢", "egy", 150, cx, 900, GOLD, a, anchor="cb", glow=0.6)
    arm = 330
    c, s = math.cos(tilt), math.sin(tilt)
    pl, pr = np.array([cx - arm * c, top - arm * s]), np.array([cx + arm * c, top + arm * s])
    L.line(np.array([pl, pr]), GOLD, a, 6, glow=0.8)
    L.circle((cx, top), 14, GOLD, a, glow=1)
    for p, glyph, gc in ((pl, "𓄣", (0.95, 0.1, 0.15)), (pr, "𓆄", WHITE)):
        pan_y = p[1] + 220
        L.lines([np.array([p, [p[0] - 90, pan_y]]), np.array([p, [p[0] + 90, pan_y]])], GOLD, a * 0.8, 2)
        arc = np.stack([p[0] + np.linspace(-110, 110, 20), pan_y + 30 * np.sin(np.linspace(0, math.pi, 20))], 1)
        L.line(arc, GOLD, a, 4, glow=0.6)
        text(L, glyph, "egy", 110, p[0], pan_y - 6, gc, a, anchor="cb", glow=0.8)


def equations(L, t, lt, a=1.0):
    k = a * sst(0.1, 0.4, lt)
    text(L, "情感  ⇌  劳动", "serif", 64, 120, 760, WHITE, k, glow=0.3)
    text(L, "W = c + v + m", "cinzel", 44, 120, 860, GOLD, k * (0.7 + 0.3 * math.sin(t * 9)))
    xs = [(t * 13.7 + i * 3.1) % 10 for i in range(5)]
    text(L, "x̄ = " + " ".join(f"{x:.1f}" for x in xs) + f"  → {sum(xs) / 5:.3f}", "mono", 26, 120, 930, (0.9, 0.6, 0.55), k)


def card(L, cx, cy, w, h, rank, suit, a, face=True, red=True, center=None, sx=1.0):
    x0, x1 = cx - w / 2 * sx, cx + w / 2 * sx
    P = np.array([[x0, cy - h / 2], [x1, cy - h / 2], [x1, cy + h / 2], [x0, cy + h / 2]])
    if not face:
        L.fill(P, (0.45, 0.0, 0.05), a)
        L.line(P + [[6, 6], [-6, 6], [-6, -6], [6, -6]], GOLD, a, 2, closed=True)
        return
    L.fill(P, (0.97, 0.95, 0.92), a)
    col = (0.85, 0.02, 0.1) if red else (0.05, 0.05, 0.05)
    if sx > 0.6:
        text(L, rank, "anton", 48, x0 + 18, cy - h / 2 + 12, col, a)
        if center:
            text(L, center, "serif", int(130 * sx), cx, cy, col, a, anchor="cc", glow=0.5)
        else:
            text(L, suit, "serif", 120, cx, cy, col, a, anchor="cc")


def poker(L, t, lt, t_river, a=1.0):
    board = [("A", "♥"), ("K", "♥"), ("Q", "♥"), ("J", "♥")]
    for i, (r, s) in enumerate(board):
        dt = lt - i * 0.3
        if dt < 0:
            continue
        card(L, 820 + i * 220, 600, 180, 260, r, s, a * sm(dt / 0.15))
    dt = t - t_river
    cx = 820 + 4 * 220
    if dt < 0:
        card(L, cx, 600, 180, 260, "", "", a * sst(1.2, 1.4, lt), face=False)
    else:
        sx = abs(math.cos(min(1.0, dt / 0.12) * math.pi / 2 + math.pi / 2)) if dt < 0.12 else 1.0
        sc = 1 + 0.6 * math.exp(-dt / 0.15)
        card(L, cx, 600, 180 * sc, 260 * sc, "", "", a, center="红", sx=max(sx, 0.05))


# ================================================================= S5
def palm(L, x, y, hgt, lean, t, a=1.0, fingers=0.0):
    s = np.linspace(0, 1, 20)
    trunk = np.stack([x + lean * hgt * s ** 2, y - hgt * s], 1)
    L.line(trunk, DARK, a, 16)
    top = trunk[-1]
    rr = np.random.default_rng(int(x))
    fr = []
    for k in range(9):
        ang = -math.pi / 2 + (k - 4) * 0.42 + 0.05 * math.sin(t * 1.2 + k)
        ln = rr.uniform(200, 300)
        pts = [top]
        p = top.copy()
        a_ = ang
        for j in range(12):
            a_ += 0.11 * (1 if math.cos(ang) > 0 else -1)
            p = p + np.array([math.cos(a_), math.sin(a_)]) * ln / 12
            pts.append(p)
        pts = np.array(pts)
        fr.append(pts)
        mid = pts[6]
        for side in (-0.6, 0.6):
            fa = a_ + side
            fr.append(np.array([mid, mid + np.array([math.cos(fa), math.sin(fa)]) * (90 + 120 * fingers)]))
            q = pts[9]
            fr.append(np.array([q, q + np.array([math.cos(fa + side * 0.4), math.sin(fa + side * 0.4)]) * 60]))
    L.lines(fr, DARK, a, 5)


def palms(L, t, lt, a=1.0, fingers=0.0):
    palm(L, 230, 1100, 640, 0.25, t, a, fingers)
    palm(L, 1650, 1100, 720, -0.2, t + 1, a, fingers)
    palm(L, 1880, 1100, 480, -0.35, t + 2, a, fingers)


_rv = np.random.default_rng(12)
SUN_SHARDS = []
_cent = _rv.uniform(-1, 1, (16, 2))
for i in range(16):
    ang = np.linspace(0, 2 * math.pi, 9)[:-1] + _rv.uniform(0, 0.4)
    SUN_SHARDS.append(dict(c=_cent[i] * 70, r=_rv.uniform(28, 48), ang=ang, v=_rv.uniform(300, 900), rot=_rv.uniform(-6, 6)))


def sun_shards(L, t, t_break, cx=1257, cy=475, a=1.0):
    dt = t - t_break
    if dt < 0:
        return
    for s in SUN_SHARDS:
        d = s["c"] / (np.linalg.norm(s["c"]) + 1e-6)
        pos = np.array([cx, cy]) + s["c"] + d * s["v"] * dt + [0, 300 * dt * dt]
        rr = s["rot"] * dt
        poly = np.stack([np.cos(s["ang"] + rr) * s["r"], np.sin(s["ang"] + rr) * s["r"] * 0.6], 1)[::2] + pos
        L.fill(poly, (1, 0.35, 0.1), a * max(0, 1 - dt / 2.5), glow=1.4)
        L.line(poly, (1, 0.85, 0.6), a * max(0, 1 - dt / 2.5), 1, glow=1.5, closed=True)


def gap(L, t, lt, a=1.0):
    w = 4 + 30 * sst(0.3, 2.0, lt)
    ys = np.linspace(-10, H + 10, 40)
    xs = 960 + 14 * np.sin(ys * 0.02 + t)
    for k, (al, th) in enumerate([(0.15, w * 3), (0.5, w), (1.0, max(2, w * 0.3))]):
        L.line(np.stack([xs, ys], 1), (1, 0.8, 0.6), a * al, th, glow=0.8 if k == 2 else 0.2)


def gen_veins():
    r = np.random.default_rng(21)
    segs = []

    def grow(x, y, ang, t0, depth, ln):
        pts = [(x, y, t0)]
        s = 0
        while s < ln:
            ang += r.normal(0, 0.15)
            x += math.cos(ang) * 8
            y += math.sin(ang) * 8
            s += 8
            tt = t0 + s / 650
            pts.append((x, y, tt))
            if depth < 4 and r.random() < 0.06:
                grow(x, y, ang + r.choice([-1, 1]) * r.uniform(0.4, 0.9), tt, depth + 1, ln * 0.55)
        segs.append((np.array(pts), depth))

    for k in range(10):
        ang = k * 2 * math.pi / 10 + r.normal(0, 0.1)
        grow(960 + math.cos(ang) * 1100, 540 + math.sin(ang) * 650, ang + math.pi, r.uniform(0, 0.4), 0, 900)
    return segs


VEINS = gen_veins()


def veins(L, t, lt, a=1.0):
    pl = []
    for pts, depth in VEINS:
        m = pts[:, 2] <= lt
        if m.sum() > 1:
            pl.append((pts[m][:, :2], depth))
    for depth in range(5):
        P = [p for p, d in pl if d == depth]
        L.lines(P, (0.85, 0.02, 0.08), a * 0.8, max(1, 3 - depth), glow=0.45)


# ================================================================= S6
LANE_X0, LANE_W, JUDGE_Y, SPEED = 1200, 120, 900, 1400


def rhythm_ui(L, t, a=1.0):
    from common import CHART, MISS_BAR
    t_miss = bar(MISS_BAR)
    if t >= t_miss + 0.8:
        return
    broken = t >= t_miss
    k = a * (1 - sst(t_miss, t_miss + 0.8, t)) if broken else a
    shake = (np.random.default_rng(int(t * 60)).normal(0, 16, 2) if broken else np.zeros(2))
    x0 = LANE_X0 + shake[0]
    L.fill(np.array([[x0, 0], [x0 + 4 * LANE_W, 0], [x0 + 4 * LANE_W, H], [x0, H]]), (0.0, 0.0, 0.0), 0.45 * k)
    L.lines([np.array([[x0 + i * LANE_W, 0], [x0 + i * LANE_W, H]]) for i in range(5)], (1, 0.8, 0.8), 0.3 * k, 1)
    L.line(np.array([[x0 - 10, JUDGE_Y], [x0 + 4 * LANE_W + 10, JUDGE_Y]]), WHITE, k, 3, glow=1.0)
    combo = 0
    last_hit = None
    for (tn, m, lane) in CHART:
        if tn <= t:
            combo += 1
            last_hit = (tn, lane)
            dt = t - tn
            if dt < 0.22 and not broken:
                lx = x0 + lane * LANE_W
                L.fill(np.array([[lx, 0], [lx + LANE_W, 0], [lx + LANE_W, JUDGE_Y], [lx, JUDGE_Y]]), (1, 0.2, 0.25), 0.25 * k * (1 - dt / 0.22), glow=0.3 * (1 - dt / 0.22))
                L.circle((lx + LANE_W / 2, JUDGE_Y), 20 + dt * 400, (1, 0.6, 0.6), k * (1 - dt / 0.22), th=3, glow=1.5)
            continue
        y = JUDGE_Y - (tn - t) * SPEED
        if y < -40:
            break
        if broken:
            continue
        lx = x0 + lane * LANE_W
        L.fill(np.array([[lx + 6, y - 12], [lx + LANE_W - 6, y - 12], [lx + LANE_W - 6, y + 12], [lx + 6, y + 12]]), WHITE, k, glow=0.6)
        L.line(np.array([[lx + 6, y + 12], [lx + LANE_W - 6, y + 12]]), (1, 0.1, 0.2), k, 4, glow=1.2)
    cx = x0 + 2 * LANE_W
    if broken:
        dt = t - t_miss
        text(L, "MISS", "anton", 260, cx, 420, (1, 0.08, 0.12), min(1, 3 * (1 - dt / 0.8)), anchor="cc", glow=1.4,
             scale=1 + 0.4 * math.exp(-dt / 0.08))
        return
    if combo:
        val = "∞" if t >= bar(MISS_BAR - 1) else f"{combo * 7}"
        pulse = 1 + 0.12 * math.exp(-(t - last_hit[0]) / 0.06) if last_hit else 1
        text(L, "COMBO", "anton", 40, cx, 250, GOLD, k * 0.9, anchor="cc")
        text(L, val, "anton" if val != "∞" else "serif", 150, cx, 360, WHITE, k * 0.85, anchor="cc", glow=0.5, scale=pulse)
        if last_hit and t - last_hit[0] < 0.25:
            dt = t - last_hit[0]
            text(L, "PERFECT", "anton", 52, cx, JUDGE_Y - 90 - dt * 80, GOLD, k * (1 - dt / 0.25), anchor="cc", glow=1.0)
    score = combo * 2333
    text(L, "SCORE", "anton", 26, 1860, 40, GOLD, k, anchor="rt")
    text(L, f"{score:010d}", "anton", 54, 1860, 72, WHITE, k, anchor="rt")
    text(L, "红色  ·  Apokal-Panra", "serif_sb", 26, 60, 40, WHITE, k * 0.9)
    text(L, "LV.15  ★★★★★★★★★★★★★★★", "serif_sb", 20, 60, 80, GOLD, k * 0.8)


def monk(L, t, lt, a=1.0):
    x = 1700 - lt * 120
    y = 720 + 4 * abs(math.sin(lt * 4))
    L.fill(np.array([[x - 26, y], [x + 26, y], [x + 12, y - 90], [x - 12, y - 90]]), DARK, a)
    L.circle((x, y - 104), 14, DARK, a)
    L.line(np.array([[x - 34, y + 4], [x - 44, y - 140]]), DARK, a, 4)
    L.circle((x - 44, y - 146), 7, GOLD, a * 0.7, th=2)
    L.line(np.array([[0, y + 2], [W, y + 2]]), DARK, a * 0.5, 2)


MANTRA = "色即是空空即是色超越性观点诵读静默取经无我无相无住幸福"


def mandala(L, t, lt, a=1.0, cx=600, cy=560):
    k = sst(0, 0.8, lt) * a
    for ring in range(5):
        R = 110 + ring * 85
        n = int(2 * math.pi * R / 50)
        rot = t * (0.25 + 0.1 * ring) * (1 if ring % 2 else -1)
        for i in range(n):
            ang = rot + i * 2 * math.pi / n
            ch = MANTRA[(i + ring * 7) % len(MANTRA)]
            text(L, ch, "serif", 34 - ring * 2, cx + R * math.cos(ang), cy + R * math.sin(ang),
                 GOLD if ring % 2 == 0 else (0.95, 0.4, 0.3), k * (0.9 - ring * 0.1), anchor="cc", rot=ang + math.pi / 2,
                 glow=0.3 if ring == 0 else 0)
        L.circle((cx, cy), R + 42, GOLD, k * 0.4, th=1)
    text(L, "超", "serif", 140, cx, cy, WHITE, k, anchor="cc", glow=0.8)


def wadang(L, t, lt, a=1.0, cx=600, cy=560, R=230):
    k = sst(0, 0.6, lt) * a
    rot = 0.05 * math.sin(t)
    L.circle((cx, cy), R, (0.45, 0.18, 0.1), k * 0.95)
    L.circle((cx, cy), R, GOLD, k, th=6, glow=0.6)
    L.circle((cx, cy), R - 34, GOLD, k, th=2)
    L.circle((cx, cy), 46, GOLD, k, th=3)
    for q in range(4):
        ang = rot + q * math.pi / 2
        L.line(np.array([[cx + 46 * math.cos(ang), cy + 46 * math.sin(ang)], [cx + (R - 34) * math.cos(ang), cy + (R - 34) * math.sin(ang)]]), GOLD, k, 3)
    # 长乐未央 — read top, bottom, right, left; the 长 is missing
    pos = {"乐": math.pi / 4 * 7, "未": math.pi / 4, "央": math.pi / 4 * 3}
    for ch, ang in pos.items():
        text(L, ch, "serif", 110, cx + 0.55 * R * math.cos(ang + rot), cy + 0.55 * R * math.sin(ang + rot), (0.98, 0.85, 0.6), k,
             anchor="cc", rot=rot, glow=0.5)


def burning_verses(L, t, lt, a=1.0):
    verses = ["玫瑰是红的", "我爱你如玫瑰", "rose is a rose is a rose", "一朵玫瑰的名字", "O Rose, thou art sick", "玫瑰与诗"]
    for i, v in enumerate(verses):
        y = 140 + i * 130
        burn = sst(0.4 + i * 0.25, 1.6 + i * 0.25, lt)
        x = 100 + (i % 2) * 60
        w, h = text(L, v, "serif", 54, x, y, WHITE, a * (1 - burn) * 0.85)
        if 0 < burn < 1:
            ex = x + w * (1 - burn)
            L.line(np.array([[ex, y], [ex, y + h]]), (1, 0.6, 0.2), a, 4, glow=2.0)


CAFF_ATOMS = {
    "N1": (0, -1.0), "C2": (0.866, -0.5), "N3": (0.866, 0.5), "C4": (0, 1.0), "C5": (-0.866, 0.5), "C6": (-0.866, -0.5),
    "N7": (-1.85, 0.85), "C8": (-2.45, 0.0), "N9": (-1.85, -0.85), "O2": (1.73, -1.0), "O6": (-0.866, -1.5) ,
    "Me1": (0, -2.0), "Me3": (1.73, 1.0), "Me7": (-2.3, 1.85),
}
CAFF_BONDS = [("N1", "C2"), ("C2", "N3"), ("N3", "C4"), ("C4", "C5"), ("C5", "C6"), ("C6", "N1"), ("C5", "N7"), ("N7", "C8"),
              ("C8", "N9"), ("N9", "C4"), ("C2", "O2"), ("C6", "O6"), ("N1", "Me1"), ("N3", "Me3"), ("N7", "Me7")]
CAFF_DOUBLE = [("C2", "O2"), ("C6", "O6"), ("C4", "C5"), ("C8", "N9")]


def caffeine(L, t, lt, a=1.0, cx=760, cy=740, s=105):
    k = a
    P = {n: np.array([cx + x * s, cy + y * s]) for n, (x, y) in CAFF_ATOMS.items()}
    grow = sst(0, 1.6, lt)
    for i, (p, q) in enumerate(CAFF_BONDS):
        f = sm(grow * len(CAFF_BONDS) - i)
        if f <= 0:
            continue
        A, B = P[p], P[q]
        B2 = A + (B - A) * f
        L.line(np.array([A, B2]), WHITE, k, 3, glow=0.8)
        if (p, q) in CAFF_DOUBLE and f > 0.99:
            n = np.array([-(B - A)[1], (B - A)[0]]) / np.linalg.norm(B - A) * 9
            L.line(np.array([A + n + (B - A) * 0.15, B - (B - A) * 0.15 + n]), WHITE, k, 2, glow=0.5)
    for n, p in P.items():
        lab = {"N": "N", "O": "O"}.get(n[0])
        if n.startswith("Me"):
            lab = "CH3"
        if lab and grow > 0.5:
            L.circle(p, 26, (0.12, 0.02, 0.0), k)
            text(L, lab, "cinzel", 34, p[0], p[1], (1, 0.75, 0.5) if lab == "O" else WHITE, k, anchor="cc", glow=0.4)
    text(L, "C8 H10 N4 O2", "cinzel", 60, cx, cy + 3.0 * s, GOLD, k * sst(1.5, 2.0, lt), anchor="cc", glow=0.6)
    for i in range(8):
        ph = (lt * 1.4 + i / 8) % 1.0
        L.circle((cx - 3.7 * s, 120 + ph * 800), 7, (0.35, 0.15, 0.05), k * (1 - ph), glow=0.3)


def birds_branch(L, t, lt, t_fall, a=1.0):
    dt = max(0.0, t - t_fall)
    ang = 0.6 * sm(dt / 0.6) + 0.5 * dt * dt
    pivot = np.array([1450.0, 380.0])
    base = np.array([[1950, 330], [1450, 380]])
    twig = np.array([[1450, 380], [1100, 430], [800, 520]])
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, -s], [s, c]])
    twig2 = (twig - pivot) @ R.T + pivot + [0, 200 * dt * dt]
    L.line(base, DARK, a, 22)
    L.line(twig2, DARK, a, 14)
    L.lines([np.array([twig2[1], twig2[1] + (np.array([-80, -90]) @ R.T)]), np.array([twig2[2], twig2[2] + (np.array([-60, 70]) @ R.T)])], DARK, a, 6)
    for i in range(5):
        u = 0.15 + i * 0.18
        p = twig2[0] + (twig2[2] - twig2[0]) * u
        bow = 0.9 * sm((lt - 0.5 - i * 0.1) / 0.4)
        body = cv2.ellipse2Poly((0, 0), (26, 15), 0, 0, 360, 20).astype(float)
        L.fill(body + p + [0, -22], DARK, a)
        hx, hy = p[0] - 22 - 10 * bow, p[1] - 40 + 20 * bow
        L.circle((hx, hy), 11, DARK, a)
        L.fill(np.array([[hx - 10, hy], [hx - 24, hy + 4 + 4 * bow], [hx - 9, hy + 5]]), GOLD, a * 0.8)
        L.fill(np.array([p + [22, -26], p + [56, -40], p + [52, -18]]), DARK, a)


def swallows(L, t, lt, a=1.0):
    for i in range(13):
        row = abs(i - 6)
        x = -200 + lt * 520 - row * 90 + 30 * math.sin(i)
        y = 260 + row * 30 * (1 if i < 6 else 1) + (i - 6) * 8 + 10 * math.sin(lt * 3 + i)
        flap = math.sin(lt * 14 + i)
        s = 1.0 - row * 0.05
        P = np.array([[x, y], [x - 30 * s, y - 22 * s * flap], [x - 50 * s, y - 10 * s * flap], [x - 6, y + 2],
                      [x - 46 * s, y + 18 * s], [x - 22 * s, y + 6], [x - 50 * s, y + 28 * s], [x - 12, y + 6], [x + 14 * s, y + 2]])
        L.fill(P, DARK, a)


ROSE_IMG = None


def _rose_img():
    global ROSE_IMG
    if ROSE_IMG is None:
        im = np.zeros((600, 600, 4), np.uint8)
        for k, (R, col) in enumerate([(270, (150, 0, 20)), (210, (200, 10, 35)), (150, (230, 30, 55)), (90, (255, 70, 90))]):
            P = rose_curve(300, 300, R, k * 0.7, k=7 / 3, n=700)
            cv2.fillPoly(im, [np.round(P * 16).astype(np.int32)], col + (255,), cv2.LINE_AA, 4)
            cv2.polylines(im, [np.round(P * 16).astype(np.int32)], True, (255, 170, 170, 255), 2, cv2.LINE_AA, 4)
        ROSE_IMG = im
    return ROSE_IMG


_rr = np.random.default_rng(31)
ROSE_SITES = _rr.uniform(0, 600, (28, 2))


def rose_shatter(L, t, t_smash, a=1.0, cx=1300, cy=540):
    img = _rose_img()
    dt = t - t_smash
    if dt < 0:
        h, w = img.shape[:2]
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        reg = L.over[y0:y0 + h, x0:x0 + w].astype(np.float32)
        src = img.astype(np.float32) * a
        L.over[y0:y0 + h, x0:x0 + w] = np.clip(reg * (1 - src[..., 3:4] / 255) + src, 0, 255).astype(np.uint8)
        L.used = True
        return
    yy, xx = np.mgrid[0:600, 0:600]
    for i, s in enumerate(ROSE_SITES[:28]):
        d = s - 300
        dirn = d / (np.linalg.norm(d) + 1e-6)
        v = 500 + 300 * (i % 3)
        off = dirn * v * dt + [0, 400 * dt * dt]
        ang = (i % 5 - 2) * 1.6 * dt
        # cell = pixels nearest to this site (approximated by a disk)
        pts = cv2.ellipse2Poly((int(s[0]), int(s[1])), (60, 60), 0, 0, 360, 45).astype(float)
        c, sn = math.cos(ang), math.sin(ang)
        poly = (pts - s) @ np.array([[c, sn], [-sn, c]]) + s + off + [cx - 300, cy - 300]
        col = img[int(s[1]), int(s[0])]
        if col[3] < 10:
            continue
        L.fill(poly, (col[0] / 255, col[1] / 255, col[2] / 255), a * max(0, 1 - dt / 2.0))
        L.line(poly, (1, 0.8, 0.8), a * max(0, 1 - dt / 2.0) * 0.8, 1, glow=1.0, closed=True)
        if i % 4 == 0:
            text(L, "我", "serif", 30, poly[:, 0].mean(), poly[:, 1].mean(), WHITE, a * 0.6 * max(0, 1 - dt / 2.0), anchor="cc", rot=ang)


PROFILE = np.array([[0, -260], [40, -250], [70, -215], [82, -170], [80, -140], [96, -110], [86, -100], [92, -84], [82, -76],
                    [86, -60], [70, -46], [52, -40], [46, -10], [60, 60], [130, 120], [150, 260], [-160, 260], [-120, 120],
                    [-60, 40], [-70, -40], [-110, -60], [-120, -140], [-100, -220], [-50, -262]], float)


def silhouette(L, t, lt, a=1.0, cx=1250, cy=560):
    k = sst(0.2, 1.5, lt) * a
    P = PROFILE * 1.5 + [cx, cy + 40 * math.sin(t * 0.4)]
    L.fill(P, (0.02, 0.0, 0.01), k * 0.88)
    L.line(P, (1, 0.25, 0.25), k, 2, glow=1.4, closed=True)


_rb = np.random.default_rng(41)
STROKES = [(_rb.uniform(-300, 400, 4), _rb.uniform(100, 1000, 4), _rb.uniform(0.3, 2.8), [(0.9, 0.03, 0.1), WHITE, DARK][i % 3], _rb.uniform(50, 120))
           for i in range(8)]


def brush(L, t, lt, a=1.0):
    for i, (xs, ys, t0, col, wid) in enumerate(STROKES):
        p = sst(t0, t0 + 0.45, lt)
        if p <= 0:
            continue
        s = np.linspace(0, p, 60)
        bx = (1 - s) ** 3 * xs[0] + 3 * (1 - s) ** 2 * s * (xs[1] + 600) + 3 * (1 - s) * s ** 2 * (xs[2] + 1200) + s ** 3 * (xs[3] + 1800)
        by = (1 - s) ** 3 * ys[0] + 3 * (1 - s) ** 2 * s * ys[1] + 3 * (1 - s) * s ** 2 * ys[2] + s ** 3 * ys[3]
        P = np.stack([bx, by], 1)
        bristles = []
        rr = np.random.default_rng(i)
        for b in range(18):
            off = (b / 17 - 0.5) * wid
            jit = rr.normal(0, 3, len(P))
            bristles.append(P + np.stack([jit * 0.3, off + jit], 1))
        L.lines(bristles, col, a * 0.85, 4, glow=0.3 if col != DARK else 0)


KNIGHT = np.array([[-70, 150], [75, 150], [75, 128], [55, 120], [58, 95], [40, 85], [52, 40], [60, 0], [58, -40], [45, -80],
                   [20, -110], [0, -125], [-15, -150], [-25, -120], [-45, -110], [-70, -85], [-95, -55], [-110, -25],
                   [-100, -10], [-80, -12], [-60, -25], [-40, -20], [-30, -5], [-50, 30], [-60, 60], [-50, 85], [-60, 95],
                   [-58, 120], [-75, 128]], float)


def knights(L, t, lt, a=1.0, cx=620, cy=600):
    z = lt * 0.55
    for k in range(8, -1, -1):
        e = k - (z % 1.0)
        s = 0.18 * 1.75 ** e
        if s < 0.05 or s > 9:
            continue
        P = KNIGHT * s + [cx, cy]
        fade = min(1, s / 0.15) * (1 - sst(4.5, 9, s))
        if s < 2.2:
            L.fill(P, (0.9, 0.05, 0.1) if (k + int(z)) % 2 else (0.04, 0.0, 0.02), a * fade)
        L.line(P, GOLD, a * fade, 2, glow=0.9, closed=True)


def eiffel(L, t, lt, a=1.0, x=1560, y=820, s=0.55):
    k = a * (1 - sst(2.5, 6.0, lt))
    P = [np.array([[x - 90 * s, y], [x - 20 * s, y - 300 * s], [x, y - 520 * s], [x + 20 * s, y - 300 * s], [x + 90 * s, y]]),
         np.array([[x - 60 * s, y - 110 * s], [x + 60 * s, y - 110 * s]]), np.array([[x - 32 * s, y - 230 * s], [x + 32 * s, y - 230 * s]]),
         np.array([[x - 60 * s, y], [x, y - 70 * s], [x + 60 * s, y]])]
    L.lines(P, (1, 0.9, 0.85), k * 0.6, 2, glow=0.3)
    text(L, "PARIS  ✕", "cinzel", 24, x, y + 30, (1, 0.9, 0.85), k * 0.5, anchor="ct")


def naive(L, t, lt, a=1.0):
    k = sst(0.1, 0.6, lt) * a
    text(L, "naiv", "frak", 170, 1180, 360, WHITE, k, anchor="cc", glow=0.4)
    text(L, "≠", "cinzel", 120, 1180, 560, (1, 0.2, 0.25), k, anchor="cc", glow=0.8)
    text(L, "sentimentalisch", "frak", 110, 1180, 740, (1, 0.85, 0.8), k * 0.9, anchor="cc")
    text(L, "— Schiller, 1795", "cinzel", 26, 1180, 860, GOLD, k * 0.7, anchor="cc")


def crack_texture(cx=1100, cy=430, seed=3):
    r = np.random.default_rng(seed)
    im = np.zeros((H, W), np.uint8)
    for k in range(18):
        ang = k * 2 * math.pi / 18 + r.normal(0, 0.12)
        x, y = cx, cy
        pts = [(x, y)]
        ln = r.uniform(500, 1400)
        s = 0
        while s < ln:
            ang += r.normal(0, 0.1)
            st = r.uniform(20, 60)
            x += math.cos(ang) * st
            y += math.sin(ang) * st
            s += st
            pts.append((x, y))
        cv2.polylines(im, [np.round(np.array(pts) * 16).astype(np.int32)], False, 255, 1, cv2.LINE_AA, 4)
    for rad in (90, 190, 330):
        pts = []
        for a in np.linspace(0, 2 * math.pi, 50):
            rr = rad * (1 + r.normal(0, 0.06))
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        for i in range(0, 50, 2):
            seg = np.array(pts[i:i + 2])
            if len(seg) == 2:
                cv2.polylines(im, [np.round(seg * 16).astype(np.int32)], False, 255, 1, cv2.LINE_AA, 4)
    scratch = np.array([[200, 820], [700, 560], [1300, 300], [1800, 120]], float)
    cv2.polylines(im, [np.round(scratch * 16).astype(np.int32)], False, 255, 2, cv2.LINE_AA, 4)
    return im.astype(np.float32) / 255
