"""Kinetic typography for 《红色》: lines slam in character by character on the 16th-note grid;
keywords burn red; giant poster words land behind them."""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import *

F_LINE = ImageFont.truetype(f"{FONT_DIR}/NotoSerifSC-Black.otf", 54)
F_SOLO = ImageFont.truetype(f"{FONT_DIR}/NotoSerifSC-Black.otf", 78)
F_POST_CJK = ImageFont.truetype(f"{FONT_DIR}/NotoSerifSC-Black.otf", 560)
F_POST_LAT = ImageFont.truetype(f"{FONT_DIR}/Anton.ttf", 300)
F_TITLE = ImageFont.truetype(f"{FONT_DIR}/NotoSerifSC-Black.otf", 600)
F_SIG = ImageFont.truetype(f"{FONT_DIR}/Unifraktur.ttf", 190)
F_EGY = ImageFont.truetype(f"{FONT_DIR}/NotoSansEgypt.ttf", 300)

CREAM = np.array([0.97, 0.94, 0.90], np.float32)
KRED = np.array([1.0, 0.13, 0.2], np.float32)
GHOST = np.array([1.0, 0.0, 0.08], np.float32)
_G = {}


def glyph(s, f, stroke=0):
    k = (s, id(f), stroke)
    if k not in _G:
        l, t, r, b = f.getbbox(s, stroke_width=stroke)
        asc, desc = f.getmetrics()
        pad = 14 + stroke
        if len(s) == 1 and f in (F_LINE, F_SOLO):
            w = int(math.ceil(f.getlength(s))) + 2 * pad
            im = Image.new("L", (w, asc + desc + 2 * pad), 0)
            ImageDraw.Draw(im).text((pad, pad), s, font=f, fill=255)
        else:
            im = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
            ImageDraw.Draw(im).text((pad - l, pad - t), s, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
        a = np.asarray(im, np.float32) / 255
        sh = cv2.GaussianBlur(a, (0, 0), 4.0)
        _G[k] = (a, sh, pad)
    return _G[k]


def outline(s, f, w=5):
    k = (s, id(f), "outline", w)
    if k not in _G:
        a_st = glyph(s, f, stroke=w)[0]
        l, t, r, b = f.getbbox(s, stroke_width=w)
        pad = 14 + w
        im = Image.new("L", a_st.shape[::-1], 0)
        ImageDraw.Draw(im).text((pad - l, pad - t), s, font=f, fill=255)
        a_f = np.asarray(im, np.float32) / 255
        _G[k] = (np.clip(a_st - a_f, 0, 1), None, pad)
    return _G[k]


def blit(T, a, x, y, col, op, add=False):
    h, w = a.shape
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1 or op <= 0.002:
        return
    sub = a[y0 - y:y1 - y, x0 - x:x1 - x, None] * op
    reg = T[y0:y1, x0:x1]
    if add:
        reg[..., :3] += sub * col
    else:
        reg *= 1 - sub
        reg[..., :3] += sub * col
        reg[..., 3:] += sub


def scaled(a, s):
    if abs(s - 1) < 1e-3:
        return a
    return cv2.resize(a, (max(1, int(a.shape[1] * s)), max(1, int(a.shape[0] * s))), interpolation=cv2.INTER_LINEAR)


def draw_char(T, ch, x, y, age, col, op, f=F_LINE, shadow=0.65):
    a, sh, pad = glyph(ch, f)
    s = 1 + 0.85 * math.exp(-age / 0.045) if age < 0.3 else 1.0
    o = op * min(1.0, age / 0.03)
    if s != 1.0:
        a2, sh2 = scaled(a, s), scaled(sh, s)
        dx, dy = (a2.shape[1] - a.shape[1]) / 2, (a2.shape[0] - a.shape[0]) / 2
    else:
        a2, sh2, dx, dy = a, sh, 0, 0
    px, py = x - pad - dx, y - pad - dy
    if shadow:
        blit(T, sh2, px + 3, py + 5, np.zeros(3, np.float32), shadow * o)
    g = math.exp(-age / 0.07) if age < 0.4 else 0.0
    if g > 0.01:
        blit(T, a2, px + 9 * g, py + 4 * g, GHOST * 1.4, o * g, add=True)
        blit(T, a2, px - 6 * g, py - 3 * g, np.array([0.2, 0.6, 1.0], np.float32), o * g * 0.5, add=True)
    blit(T, a2, px, py, col, o)


def line_layout(si, li, mode):
    n = len(STANZAS[si])
    if mode == "lyric":
        return 90, 470
    if mode == "stack7":
        return 160, 540 - n * 84 / 2 + li * 84
    return 110, 540 - n * 84 / 2 + li * 84


def stanza_mode(si):
    return {5: "lyric", 6: "stack7"}.get(si, "stack")


def stanza_end(si):
    ends = [SECTION_BAR["S2"], SECTION_BAR["S3"], SECTION_BAR["S4"], RIVER_BAR + 1.5, SECTION_BAR["S6"], MISS_BAR, 180]
    return bar(ends[si])


REDACT = (2, 4, STANZAS[2][4].find("不允许的举措"), 6)


def draw_lines(T, t):
    for si, st in enumerate(STANZAS):
        t0 = bar(LINE_BAR[si][0])
        t1 = stanza_end(si)
        if not (t0 <= t < t1 + 0.6):
            continue
        fade = 1 - min(1.0, max(0.0, (t - t1) / 0.6))
        mode = stanza_mode(si)
        cur = max(li for li in range(len(st)) if bar(LINE_BAR[si][li]) <= t)
        for li, text in enumerate(st):
            if si == 6 and li == 2:
                continue
            times = CHAR_T[si][li]
            if t < times[0]:
                continue
            if mode == "lyric":
                if li < cur - 1:
                    continue
                op = 1.0 if li == cur else 0.0
                if li == cur - 1:
                    continue
            else:
                op = 1.0 if li == cur else 0.5
            op *= fade
            if si == 6:
                op *= 1 - min(1.0, max(0.0, (t - bar(169.6)) / 0.5))
            x0, y0 = line_layout(si, li, mode)
            kw = KEYWORDS[si][li]
            kpos = text.find(kw)
            x = x0
            for k, ch in enumerate(text):
                adv = F_LINE.getlength(ch)
                if times[k] <= t:
                    col = KRED if kpos >= 0 and kpos <= k < kpos + len(kw) else CREAM
                    draw_char(T, ch, x, y0, t - times[k], col, op)
                    if (si, li) == REDACT[:2] and REDACT[2] <= k < REDACT[2] + REDACT[3]:
                        rd = t - times[k] - 0.25
                        if rd > 0:
                            w = adv * min(1.0, rd / 0.08)
                            T[int(y0 + 6):int(y0 + 66), int(x):int(x + w)] = np.array([0.0, 0.0, 0.0, 1.0], np.float32) * op
                x += adv


SOLO_STARTS = [bar(170), bar(172), bar(174), bar(176)]


def draw_solo(T, t):
    text = STANZAS[6][2]
    parts = chunks(text)
    times = CHAR_T[6][2]
    idx = 0
    for j, part in enumerate(parts):
        ts = times[idx:idx + len(part)]
        idx += len(part)
        start = ts[0]
        end = SOLO_STARTS[j + 1] - 0.15 if j + 1 < len(parts) else bar(180)
        if not (start <= t < end + 0.8):
            continue
        op = 1 - min(1.0, max(0.0, (t - end) / 0.8))
        wtot = sum(F_SOLO.getlength(c) for c in part)
        x = 960 - wtot / 2
        for k, ch in enumerate(part):
            if ts[k] <= t:
                col = CREAM if j < 3 else np.array([1.0, 0.92, 0.92], np.float32)
                draw_char(T, ch, x, 500, (t - ts[k]) * 0.6, col, op, f=F_SOLO, shadow=0)
            x += F_SOLO.getlength(ch)


def poster_style(si, li):
    return ["outline", "dark", "red", "ghost"][(si * 3 + li) % 4]


def draw_posters(T, t):
    for si, st in enumerate(STANZAS):
        for li in range(len(st)):
            word = POSTER[si][li]
            if (si == 6 and li == 2) or (si, li) == (1, 1):
                continue
            tw = word_time(si, li, KEYWORDS[si][li])
            tend = bar(LINE_BAR[si][li] + (4 if not (si == 6 and li == 1) else 6))
            if si == 3 and li == 4:
                tw, tend = bar(RIVER_BAR), bar(RIVER_BAR + 2)
            if not (tw <= t < tend):
                continue
            age = t - tw
            cjk = all(ord(c) > 0x2e80 for c in word)
            f = F_POST_CJK if cjk else (F_EGY if word in ("𓄣",) else F_POST_LAT)
            sty = poster_style(si, li)
            if si == 3 and li == 4:
                sty = "white"
            s = (1 + 0.35 * math.exp(-age / 0.06)) * (1 + 0.05 * age)
            op = min(1.0, age / 0.04) * (1 - max(0.0, (t - (tend - 0.25)) / 0.25))
            if sty == "outline":
                a = outline(word, f, 6)[0]
            else:
                a = glyph(word, f)[0]
            maxw = 1500
            base = min(1.0, maxw / a.shape[1])
            a = scaled(a, base * s)
            cx = 1340 if si != 5 else 640
            if si == 3 and li == 4:
                cx = 960
            x, y = cx - a.shape[1] / 2, 540 - a.shape[0] / 2
            col, o = {"outline": (CREAM, 0.75), "dark": (np.zeros(3, np.float32), 0.55), "red": (KRED * 0.9, 0.55),
                      "ghost": (CREAM, 0.16), "white": (CREAM, 0.95)}[sty]
            if si == 5:
                o *= 0.45
            blit(T, a, x, y, np.asarray(col, np.float32), o * op)


def draw_title(T, t):
    t0, t1 = bar(TITLE_BAR), bar(SECTION_BAR["S1"])
    if not (t0 <= t < t1):
        return
    age = t - t0
    a, sh, pad = glyph(TITLE, F_TITLE)
    s = 1 + 0.6 * math.exp(-age / 0.06)
    a2 = scaled(a, s)
    sh2 = scaled(sh, s)
    x, y = 960 - a2.shape[1] / 2, 540 - a2.shape[0] / 2
    blit(T, sh2, x + 8, y + 14, np.zeros(3, np.float32), 0.8)
    blit(T, a2, x + 14 * math.exp(-age / 0.1), y, GHOST * 1.5, math.exp(-age / 0.1), add=True)
    jit = 0.0
    if age > bar(1.5):
        jit = 18 * math.sin(age * 90)
    blit(T, a2, x + jit, y, CREAM, 1.0)


def draw_signature(T, t):
    t0 = bar(181.5)
    if t < t0:
        return
    a, sh, pad = glyph(SIGNATURE, F_SIG)
    reveal = min(1.0, (t - t0) / 2.2)
    w = int(a.shape[1] * reveal)
    if w <= 0:
        return
    end_fade = 1 - min(1.0, max(0.0, (t - bar(188.2)) / 1.6))
    x, y = 960 - a.shape[1] / 2, 820 - a.shape[0] / 2
    sub = a[:, :w]
    gold = np.array([0.95, 0.72, 0.38], np.float32)
    blit(T, sh[:, :w], x + 4, y + 6, np.zeros(3, np.float32), 0.8 * end_fade)
    blit(T, sub, x, y, gold, end_fade)
    edge = np.zeros_like(sub)
    edge[:, max(0, w - 6):w] = sub[:, max(0, w - 6):w]
    if reveal < 1:
        blit(T, edge, x, y, np.array([3.0, 2.2, 1.4], np.float32), 1.0, add=True)


def render_text(t):
    T = np.zeros((H, W, 4), np.float32)
    draw_posters(T, t)
    draw_title(T, t)
    draw_lines(T, t)
    if bar(170) <= t < bar(181):
        draw_solo(T, t)
    draw_signature(T, t)
    return T
