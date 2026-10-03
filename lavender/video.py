"""Frame renderer for 《薰衣草》.  python video.py still t1 t2 ...  |  python video.py render out.mp4"""
import math
import os
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import *

cv2.setNumThreads(1)
LW, LHh = W // 4, H // 4
CX, CY = 1380.0, 530.0
TWO_PI = 2 * np.pi


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def sst(a, b, t):
    return smooth((t - a) / (b - a))


def hexc(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


def window(t, a, b, fa=0.4, fb=0.4):
    return sst(a, a + fa, t) * (1 - sst(b - fb, b, t))


# ------------------------------------------------------------------ fonts & glyphs
F_POEM = ImageFont.truetype(f"{FONT_DIR}/NotoSerifSC.otf", 44)
F_TITLE = ImageFont.truetype(f"{FONT_DIR}/NotoSerifSC-Bold.otf", 112)
F_MONO = ImageFont.truetype(f"{FONT_DIR}/wenkaimono.ttf", 26)
F_TINY = ImageFont.truetype(f"{FONT_DIR}/wenkai.ttf", 17)
F_SIG = ImageFont.truetype(f"{FONT_DIR}/JetBrainsMono-Light.ttf", 46)
F_LAT = ImageFont.truetype(f"{FONT_DIR}/CormorantGaramond-LightItalic.ttf", 64)
F_LAT.set_variation_by_axes([300])
FONTS = dict(poem=F_POEM, title=F_TITLE, sig=F_SIG, lat=F_LAT)
_GC = {}


def glyph(ch, fk):
    k = (ch, fk)
    if k not in _GC:
        f = FONTS[fk]
        pad = 16
        asc, desc = f.getmetrics()
        adv = f.getlength(ch)
        im = Image.new("L", (int(math.ceil(adv)) + 2 * pad, asc + desc + 2 * pad), 0)
        ImageDraw.Draw(im).text((pad, pad), ch, font=f, fill=255)
        a = np.asarray(im, np.float32) / 255.0
        g = cv2.GaussianBlur(a, (0, 0), 5.0)
        g = g / (g.max() + 1e-6)
        _GC[k] = (a, g, adv, pad)
    return _GC[k]


def _region(a, x, y):
    h, w = a.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return None
    return (slice(y0, y1), slice(x0, x1)), a[y0 - y:y1 - y, x0 - x:x1 - x, None]


def blit_over(fr, a, x, y, col, op):
    r = _region(a, x, y)
    if r is None or op <= 0:
        return
    sl, sub = r
    sub = sub * op
    reg = fr[sl]
    reg *= 1 - sub
    reg += sub * col


def blit_add(fr, a, x, y, col, op):
    r = _region(a, x, y)
    if r is None or op <= 0:
        return
    sl, sub = r
    fr[sl] += sub * (col * op)


def blit_mul(fr, a, x, y, op):
    r = _region(a, x, y)
    if r is None or op <= 0:
        return
    sl, sub = r
    fr[sl] *= 1 - sub * op


def rot_glyph(a, ang):
    h, w = a.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), math.degrees(ang), 1.0)
    return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR)


def shift_sub(a, fx, fy):
    M = np.float32([[1, 0, fx], [0, 1, fy]])
    return cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR)


# ------------------------------------------------------------------ text layout
FS, LINE_H, LEFT, SPW = 44, 82, 170, 22


def line_xy(si, li):
    n = len(STANZAS[si])
    ytop = H / 2 - n * LINE_H / 2 + 8
    return LEFT + STANZAS[si][li][0] * SPW, ytop + li * LINE_H


def char_xs(si, li, chars):
    x0, _ = line_xy(si, li)
    xs = []
    x = x0
    for ch in chars:
        xs.append(x)
        x += glyph(ch, "poem")[2]
    return xs, x


FINAL_TEXT = [[txt for _, txt in st] for st in STANZAS]


def find_idx(si, li, sub, occ=0):
    s, k = FINAL_TEXT[si][li], -1
    for _ in range(occ + 1):
        k = s.index(sub, k + 1)
    return k


# homoglyph / subliminal swaps: (si, li, idx) -> list of (t0, t1, replacement)
SWAPS = {}


def _swap(si, li, sub, rep, t0, frames, occ=0):
    k = find_idx(si, li, sub, occ)
    SWAPS.setdefault((si, li, k), []).append((t0, t0 + frames / FPS + 1e-4, rep))


_swap(0, 3, "我", "找", 31.0, 3)
_swap(1, 0, "玫", "玖", 40.6, 3)
_swap(1, 1, "绝", "絶", 45.2, 4)
_swap(2, 3, "好", "妤", 74.0, 4)
_swap(2, 3, "人", "入", 74.6, 3)
_swap(3, 2, "心", "必", 100.2, 3)
_swap(4, 3, "理", "埋", 120.0, 4)
_swap(5, 1, "爱", "受", 130.9, 3)
_swap(7, 3, "安", "亡", find_time(7, 3, "安", 1) + 0.55, 2, occ=1)


def stanza_window(si):
    t0 = LINE_PLAN[si][0][0]
    if si in DISSOLVE:
        return t0, DISSOLVE[si] + 1.6
    return t0, {5: CUT, 6: RAW[0], 7: 206.0}[si]


def text_color(t):
    if t < 150:
        return np.array([0.95, 0.91, 0.85], np.float32)
    if t < 176:
        return np.array([0.86, 0.88, 0.82], np.float32)
    return np.array([0.98, 0.97, 0.95], np.float32)


GLOW_COL = np.array([0.55, 0.42, 0.85], np.float32)
FLASH_COL = np.array([0.75, 0.62, 1.0], np.float32)
CURSOR_COL = np.array([0.78, 0.68, 1.0], np.float32)


def float_params(si, li, k):
    r = np.random.default_rng(5000 + li * 97 + k)
    u = r.random(6)
    return u


def draw_poem(fr, t):
    tc = text_color(t)
    bright_bg = 176.5 < t < 204
    glow_amt = 0.0 if bright_bg else 0.22
    shadow_amt = 0.55 if bright_bg else 0.25
    cursor = None
    for si in range(len(STANZAS)):
        a, b = stanza_window(si)
        if not (a <= t < b):
            continue
        for li in range(len(STANZAS[si])):
            buf = line_state(si, li, t)
            if not buf:
                continue
            chars = [c for c, _ in buf]
            xs, xend = char_xs(si, li, chars)
            _, ytop = line_xy(si, li)
            last_op = max(te for te, _, _ in OPS[si][li] if te <= t)
            cursor = (xend, ytop, last_op, si)
            for k, ((ch, tt), x) in enumerate(zip(buf, xs)):
                age = t - tt
                op = smooth(age / 0.16)
                flash = max(0.0, 1 - age / 0.35)
                dx = dy = 0.0
                ang = 0.0
                if si in DISSOLVE:
                    xn = np.clip((x - LEFT) / 1200, 0, 1)
                    td = DISSOLVE[si] + 0.9 * xn
                    op *= 1 - smooth((t - td) / 0.28)
                    cursor = None if t > DISSOLVE[si] else cursor
                for (s0, s1, rep) in SWAPS.get((si, li, k), []):
                    if s0 <= t < s1:
                        ch = rep
                if si == 2 and li == 4 and k >= len(FINAL_TEXT[2][4]) - 3 and t < 80.5:
                    rr = np.random.default_rng(int(t * FPS) * 31 + k)
                    amp = 2.5 * (1 - sst(79.0, 80.5, t))
                    dx, dy = rr.normal(0, amp, 2)
                if si == 7:
                    u = float_params(si, li, k)
                    if li < 3 or (li == 3 and tt < GRAVITY):
                        ls = GRAVITY + 0.2 + u[0] * 2.8
                    elif li == 3:
                        ls = 196.0 + u[0] * 2.0
                    elif li == 4:
                        ls = 199.3 + u[0] * 2.0
                    else:
                        ls = 1e9
                    tau = t - ls
                    if tau > 0:
                        acc = 22 + 40 * u[1]
                        dy = -0.5 * acc * tau * tau
                        dx = 14 * math.sin(0.9 * tau + u[2] * 6) * math.sqrt(tau) + 9 * tau
                        ang = (u[3] - 0.5) * 0.45 * tau
                        op *= 1 - sst(4.0, 8.5, tau)
                    op *= 1 - sst(203.6, 205.9, t)
                    if li == 5 and t > 200.2:
                        tau = t - 200.2
                        dx = 0.5 * 55 * tau * tau
                        dy = -2.5 * tau
                        op *= 1 - sst(4.0, 5.6, tau)
                if op <= 0.003:
                    continue
                al, gl, adv, pad = glyph(ch, "poem")
                px, py = x - pad + dx, ytop - pad + dy
                if ang != 0.0 or (si == 7 and (dx or dy)):
                    fx, fy = px - math.floor(px), py - math.floor(py)
                    if ang != 0.0:
                        al, gl = rot_glyph(al, ang), rot_glyph(gl, ang)
                    al, gl = shift_sub(al, fx, fy), shift_sub(gl, fx, fy)
                ix, iy = int(math.floor(px)), int(math.floor(py))
                if shadow_amt:
                    blit_mul(fr, gl, ix + 2, iy + 3, shadow_amt * op)
                if glow_amt or flash:
                    blit_add(fr, gl, ix, iy, GLOW_COL, (glow_amt + 0.9 * flash) * op)
                blit_over(fr, al, ix, iy, tc * (1 - 0.3 * flash) + FLASH_COL * 0.3 * flash, op)
    # cursor
    if cursor is not None and t < 199:
        xend, ytop, last_op, si = cursor
        idle = t - last_op
        on = idle < 0.45 or ((t * 1.15) % 1.0) < 0.55
        if on:
            x0, y0 = int(xend + 4), int(ytop + 6)
            fr[y0:y0 + 48, x0:x0 + 3] = fr[y0:y0 + 48, x0:x0 + 3] * 0.2 + CURSOR_COL * 0.8


# ------------------------------------------------------------------ particle swarm
N = 14000
_r = np.random.default_rng(7)
U = _r.random((N, 12)).astype(np.float32)
G = _r.standard_normal((N, 6)).astype(np.float32)
SWIRL = (G[:, 4:6] * 140).astype(np.float32)


def mixcol(cols, u):
    cols = np.array([hexc(c) for c in cols], np.float32)
    k = np.minimum((u * len(cols)).astype(int), len(cols) - 1)
    return cols[k]


C_LAV = mixcol(["b79ce8", "9b7fd8", "c8b2f2", "a88be0", "e0c9ff", "8e6fcf"], U[:, 9])
C_CREAM = mixcol(["fff1dd", "f6e4cf", "ffe9f0"], U[:, 9])
C_ROSE = mixcol(["ff8fb0", "e8577e", "ffc2d4", "c93d64"], U[:, 9])
C_STAR = mixcol(["fff2d8", "dfe8ff", "ffe0f4", "cfd9ff", "fff9ee"], U[:, 9])
C_GREEN = mixcol(["6d8f5a", "58784a", "84a46a"], U[:, 10])
C_MATCHA = mixcol(["9dbb6a", "86a85a", "b2cc84"], U[:, 10])
C_GOLD = mixcol(["e6a95c", "d4924a", "f0c07a"], U[:, 10])
C_SILVER = mixcol(["dfe8ff", "c7d6ff", "eef2ff"], U[:, 10])
C_AZURE = mixcol(["bfe6ff", "e7f6ff", "d9d2ff", "a8d8ff"], U[:, 10])


def P(x, y):
    return np.stack([x, y], 1).astype(np.float32)


def sh_void(t):
    return P(960 + G[:, 0] * 2, 540 + G[:, 1] * 2), C_LAV, np.zeros(N, np.float32)


def sh_seed(t):
    r = 3.0 + 1.2 * math.sin(t * 2.2)
    b = np.full(N, 0.006 * (0.8 + 0.2 * math.sin(3.1 * t)), np.float32)
    return P(960 + G[:, 0] * r, 540 + G[:, 1] * r), C_CREAM, b


def sh_dust(t, dim=1.0):
    x = U[:, 0] * W + 30 * np.sin(0.13 * t + U[:, 1] * 6.28) + 6 * np.sin(0.4 * t + U[:, 2] * 9)
    y = U[:, 3] * H + 22 * np.sin(0.11 * t + U[:, 4] * 6.28) - 4 * np.sin(0.3 * t + U[:, 5] * 9)
    b = (0.04 + 0.25 * U[:, 6] ** 4) * (0.6 + 0.4 * np.sin(t * (1 + 2 * U[:, 7]) + U[:, 8] * 30)) * dim
    return P(x, y), C_LAV * 0.6 + C_CREAM * 0.4, b.astype(np.float32)


def sh_ash(t):
    x = U[:, 0] * W + 25 * np.sin(0.5 * t + U[:, 1] * 6.28)
    y = (U[:, 3] * (H + 100) + (18 + 20 * U[:, 4]) * (t - 152.5)) % (H + 100) - 50
    b = (0.03 + 0.12 * U[:, 6] ** 3).astype(np.float32)
    return P(x, y), C_CREAM * 0.5 + 0.2, b


def _make_spikes():
    rs = np.random.default_rng(5)
    out = []
    for li, (cnt, by, scl) in enumerate([(20, 700, 0.36), (14, 845, 0.63), (9, 1045, 1.0)]):
        for bx in np.linspace(-90, W + 90, cnt) + rs.normal(0, 45, cnt):
            k = int(7 + 10 * scl)
            for q in range(k):
                th = (q / (k - 1) - 0.5) * 1.25 + rs.normal(0, 0.07)
                L = 235 * scl * (0.72 + 0.32 * rs.random()) * (1 - 0.25 * abs(th))
                out.append((bx + rs.normal(0, 7 * scl), by + rs.normal(0, 9 * scl), th, L, scl, rs.random() * 6.28))
    return np.array(out, np.float32)


SPK = _make_spikes()
_w = SPK[:, 3] ** 1.2
SPK_CW = np.cumsum(_w) / _w.sum()
C_LAVDEEP = mixcol(["8a5cf0", "a678ff", "7a4fd8", "b48cff"], U[:, 9])


def sh_field(t):
    j = np.minimum(np.searchsorted(SPK_CW, U[:, 0]), len(SPK) - 1)
    bx, by, th, L, scl, ph = (SPK[j, k] for k in range(6))
    spike = U[:, 2] < 0.8
    v = np.where(spike, 0.58 + 0.42 * U[:, 1], 0.58 * U[:, 1])
    whorl = 0.3 + 0.7 * np.sin(v * 46 + ph) ** 2
    thick = np.where(spike, 0.045 * L * (1 - 0.65 * (v - 0.58) / 0.42) * whorl, 0.006 * L)
    sway = 9 * scl * np.sin(0.9 * t + bx * 0.004 + ph * 0.15) + 2.5 * scl * np.sin(2.3 * t + ph)
    x = bx + L * v * np.sin(th) + sway * v ** 2 + G[:, 0] * thick
    y = by - L * v * np.cos(th) + G[:, 1] * thick * 0.4
    pol = U[:, 3] < 0.05
    x = np.where(pol, U[:, 4] * W + 40 * np.sin(0.2 * t + U[:, 5] * 9), x)
    y = np.where(pol, 300 + U[:, 5] * 620 + 25 * np.sin(0.3 * t + U[:, 6] * 9) - 6 * t, y)
    far = (1 - scl)[:, None]
    col = np.where(spike[:, None], C_LAVDEEP * (1 - far) + (C_LAV * 0.6 + C_SILVER * 0.4) * far, C_GREEN)
    b = np.where(spike, 0.20 + 0.20 * scl, 0.10)
    b = np.where(pol, 0.35 * U[:, 7] ** 2, b)
    return P(x, y), col, b.astype(np.float32)


def sh_donut(t):
    u, v = U[:, 0] * TWO_PI, U[:, 1] * TWO_PI
    R, r = 205, 86
    cx, cy, cz = (R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)
    bb, aa = 0.35 * t, 1.0 + 0.12 * math.sin(0.3 * t)
    x1 = cx * math.cos(bb) - cy * math.sin(bb)
    y1 = cx * math.sin(bb) + cy * math.cos(bb)
    y2 = y1 * math.cos(aa) - cz * math.sin(aa)
    z2 = y1 * math.sin(aa) + cz * math.cos(aa)
    s = 900 / (900 + z2)
    frost = np.sin(v) > -0.15
    col = np.where(frost[:, None], C_LAV * 0.5 + C_ROSE * 0.5, C_GOLD * 0.9)
    spr = frost & (U[:, 2] < 0.04)
    col = np.where(spr[:, None], mixcol(["7ff3ff", "fff27a", "ff7ad9", "ffffff"], U[:, 3]), col)
    b = (0.20 + 0.25 * spr) * (0.45 + 0.55 * np.clip(0.5 - z2 / (2 * (R + r)), 0, 1))
    return P(CX + x1 * s, CY + y2 * s), col, b.astype(np.float32)


def sh_crepe(t):
    L = 24
    k = (U[:, 0] * L).astype(int)
    Wd = 560
    amp = 5 + 16 * sst(18.5, 23.0, t)
    x = CX - Wd / 2 + U[:, 1] * Wd + amp * np.sin(0.5 * t + k * 0.7) * (k % 2 * 2 - 1)
    yk = CY + 185 - k * 15.5
    y = yk + 2.2 * np.sin(x * 0.02 + k * 1.3 + 0.4 * t) + G[:, 0] * 1.6
    powder = U[:, 2] < 0.07
    y = np.where(powder, CY + 185 - (L - 1) * 15.5 - 5 - np.abs(G[:, 1]) * 9, y)
    col = np.where((k % 2 == 0)[:, None] | powder[:, None], C_MATCHA, C_CREAM)
    b = np.where(powder, 0.30, 0.26) * (0.8 + 0.2 * np.sin(x * 0.05 + t))
    return P(x, y), col, b.astype(np.float32)


EYE_LASH = np.linspace(-0.82, 0.82, 11)


def eye_open(t):
    o = sst(29.6, 30.0, t) * (1 - sst(30.9, 31.25, t))
    o += 0.28 * window(t, 32.1, 32.35, 0.05, 0.12)
    return float(o)


def sh_eye(t):
    o = eye_open(t)
    xr = U[:, 0] * 2 - 1
    Wd = 250
    x = CX + xr * Wd
    low = CY - 10 + 58 * (1 - xr ** 2)
    up = low - o * 150 * (1 - xr ** 2)
    part = U[:, 1]
    y = np.where(part < 0.5, up, low) + G[:, 0] * 1.2
    b = np.where(part < 0.5, 0.42, 0.14 + 0.25 * o)
    col = C_LAV.copy()
    lash = (part >= 0.6) & (part < 0.76)
    li = (U[:, 2] * len(EYE_LASH)).astype(int)
    lx = EYE_LASH[li]
    lbx, lby = CX + lx * Wd, CY - 10 + 58 * (1 - lx ** 2) - o * 150 * (1 - lx ** 2)
    dxl, dyl = lx * 0.6, 1 - 2.2 * o
    nrm = np.sqrt(dxl ** 2 + dyl ** 2) + 1e-6
    sl = U[:, 3] * 40
    x = np.where(lash, lbx + dxl / nrm * sl, x)
    y = np.where(lash, lby + dyl / nrm * sl, y)
    b = np.where(lash, 0.3, b)
    iris = part >= 0.76
    ang = U[:, 4] * TWO_PI
    rr = np.where(U[:, 5] < 0.55, 62, 24 + 38 * np.sqrt(U[:, 6]))
    icx, icy = CX - 34 * o, CY - 10 + 58 - 75 * o
    ix, iy = icx + rr * o * np.cos(ang), icy + rr * o * np.sin(ang)
    hl = U[:, 7] < 0.08
    ix = np.where(hl, icx - 14 * o + G[:, 1] * 3, ix)
    iy = np.where(hl, icy - 16 * o + G[:, 2] * 3, iy)
    x = np.where(iris, x * (1 - o) + ix * o, x)
    y = np.where(iris, y * (1 - o) + iy * o, y)
    b = np.where(iris, 0.08 + 0.5 * o, b)
    col = np.where(iris[:, None], np.where(hl[:, None], C_CREAM, mixcol(["6c3fd8", "9a6cff", "4b2aa8"], U[:, 8])), col)
    return P(x, y), col, b.astype(np.float32)


def sh_rose(t):
    th = U[:, 0] * 8 * np.pi
    R = 250 * (1 + 0.02 * math.sin(1.3 * t))
    r = R * np.cos(1.25 * th)
    ang = th + 0.04 * t
    deny = 16 * sst(40.3, 41.2, t) * (1 - sst(44.0, 45.6, t))
    fr = int(t * FPS)
    jx = np.sin(U[:, 1] * 9137.0 + fr * 12.9898) * deny
    jy = np.sin(U[:, 2] * 7919.0 + fr * 78.233) * deny
    x = CX + r * np.cos(ang) + G[:, 0] * 1.3 + jx
    y = CY + r * np.sin(ang) + G[:, 1] * 1.3 + jy
    q = np.abs(r) / R
    col = C_ROSE * (0.6 + 0.4 * q[:, None]) + C_LAV * (0.4 - 0.4 * q[:, None])
    b = 0.22 + 0.25 * (1 - q)
    return P(x, y), col, b.astype(np.float32)


def sh_waves(t):
    nl = 9
    l = (U[:, 0] * nl).astype(int)
    x = U[:, 1] * (W + 100) - 50
    env = 0.55 + 0.45 * np.sin(x * 0.002 + l + 0.3 * t)
    y = 250 + l * 72 + (26 + 10 * (l % 3)) * np.sin(x * (0.0055 + 0.0012 * l) - (1.3 + 0.17 * l) * t + l * 0.9) * env
    y = y + G[:, 0] * 1.1
    col = np.where((l % 3 == 0)[:, None], C_ROSE, np.where((l % 3 == 1)[:, None], C_LAV, C_CREAM))
    b = 0.42 + 0.2 * np.sin(x * 0.01 - 2 * t + l)
    return P(x, y), col, b.astype(np.float32)


def _bear_points():
    S = 1
    im = Image.new("L", (900, 1000), 0)
    out = Image.new("L", (900, 1000), 0)
    d, do = ImageDraw.Draw(im), ImageDraw.Draw(out)
    ox, oy = 450, 470

    def ell(cx, cy, rx, ry, fill=True, w=3):
        box = [ox + cx - rx, oy + cy - ry, ox + cx + rx, oy + cy + ry]
        if fill:
            d.ellipse(box, fill=255)
        do.ellipse(box, outline=255, width=w)

    for sx in (-1, 1):
        ell(sx * 92, -208, 46, 46)
        ell(sx * 150, 60, 46, 74)
        ell(sx * 82, 228, 58, 52)
    ell(0, 105, 138, 165)
    ell(0, -118, 120, 112)
    ell(0, -80, 50, 36)
    for sx in (-1, 1):
        do.ellipse([ox + sx * 92 - 24, oy - 232, ox + sx * 92 + 24, oy - 184], outline=255, width=2)
        do.ellipse([ox + sx * 82 - 30, oy + 210, ox + sx * 82 + 30, oy + 252], outline=255, width=2)
    fill = np.argwhere(np.asarray(im) > 128)[:, ::-1].astype(np.float32) - [ox, oy]
    edge = np.argwhere(np.asarray(out) > 128)[:, ::-1].astype(np.float32) - [ox, oy]
    return fill, edge


BEAR_FILL, BEAR_EDGE = _bear_points()
BEAR_EYES = np.array([[-42, -138], [42, -138]], np.float32)


def sh_bear(t, cx=CX, cy=CY + 20):
    part = U[:, 1]
    fi = (U[:, 0] * len(BEAR_FILL)).astype(int)
    ei = (U[:, 0] * len(BEAR_EDGE)).astype(int)
    pf = BEAR_FILL[fi] + G[:, :2] * 0.6
    pe = BEAR_EDGE[ei] + G[:, :2] * 0.5
    eye = part > 0.94
    nose = (part > 0.91) & ~eye
    p = np.where((part < 0.42)[:, None], pf, pe)
    blink = 1 - window(t, 55.7, 55.95, 0.07, 0.1) - window(t, 179.0, 179.25, 0.07, 0.1)
    ek = (U[:, 2] < 0.5).astype(int)
    ea, er = U[:, 3] * TWO_PI, 9 * np.sqrt(U[:, 4])
    pey = np.stack([BEAR_EYES[ek, 0] + er * np.cos(ea), BEAR_EYES[ek, 1] + er * np.sin(ea) * blink], 1)
    pno = np.stack([G[:, 0] * 7, -100 + G[:, 1] * 4], 1)
    p = np.where(eye[:, None], pey, np.where(nose[:, None], pno, p))
    # slow head tilt — the bear looks at you
    tilt = 0.24 * sst(53.6, 55.4, t) * (1 - sst(56.5, 57.5, t)) + 0.18 * window(t, 176.4, 179.8, 1.0, 0.8)
    head = p[:, 1] < -30
    ca, sa = math.cos(tilt), math.sin(tilt)
    hx, hy = p[:, 0], p[:, 1] + 30
    p = np.where(head[:, None], np.stack([hx * ca - hy * sa, hx * sa + hy * ca - 30], 1), p)
    sc = 1 + 0.012 * math.sin(1.6 * t)
    pos = P(cx + p[:, 0] * sc, cy + p[:, 1] * sc)
    col = np.where((part < 0.42)[:, None], C_GOLD * 0.7 + C_CREAM * 0.3, C_CREAM * 0.7 + C_LAV * 0.3)
    col = np.where(eye[:, None], np.array([0.85, 0.75, 1.0], np.float32), col)
    b = np.where(part < 0.42, 0.07, 0.30)
    b = np.where(eye, 0.6 * blink + 0.05, b)
    return pos, col, b.astype(np.float32)


def sh_sky(t, t0=58.0, rot=0.006, dim=1.0):
    x0, y0 = U[:, 0] * (W + 300) - 150, U[:, 1] * (H + 300) - 150
    ang = rot * (t - t0)
    px, py = 960, 2000
    dx, dy = x0 - px, y0 - py
    x = px + dx * math.cos(ang) - dy * math.sin(ang)
    y = py + dx * math.sin(ang) + dy * math.cos(ang)
    mag = U[:, 2] ** 6
    tw = 0.6 + 0.4 * np.sin(t * (1.3 + 4 * U[:, 3]) + U[:, 4] * 50)
    b = (0.07 + 2.6 * mag) * tw * dim
    return P(x, y), C_STAR, b.astype(np.float32)


_rb = np.random.default_rng(9)
NBL = 95
BL_X = _rb.random(NBL) * (W + 100) - 50
BL_H = 110 + 250 * _rb.random(NBL)
BL_C = (_rb.random(NBL) - 0.5) * 0.7
BL_P = _rb.random(NBL) * 6.28


def sh_dew(t):
    sp, sc, sb = sh_sky(t, dim=0.55)
    part = U[:, 6]
    j = (U[:, 7] * NBL).astype(int)
    v = U[:, 8]
    sway = 6 * np.sin(0.8 * t + BL_P[j])
    bx = BL_X[j] + BL_C[j] * (v ** 2) * BL_H[j] + sway * v ** 2
    by = 1095 - v * BL_H[j]
    blade = (part >= 0.35) & (part < 0.72)
    drop = part >= 0.72
    vd = 0.86 - 0.32 * sst(64.0, 67.5, t) * U[:, 10]
    dcx = BL_X[j] + BL_C[j] * (vd ** 2) * BL_H[j] + sway * vd ** 2
    dcy = 1095 - vd * BL_H[j] - 9
    rr = 4 + 7 * (j % 5) / 4
    a = U[:, 11] * TWO_PI
    dx_, dy_ = dcx + rr * np.cos(a), dcy + rr * np.sin(a)
    x = np.where(blade, bx + G[:, 0] * 0.5, np.where(drop, dx_, sp[:, 0]))
    y = np.where(blade, by, np.where(drop, dy_, sp[:, 1]))
    glit = 0.5 + 0.5 * np.sin(t * 3 + j * 1.7) ** 8
    col = np.where(blade[:, None], C_GREEN * 0.9, np.where(drop[:, None], C_SILVER, sc))
    b = np.where(blade, 0.30, np.where(drop, 0.30 + 0.9 * glit * (np.cos(a + 2.2) > 0.6), sb))
    return P(x, y), col, b.astype(np.float32)


_rc = np.random.default_rng(13)
CHIPS = np.stack([_rc.uniform(-130, 130, 12), _rc.uniform(-130, 130, 12)], 1)
CHIPS = CHIPS[np.hypot(CHIPS[:, 0], CHIPS[:, 1]) < 140]


def sh_cookie(t):
    a = U[:, 0] * TWO_PI
    rr = np.sqrt(U[:, 1])
    edge = 178 * (1 + 0.035 * np.sin(7 * a) + 0.02 * np.sin(13 * a + 1))
    x, y = rr * edge * np.cos(a), rr * edge * np.sin(a)
    d = np.hypot(x[:, None] - CHIPS[None, :, 0], y[:, None] - CHIPS[None, :, 1])
    ci = d.argmin(1)
    inchip = d.min(1) < 19
    ca = U[:, 2] * TWO_PI
    x = np.where(inchip, CHIPS[ci, 0] + 19 * np.cos(ca), x)
    y = np.where(inchip, CHIPS[ci, 1] + 19 * np.sin(ca), y)
    bx, by, br = 135, -120, 92
    bite = np.hypot(x - bx, y - by) < br
    fall = np.clip(t - 75.4, 0, None)
    cang = U[:, 3] * 2.2 + 2.6
    crx = bx + (br + 10 + 40 * U[:, 4]) * np.cos(cang)
    cry = by + (br + 10 + 40 * U[:, 4]) * np.sin(cang) + 0.5 * 160 * fall ** 2 * U[:, 5]
    x, y = np.where(bite, crx, x), np.where(bite, cry, y)
    rot = 0.08 * t
    xr, yr = x * math.cos(rot) - y * math.sin(rot), x * math.sin(rot) + y * math.cos(rot)
    col = np.where(inchip[:, None], C_GOLD * 0.45, C_GOLD)
    b = np.where(inchip, 0.30, 0.13) * np.where(bite, 0.8, 1.0)
    return P(CX + xr, CY + yr), col, b.astype(np.float32)


def sh_warp(t):
    z = ((U[:, 0] - 0.1 * (t - 79.0)) % 1.0) * 0.96 + 0.04
    X, Y = (U[:, 1] - 0.5) * 2600, (U[:, 2] - 0.5) * 1600
    x, y = 960 + X * 0.3 / z, 540 + Y * 0.3 / z
    b = (0.12 + 3.0 * U[:, 3] ** 5) * sst(1.0, 0.75, z) * (0.4 + 0.6 * (1 - z))
    return P(x, y), C_STAR, b.astype(np.float32)


NMET = 16


def sh_wishes(t):
    pos, col, b = sh_sky(t, t0=80.0, rot=0.004)
    frac = 0.18 + 0.82 * sst(88.2, 93.2, t)
    b = b * smooth((frac - U[:, 6]) / 0.04) * (1 + 0.5 * sst(88, 93, t))
    met = U[:, 7] < 0.10
    m = (U[:, 8] * NMET).astype(int)
    rm = np.random.default_rng(21)
    tm = 88.6 + np.cumsum(0.55 * 0.86 ** np.arange(NMET))
    p0 = np.stack([rm.uniform(500, 2000, NMET), rm.uniform(-80, 420, NMET)], 1)
    dirs = np.stack([-np.ones(NMET), rm.uniform(0.35, 0.7, NMET)], 1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    tau = t - tm[m]
    act = (tau > 0) & (tau < 0.9)
    s = U[:, 9]
    head = p0[m] + dirs[m] * (np.clip(tau, 0, 0.9) * 1100)[:, None]
    mp = head - dirs[m] * (s * 170)[:, None]
    mb = 1.2 * (1 - s) ** 2 * np.sin(np.clip(tau / 0.9, 0, 1) * np.pi)
    pos = np.where(met[:, None], mp, pos)
    b = np.where(met, np.where(act, mb, 0.0), b)
    col = np.where(met[:, None], C_CREAM, col)
    return pos.astype(np.float32), col, b.astype(np.float32)


def sh_point(t):
    return P(CX + G[:, 0] * 3, CY - 30 + G[:, 1] * 3), C_SILVER, np.full(N, 0.006, np.float32)


def sh_moon(t):
    R = 172
    cx, cy = CX, CY - 30 - 0.6 * (t - 103)
    a = U[:, 0] * TWO_PI
    r = R * np.sqrt(U[:, 1])
    x, y = r * np.cos(a), r * np.sin(a)
    m = (np.sin(x * 0.021 + 1.0) * np.cos(y * 0.017 - 0.5) + 0.6 * np.sin(x * 0.043 - y * 0.031 + 2.0)
         + 0.3 * np.sin(0.09 * x + 0.07 * y))
    rim = U[:, 2] < 0.16
    halo = (U[:, 2] >= 0.16) & (U[:, 2] < 0.24)
    x = np.where(rim, R * np.cos(a), np.where(halo, R * (1.12 + 0.7 * U[:, 3] ** 2) * np.cos(a), x))
    y = np.where(rim, R * np.sin(a), np.where(halo, R * (1.12 + 0.7 * U[:, 3] ** 2) * np.sin(a), y))
    b = 0.30 * (0.35 + 0.65 * np.clip(0.6 - 0.5 * m, 0, 1))
    b = np.where(rim, 0.55, np.where(halo, 0.04, b))
    cheese = window(t, 104.6, 107.4, 0.8, 1.0)
    col = C_SILVER * (1 - cheese) + hexc("ffe39a") * cheese
    return P(cx + x, cy + y), col, b.astype(np.float32)


_rn = np.random.default_rng(17)
NNOTE = 34
NOTE_X = np.sort(_rn.uniform(220, 1780, NNOTE))
NOTE_L = _rn.integers(0, 9, NNOTE)
NOTE_TD = 108.4 + np.arange(NNOTE) * 0.17 + _rn.uniform(0, 0.6, NNOTE)
NOTE_PH = _rn.uniform(0, 6.28, NNOTE)


def staff_y(x, t):
    return 150 + 16 * np.sin(x * 0.003 + 0.5 * t) + 0.04 * (x - 960) + 18 * max(0.0, t - 107.8)


def sh_staff(t):
    part = U[:, 6]
    line = part < 0.30
    note = (part >= 0.30) & (part < 0.52)
    rain = part >= 0.52
    li = (U[:, 0] * 5).astype(int)
    x = 140 + U[:, 1] * 1680
    y = staff_y(x, t) + li * 24 + G[:, 0] * 0.6
    n = (U[:, 2] * NNOTE).astype(int)
    nx = NOTE_X[n]
    ny = staff_y(nx, t) + NOTE_L[n] * 12 - 0
    tau = np.clip(t - NOTE_TD[n], 0, None)
    fx = nx - 25 * tau + 40 * np.sin(1.4 * tau + NOTE_PH[n]) * np.minimum(tau, 1)
    fy = ny + 0.5 * 70 * tau ** 2
    head = U[:, 3] < 0.62
    ha = U[:, 4] * TWO_PI
    hr = np.sqrt(U[:, 5])
    hxl, hyl = 10 * hr * np.cos(ha), 7 * hr * np.sin(ha)
    c, s = math.cos(-0.35), math.sin(-0.35)
    hx, hy = hxl * c - hyl * s, hxl * s + hyl * c
    sx, sy = 9 + 0 * U[:, 4], -U[:, 4] * 52
    px = fx + np.where(head, hx, sx)
    py = fy + np.where(head, hy, sy)
    dd = (U[:, 0] * 420).astype(int)
    rs = np.random.default_rng(3)
    rx0 = rs.uniform(-200, W + 100, 420)
    ry0 = rs.uniform(0, H + 200, 420)
    rv = rs.uniform(1300, 1800, 420)
    ry = (ry0[dd] + rv[dd] * t) % (H + 200) - 100
    rx = rx0[dd] + 0.18 * ry
    sl = U[:, 1] * 46
    rx, ry = rx - 0.18 * sl, ry - sl
    rainamt = sst(107.4, 109.0, t) * (1 - sst(119.0, 121.5, t))
    x = np.where(line, x, np.where(note, px, rx))
    y = np.where(line, y, np.where(note, py, ry))
    col = np.where(rain[:, None], C_SILVER, C_CREAM)
    b = np.where(line, 0.16, np.where(note, 0.30, 0.10 * rainamt))
    return P(x, y), col, b.astype(np.float32)


def frame_material(t):
    ts = [find_time(4, 3, "象"), find_time(4, 3, "璞"), find_time(4, 3, "大")]
    return [t >= tt for tt in ts]


MAT_COL = [hexc("b07a4a"), hexc("fff3d6"), hexc("7fe0b0"), hexc("e9ecf5")]


def sh_frame(t):
    hw, hh, th = 270, 210, 38
    per = 2 * (2 * hw + 2 * hh)
    s = U[:, 0] * per
    w = (U[:, 1] - 0.5) * th
    x = np.zeros(N, np.float32)
    y = np.zeros(N, np.float32)
    L1, L2, L3 = 2 * hw, 2 * hw + 2 * hh, 4 * hw + 2 * hh
    seg0, seg1, seg2 = s < L1, (s >= L1) & (s < L2), (s >= L2) & (s < L3)
    seg3 = s >= L3
    x = np.where(seg0, -hw + s, np.where(seg1, hw + w, np.where(seg2, hw - (s - L2), -hw + w)))
    y = np.where(seg0, -hh + w, np.where(seg1, -hh + (s - L1), np.where(seg2, hh + w, hh - (s - L3))))
    edge = np.abs(np.abs(w) - th / 2) < 1.6
    along = np.where(seg0 | seg2, x, y)
    across = w
    ts = frame_material(t)
    k = sum(ts)
    tcol = MAT_COL[k]
    tex = 1.0
    if k == 0:
        tex = 0.7 + 0.3 * np.sin(along * 0.12 + 5 * np.sin(across * 0.15 + along * 0.01))
    elif k == 2:
        tex = 0.75 + 0.25 * np.sin(along * 0.03 + 3 * np.sin(across * 0.2)) * np.cos(along * 0.011)
    elif k == 3:
        vein = np.abs(np.sin(0.025 * x + 0.018 * y + 3 * np.sin(0.012 * y + 0.01 * x)))
        tex = np.where(vein < 0.09, 0.35, 1.0)
    inside = U[:, 2] < 0.12
    x = np.where(inside, (U[:, 3] - 0.5) * 2 * (hw - 30) + 6 * math.sin(t), x)
    y = np.where(inside, (U[:, 4] - 0.5) * 2 * (hh - 30) + 6 * np.sin(t * 0.7 + U[:, 5] * 6), y)
    flash = max([0.0] + [max(0.0, 1 - (t - tt) / 0.4) for tt in [find_time(4, 3, c) for c in "象璞大"] if t >= tt])
    b = np.where(edge, 0.32, 0.12) * tex * (1 + 1.5 * flash)
    b = np.where(inside, 0.03, b)
    col = np.broadcast_to(tcol, (N, 3)).astype(np.float32)
    return P(CX + x, CY + y + 4 * math.sin(0.5 * t)), col, b.astype(np.float32)


def sh_binary(t):
    ph = TWO_PI * (t - 123) / 6.0
    rad = 125 + 75 * sst(128.4, 131.6, t)
    c1 = np.array([CX + rad * math.cos(ph), CY + rad * 0.42 * math.sin(ph)], np.float32)
    c2 = np.array([CX - rad * math.cos(ph), CY - rad * 0.42 * math.sin(ph)], np.float32)
    part = U[:, 0]
    k1, k2 = part < 0.42, (part >= 0.42) & (part < 0.84)
    rr = 24 * np.abs(G[:, 2]) ** 1.3
    ang = U[:, 1] * TWO_PI
    off = np.stack([rr * np.cos(ang), rr * np.sin(ang)], 1)
    oa = U[:, 2] * TWO_PI
    ring = np.stack([CX + rad * np.cos(oa), CY + rad * 0.42 * np.sin(oa)], 1)
    pos = np.where(k1[:, None], c1 + off, np.where(k2[:, None], c2 + off, ring))
    col = np.where(k1[:, None], C_LAV, np.where(k2[:, None], C_ROSE * 0.5 + C_CREAM * 0.5, C_LAV * 0.6))
    b = np.where(k1 | k2, 0.05 + 0.10 * np.exp(-rr / 10), 0.035)
    return pos.astype(np.float32), col, b.astype(np.float32)


def dirt(t):
    return sst(137.8, 139.8, t) * (1 - sst(141.8, 144.0, t))


def sh_globe(t):
    lat = np.arcsin(2 * U[:, 0] - 1)
    lon = U[:, 1] * TWO_PI
    land = (np.sin(3 * lon) * np.cos(2 * lat) + 0.6 * np.sin(5 * lon + 1) * np.sin(3 * lat + 0.5)
            + 0.4 * np.cos(7 * lon - 2 * lat)) > 0.3
    lo = lon + 0.25 * t
    x, y, z = np.cos(lat) * np.sin(lo), -np.sin(lat), np.cos(lat) * np.cos(lo)
    tl = 0.38
    y2, z2 = y * math.cos(tl) - z * math.sin(tl), y * math.sin(tl) + z * math.cos(tl)
    R = 235
    dd = dirt(t)
    fr = int(t * FPS)
    jit = dd * 3 * np.sin(U[:, 2] * 5000 + fr * 3.7)
    pos = P(CX + R * x + jit, CY + R * y2)
    col = np.where(land[:, None], C_CREAM * 0.6 + C_LAV * 0.4, mixcol(["4f5fd0", "6a5acd", "3c4aa0"], U[:, 3]))
    col = col * (1 - dd) + hexc("6b4a2a") * dd
    b = np.where(land, 0.62, 0.20) * np.where(z2 > 0, 1.0, 0.22) * (1 - 0.35 * dd)
    return pos, col, b.astype(np.float32)


RIP_C = np.array([[CX - 160, CY + 70], [CX + 130, CY - 40], [CX + 10, CY + 150]], np.float32)


def sh_ripples(t):
    c = (U[:, 0] * 3).astype(int)
    k = (U[:, 1] * 7).astype(int)
    r = ((t - 141.0) * 85 + k * 46 + c * 20) % 330
    a = U[:, 2] * TWO_PI
    x = RIP_C[c, 0] + r * np.cos(a)
    y = RIP_C[c, 1] + r * 0.36 * np.sin(a)
    b = 0.26 * (1 - r / 330) ** 1.5 * sst(0, 30, r)
    col = C_SILVER * 0.6 + C_LAV * 0.4
    return P(x, y), col, b.astype(np.float32)


def sh_sprig(t):
    part = U[:, 0]
    bx, by, tx, ty = CX + 40, CY + 330, CX - 10, CY - 290
    sw = 0.05 * math.sin(0.9 * t)
    stem = part < 0.22
    leaf = (part >= 0.22) & (part < 0.32)
    v = U[:, 1]
    sxv = bx + (tx - bx) * v + 22 * np.sin(np.pi * v * 1.2)
    syv = by + (ty - by) * v
    nw = 15
    w = (U[:, 2] * nw).astype(int)
    vw = 0.56 + 0.44 * w / (nw - 1)
    wx = bx + (tx - bx) * vw + 22 * np.sin(np.pi * vw * 1.2)
    wy = by + (ty - by) * vw
    size = 21 * (1 - 0.5 * (vw - 0.56) / 0.44)
    fa = U[:, 3] * TWO_PI
    fr = size * np.sqrt(U[:, 4])
    fx = wx + fr * np.cos(fa) * 1.15
    fy = wy + fr * np.sin(fa) * 0.75
    side = np.where(U[:, 5] < 0.5, -1, 1)
    lv = U[:, 1]
    lx = bx + side * lv * 120 + 10 * np.sin(lv * 3)
    ly = by - lv * 170 - side * 0 + 30 * lv ** 2
    x = np.where(stem, sxv, np.where(leaf, lx, fx)) + G[:, 0] * 0.6
    y = np.where(stem, syv, np.where(leaf, ly, fy))
    dx, dy = x - bx, y - by
    ca, sa = math.cos(sw), math.sin(sw)
    x, y = bx + dx * ca - dy * sa, by + dx * sa + dy * ca
    col = np.where((stem | leaf)[:, None], C_GREEN, mixcol(["9a6cff", "b38cff", "7d55e0", "d6bfff"], U[:, 6]))
    b = np.where(stem | leaf, 0.16, 0.075)
    return P(x, y), col, b.astype(np.float32)


MOON7 = (CX, 300.0, 150.0)


def sh_moon_absent(t):
    pos, col, b = sh_ash(t)
    ring = U[:, 11] < 0.30
    a = U[:, 0] * TWO_PI
    rx = MOON7[0] + MOON7[2] * np.cos(a) + G[:, 0] * 0.8
    ry = MOON7[1] + MOON7[2] * np.sin(a) + G[:, 1] * 0.8
    pos = np.where(ring[:, None], P(rx, ry), pos)
    b = np.where(ring, 0.07 * (0.6 + 0.4 * np.sin(a * 3 + t)), b)
    return pos, col, b.astype(np.float32)


def sh_river(t):
    bnd = (U[:, 0] * 7).astype(int)
    sp = 140 + 85 * bnd
    x = (U[:, 1] * (W + 400) + sp * (t - 164.4)) % (W + 400) - 200
    y = 540 + bnd * 72 + 22 * np.sin(0.004 * x + bnd + 0.7 * t) + G[:, 0] * 7
    col = mixcol(["8fa6d6", "6d85b8", "b6c4e6"], U[:, 2])
    b = 0.10 + 0.10 * U[:, 3]
    return P(x, y), col, b.astype(np.float32)


def sh_accent(t):
    p0 = np.array([360.0, 120.0])
    d = np.array([1.0, 0.55])
    d /= np.linalg.norm(d)
    L = 40 + 3000 * sst(167.1, 170.35, t) ** 1.7
    s = U[:, 0]
    th = 11 * (1 - 0.75 * s)
    nrm = np.array([-d[1], d[0]])
    x = p0[0] + d[0] * L * s + nrm[0] * G[:, 0] * th / 2
    y = p0[1] + d[1] * L * s + nrm[1] * G[:, 0] * th / 2
    col = np.broadcast_to(hexc("ffd9cf"), (N, 3)).astype(np.float32)
    b = np.full(N, 0.30, np.float32) * (1 + 0.6 * sst(169, 170.3, t))
    return P(x, y), col, b


def jelly_points(t, cx, cy, scale):
    pulse = math.sin(TWO_PI * t / 1.8)
    a = 150 * (1 + 0.07 * pulse) * scale
    bb = 118 * (1 - 0.05 * pulse) * scale
    part = U[:, 6]
    psi = np.arccos(1 - U[:, 7])
    phi = U[:, 8] * TWO_PI
    bxp = a * np.sin(psi) * np.cos(phi)
    byp = -bb * np.cos(psi)
    bell = part < 0.40
    rim = (part >= 0.40) & (part < 0.48)
    arm = (part >= 0.48) & (part < 0.60)
    ten = (part >= 0.60) & (part < 0.92)
    rider = part >= 0.985
    rmx = a * np.cos(phi)
    rmy = 3 * np.sin(phi * 8 + 2 * t) * scale
    s = U[:, 9]
    k = (U[:, 10] * 4).astype(int)
    amx = (k - 1.5) * 12 * scale + 22 * np.sin(3 * s * np.pi - 2 * t + k) * s * scale
    amy = s * 250 * scale
    kt = (U[:, 10] * 16).astype(int)
    tphi = kt / 16 * TWO_PI
    tx0 = a * np.cos(tphi) * 0.92
    tlen = (330 + 80 * (kt % 3)) * scale
    tnx = tx0 + 15 * np.sin(TWO_PI * (1.2 * s - 0.5 * t) + kt) * s ** 0.8 * scale
    tny = s * tlen
    rdx, rdy = G[:, 0] * 3.5 * scale, (300 + G[:, 1] * 4) * scale
    x = np.where(bell, bxp, np.where(rim, rmx, np.where(arm, amx, np.where(ten, tnx, np.where(rider, rdx, U[:, 3] * 1800 - 900)))))
    y = np.where(bell, byp, np.where(rim, rmy, np.where(arm, amy, np.where(ten, tny, np.where(rider, rdy, U[:, 4] * 1000 - 500)))))
    b = np.where(bell, 0.09, np.where(rim, 0.32, np.where(arm, 0.18, np.where(ten, 0.14 * (1 - s), np.where(rider, 0.45, 0.03)))))
    col = np.where((bell | rim)[:, None], C_LAV * 0.5 + C_AZURE * 0.5, C_AZURE)
    col = np.where(rider[:, None], np.array([1.0, 0.95, 0.85], np.float32), col)
    return P(cx + x, cy + y), col, b.astype(np.float32)


def sh_jelly(t):
    return jelly_points(t, CX - 40, 470 - 130 * sst(178, 193, t), 1.0)


def sh_blanket(t):
    jp, jc, jb = jelly_points(t, CX + 160 + 25 * (t - 186), 230 - 9 * (t - 186), 0.5)
    gu = (U[:, 0] * 110).astype(int) / 109.0
    gv = (U[:, 1] * 60).astype(int) / 59.0
    X = (gu - 0.5) * 2900
    Z = 420 + gv * 2000
    Y = 250 + 55 * np.sin(0.006 * X + 1.1 * t) + 45 * np.sin(0.004 * Z - 0.8 * t + 1)
    f = 820
    x = 960 + X * f / Z
    y = 470 + Y * f / Z
    lift_t = GRAVITY + 0.3 + U[:, 2] * 4.5
    tau = np.clip(t - lift_t, 0, None)
    y = y - 0.5 * (10 + 26 * U[:, 3]) * tau ** 2
    x = x + 12 * np.sin(tau * 0.7 + U[:, 4] * 6) * np.minimum(tau, 2)
    b = 0.22 * (1 - 0.75 * gv) * (1 + 0.3 * np.sin(0.006 * X + 1.1 * t))
    isj = U[:, 5] < 0.22
    x = np.where(isj, jp[:, 0], x)
    y = np.where(isj, jp[:, 1], y)
    b = np.where(isj, jb * 1.2, b)
    col = np.where(isj[:, None], jc, C_AZURE * 0.8 + C_CREAM * 0.2)
    ftau = np.clip(t - (197.6 + U[:, 6] * 2.5), 0, None)
    acc = 30 + 110 * U[:, 7]
    x = x + 0.5 * acc * ftau ** 2 * 1.0
    y = y - 0.5 * acc * ftau ** 2 * 0.62
    b = b * np.exp(-ftau / 3.5) * (1 - sst(203.5, 206.0, t))
    return P(x, y), col, b.astype(np.float32)


SHAPES = dict(void=sh_void, seed=sh_seed, dust=sh_dust, field=sh_field, donut=sh_donut, crepe=sh_crepe,
              eye=sh_eye, rose=sh_rose, waves=sh_waves, bear=sh_bear, sky=sh_sky, dew=sh_dew, cookie=sh_cookie,
              warp=sh_warp, wishes=sh_wishes, point=sh_point, moon=sh_moon, staff=sh_staff, frame=sh_frame,
              binary=sh_binary, globe=sh_globe, ripples=sh_ripples, sprig=sh_sprig, ash=sh_ash,
              moon_absent=sh_moon_absent, river=sh_river, accent=sh_accent, jelly=sh_jelly, blanket=sh_blanket)

KEYS = [
    (0.0, "void", 0.1), (3.0, "seed", 1.5), (8.4, "field", 3.6), (13.1, "donut", 2.8), (17.0, "crepe", 2.8),
    (22.4, "eye", 2.8), (33.0, "dust", 2.2), (36.4, "rose", 3.0), (46.4, "waves", 2.8), (51.6, "bear", 2.8),
    (58.4, "sky", 2.2), (62.4, "dew", 2.6), (75.0, "cookie", 2.0), (78.9, "warp", 2.2), (85.3, "dust", 1.8),
    (87.6, "wishes", 2.2), (100.9, "point", 1.5), (102.5, "moon", 2.4), (107.7, "staff", 2.6),
    (112.5, "frame", 2.4), (120.6, "dust", 1.6), (122.8, "binary", 2.4), (134.2, "globe", 2.4),
    (141.0, "ripples", 2.2), (144.8, "sprig", 2.6), (CUT, "void", 0.001), (152.5, "ash", 0.6),
    (160.6, "moon_absent", 2.2), (164.3, "river", 1.8), (166.9, "accent", 1.4), (RAW[0], "void", 0.001),
    (176.0, "bear", 1.6), (177.9, "jelly", 2.6), (185.4, "blanket", 3.0),
]


def _morph(k, t):
    t0, name, dur = KEYS[k]
    B = SHAPES[name](t)
    if k == 0:
        return B
    st = t0 + dur * 0.42 * U[:, 11]
    s = np.clip((t - st) / (dur * 0.58), 0, 1)
    if s.min() >= 1:
        return B
    A = _morph(k - 1, t)
    e = s * s * (3 - 2 * s)
    e2 = e[:, None]
    bump = np.sin(np.pi * e)[:, None]
    pos = A[0] * (1 - e2) + B[0] * e2 + SWIRL * bump
    col = A[1] * (1 - e2) + B[1] * e2
    b = A[2] * (1 - e) + B[2] * e + 0.04 * bump[:, 0] * (A[2] + B[2])
    return pos.astype(np.float32), col.astype(np.float32), b.astype(np.float32)


def swarm(t):
    k = 0
    while k + 1 < len(KEYS) and KEYS[k + 1][0] <= t:
        k += 1
    return _morph(k, t)


# ------------------------------------------------------------------ vines (S7)
def gen_vines():
    rng = np.random.default_rng(11)
    tab_tau = np.linspace(0, 12, 6000)
    tab_s = 150 * tab_tau + 26 * tab_tau ** 3
    xs, ys, bs, kinds, leaf_ids = [], [], [], [], []
    leaf_counter = [0]

    def grow(x, y, ang, t0, depth, length):
        s, curv, nextb, nextleaf = 0.0, rng.normal(0, 0.004), rng.uniform(70, 150), rng.uniform(20, 40)
        side = 1
        ds = 1.6
        while s < length:
            tb = t0 + np.interp(s, tab_s, tab_tau)
            th = 1 if depth > 0 else 2
            for o in range(th):
                xs.append(x + o * math.cos(ang + 1.57)); ys.append(y + o * math.sin(ang + 1.57))
                bs.append(tb); kinds.append(0); leaf_ids.append(-1)
            curl = 0.0
            if length - s < 70:
                curl = 0.05 * (1 - (length - s) / 70) * 3
            curv += rng.normal(0, 0.0025)
            curv *= 0.985
            if depth == 0:
                ang += 0.012 * (-math.pi / 2 - ang)
            ang += (curv + curl * (1 if depth % 2 else -1)) * ds
            x += math.cos(ang) * ds
            y += math.sin(ang) * ds
            s += ds
            if s > nextleaf:
                nextleaf += rng.uniform(22, 42)
                side = -side
                la = ang + side * rng.uniform(0.7, 1.1)
                ll = rng.uniform(9, 17)
                lid = leaf_counter[0]
                leaf_counter[0] += 1
                for q in np.linspace(0, TWO_PI, 22, endpoint=False):
                    lx = (math.cos(q) + 1) * ll / 2
                    ly = math.sin(q) * ll * 0.28
                    xs.append(x + lx * math.cos(la) - ly * math.sin(la))
                    ys.append(y + lx * math.sin(la) + ly * math.cos(la))
                    bs.append(tb + 0.12); kinds.append(1); leaf_ids.append(lid)
            if depth < 3 and s > nextb and length - s > 60:
                nextb += rng.uniform(80, 170) * (1 + depth * 0.5)
                grow(x, y, ang + rng.choice([-1, 1]) * rng.uniform(0.5, 1.1), tb, depth + 1,
                     (length - s) * rng.uniform(0.45, 0.75))
        tb = t0 + np.interp(s, tab_s, tab_tau)
        for _ in range(14):
            xs.append(x + rng.normal(0, 3)); ys.append(y + rng.normal(0, 3))
            bs.append(tb + 0.3); kinds.append(2); leaf_ids.append(-1)

    roots = [(x, H + 10, -math.pi / 2 + rng.normal(0, 0.35)) for x in np.linspace(60, 1860, 10)]
    roots += [(-10, y, -0.35 + rng.normal(0, 0.2)) for y in (300, 620, 880)]
    roots += [(W + 10, y, math.pi + 0.35 + rng.normal(0, 0.2)) for y in (420, 760)]
    for i, (x, y, a) in enumerate(roots):
        grow(x, y, a, 153.0 + rng.uniform(0, 0.8) + 0.08 * i, 0, rng.uniform(900, 1500))
    lid = np.array(leaf_ids)
    return (np.array(xs, np.float32), np.array(ys, np.float32), np.array(bs, np.float32),
            np.array(kinds), lid)


VX, VY, VB, VK, VL = gen_vines()
_rv = np.random.default_rng(23)
NLEAF = VL.max() + 2
LEAF_FALL = 161.4 + _rv.random(NLEAF) * 6.0
LEAF_PH = _rv.random(NLEAF) * 6.28
LEAF_G = 120 + 160 * _rv.random(NLEAF)
VU = _rv.random(len(VX)).astype(np.float32)
V_GREEN, V_RUST, V_SEED = hexc("4f8a4a"), hexc("c0602a"), hexc("f2c46a")


def vines(t):
    m = VB <= t
    x, y, k, lid, u = VX[m], VY[m], VK[m], VL[m], VU[m]
    age = t - VB[m]
    autumn = sst(160.6 + 0.8 * u, 162.8 + 0.8 * u, t)[:, None]
    col = V_GREEN * (1 - autumn) + V_RUST * autumn
    col = np.where((k == 2)[:, None], V_SEED, col)
    b = np.where(k == 0, 0.30, np.where(k == 1, 0.22, 0.0))
    seed_b = 0.5 * sst(156.8, 158.5, t) * (0.7 + 0.3 * np.sin(6 * t + u * 20))
    b = np.where(k == 2, seed_b, b)
    b = b * (1 + 2.5 * np.exp(-age / 0.12))
    isleaf = k == 1
    li = np.maximum(lid, 0)
    ftau = np.clip(t - LEAF_FALL[li], 0, None)
    x = np.where(isleaf, x + 30 * np.sin(1.7 * ftau + LEAF_PH[li]) * np.minimum(ftau, 1.5) + 20 * ftau, x)
    y = np.where(isleaf, y + 0.5 * LEAF_G[li] * ftau ** 2, y)
    glob = 1 - 0.6 * sst(163.6, 166.0, t)
    return np.stack([x, y], 1), col, (b * glob).astype(np.float32)


# ------------------------------------------------------------------ text-dust (stanza dissolves)
def stanza_mask(si):
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for li in range(len(STANZAS[si])):
        xs, _ = char_xs(si, li, FINAL_TEXT[si][li])
        _, yt = line_xy(si, li)
        for ch, x in zip(FINAL_TEXT[si][li], xs):
            d.text((x, yt), ch, font=F_POEM, fill=255)
    return np.asarray(im)


def title_mask():
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    x = 960 - 3 * 112 * 1.45 / 2 + 112 * 0.225
    for i, ch in enumerate(TITLE):
        d.text((x + i * 112 * 1.45, 540 - 90), ch, font=F_TITLE, fill=255)
    return np.asarray(im)


def dust_from_mask(mask, n, seed):
    pts = np.argwhere(mask > 100)[:, ::-1].astype(np.float32)
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(pts), n)
    p = pts[idx] + r.random((n, 2)).astype(np.float32)
    return p, r.random((n, 6)).astype(np.float32)


ND = 7000
TDUST = {si: dust_from_mask(stanza_mask(si), ND, 100 + si) for si in DISSOLVE}
TITLE_DUST = dust_from_mask(title_mask(), 9000, 99)


def text_dust(t, pts, u, t_d, spread=0.9):
    xn = np.clip((pts[:, 0] - LEFT) / 1200, 0, 1)
    tau = t - (t_d + spread * xn + 0.12 * u[:, 0])
    m = (tau > 0) & (tau < 4.5)
    if not m.any():
        return None
    tau, p, uu = tau[m], pts[m], u[m]
    vx, vy = 25 + 70 * uu[:, 1], -15 - 45 * uu[:, 2]
    x = p[:, 0] + vx * tau + 0.5 * 30 * tau ** 2 + 22 * tau * np.sin(3 * tau + uu[:, 3] * 6)
    y = p[:, 1] + vy * tau - 0.5 * 18 * tau ** 2 + 22 * tau * np.cos(2.3 * tau + uu[:, 4] * 6)
    b = 0.55 * np.exp(-tau / 1.1) * (0.4 + 0.6 * uu[:, 5])
    k = smooth(tau / 1.0)[:, None]
    col = np.array([0.95, 0.91, 0.85], np.float32) * (1 - k) + C_LAV[: len(tau)] * k
    return np.stack([x, y], 1), col, b.astype(np.float32)


# ------------------------------------------------------------------ background
BG_KEYS = [
    (0, "000000", "000000"), (2.5, "000000", "000000"), (8, "08050f", "120a1c"), (12, "140d26", "3d2a55"),
    (33, "140d26", "3d2a55"), (37, "170b1c", "4a2236"), (55, "170b1c", "4a2236"), (59, "05070f", "141832"),
    (85, "05070f", "141832"), (89, "020208", "0c0a1e"), (100, "020208", "0c0a1e"), (104, "060a14", "1c2638"),
    (121, "060a14", "1c2638"), (124, "120a10", "38242a"), (137.5, "120a10", "38242a"), (139.5, "1a120a", "3d2c14"),
    (142.5, "0c0f16", "1d2836"), (149.99, "0c0f16", "1d2836"), (150, "000000", "000000"),
    (152.5, "000000", "000000"), (155, "020402", "0b120b"), (160, "020402", "0b120b"),
    (163, "0a0503", "221109"), (170.39, "0a0503", "221109"), (170.4, "000000", "000000"),
    (176, "000000", "000000"), (178.6, "0f2a4d", "5b8fbf"), (186, "16376a", "7fb3df"),
    (194, "1a3f78", "8cc0e8"), (199, "0c1f3f", "3d6a9a"), (205, "000000", "000000"), (999, "000000", "000000"),
]
_BG_T = np.array([k[0] for k in BG_KEYS])
_BG_A = np.array([hexc(k[1]) for k in BG_KEYS])
_BG_B = np.array([hexc(k[2]) for k in BG_KEYS])
_YY, _XX = np.mgrid[0:LHh, 0:LW].astype(np.float32)
_YN = (_YY / (LHh - 1))[..., None]


def background(t):
    top = np.array([np.interp(t, _BG_T, _BG_A[:, c]) for c in range(3)], np.float32)
    bot = np.array([np.interp(t, _BG_T, _BG_B[:, c]) for c in range(3)], np.float32)
    img = top + (bot - top) * _YN ** 1.25
    gx, gy = CX / 4 + 6 * math.sin(0.1 * t), CY / 4
    d2 = ((_XX - gx) ** 2 + (_YY - gy) ** 2) / (95.0 ** 2)
    glow = np.exp(-d2)[..., None] * (bot * 0.35 + 0.01)
    w = 0.5 + 0.5 * np.sin(_XX * 0.03 + 0.2 * t) * np.sin(_YY * 0.05 - 0.13 * t)
    return (img + glow + (w[..., None] * 0.04) * bot).astype(np.float32)


# ------------------------------------------------------------------ overlays
def _grid_layer():
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    rows = []
    phrase = "你最好了"
    y = 14
    r = 0
    while y < H:
        s = (phrase * 60)[(r * 3) % 4:]
        d.text((10 - (r % 2) * 9, y), s, font=F_TINY, fill=255)
        rows.append(y)
        y += 25
        r += 1
    a = np.asarray(im, np.float32) / 255
    rowmap = (np.arange(H) // 25).astype(np.float32)
    return a, rowmap


GRID_A, GRID_ROW = _grid_layer()
NROWS = GRID_ROW.max() + 1


def _stain_fields():
    r = np.random.default_rng(31)
    n = np.zeros((LHh, LW), np.float32)
    for s in (40, 16, 6):
        z = r.random((LHh // 2 + 2, LW // 2 + 2)).astype(np.float32)
        z = cv2.GaussianBlur(z, (0, 0), s / 2)
        z = cv2.resize(z, (LW, LHh), interpolation=cv2.INTER_CUBIC)
        n += (z - z.mean()) / (z.std() + 1e-6) * (s / 40)
    cents = [(110, 120, 70), (230, 150, 55), (340, 90, 80), (180, 60, 40), (400, 200, 50)]
    rad = np.zeros_like(n)
    for (cx, cy, rr) in cents:
        rad = np.maximum(rad, np.exp(-((_XX - cx) ** 2 + (_YY - cy) ** 2) / (2 * rr ** 2)))
    return n, rad


STAIN_N, STAIN_R = _stain_fields()
STAIN_COL = hexc("3a2614")


def stain_alpha(t):
    g = sst(137.6, 140.2, t)
    wsh = sst(141.6, 144.2, t)
    lvl = g * (1 - wsh)
    if lvl <= 0.001:
        return None
    v = STAIN_R * 2.2 + STAIN_N * 0.5 - (2.4 - 1.6 * lvl)
    a = np.clip(v * 3, 0, 1) * lvl
    return a


_Rv = np.hypot((np.arange(W) - W / 2) / (W / 2), 0)[None, :]
_VIG = (1 - 0.42 * np.clip(np.hypot(*np.meshgrid((np.arange(W) - W / 2) / (W / 2) * 1.0,
                                                   (np.arange(H) - H / 2) / (H / 2) * 0.9)) / 1.35, 0, 1) ** 2.2
        ).astype(np.float32)[..., None]


SHARP_GAIN = 3.2


def tonemap(f):
    over = f > 0.72
    f2 = np.where(over, 0.72 + 0.28 * (1 - np.exp(-(f - 0.72) / 0.28)), f)
    return f2


def ca_shift(f, px):
    px = int(round(px))
    if px <= 0:
        return f
    out = f.copy()
    out[:, px:, 0] = f[:, :-px, 0]
    out[:, :-px, 2] = f[:, px:, 2]
    return out


def glitch_bands(f, amt, seed):
    if amt <= 0:
        return f
    r = np.random.default_rng(seed)
    nb = int(3 + amt * 14)
    for _ in range(nb):
        y0 = r.integers(0, H - 10)
        hh = int(r.integers(4, 10 + int(amt * 70)))
        sh = int(r.normal(0, 10 + amt * 120))
        f[y0:y0 + hh] = np.roll(f[y0:y0 + hh], sh, axis=1)
        if r.random() < 0.3 * amt:
            ch = r.integers(0, 3)
            f[y0:y0 + hh, :, ch] *= r.uniform(0.2, 2.2)
    return f


# ------------------------------------------------------------------ RAW (我为什么在写作)
def raw_lines():
    def wrap(s, maxw=1740):
        out, cur, w = [], "", 0
        for ch in s:
            cw = F_MONO.getlength(ch)
            if w + cw > maxw:
                out.append(cur)
                cur, w = "", 0
            cur += ch
            w += cw
        if cur:
            out.append(cur)
        return out
    L = []
    for _ in range(5):
        L += wrap(RAW_SOURCE) + [""]
    return L


RAW_LINES = raw_lines()
CORRUPT = list("我为什么在写作薰衣草晚安") + ["<br/>", "&nbsp;", "　", "█", "▓", "？"]


def raw_frame(t):
    tt = t - RAW[0]
    fi = int(t * FPS)
    r = np.random.default_rng(fi)
    im = Image.new("RGB", (W, H), (6, 5, 9))
    d = ImageDraw.Draw(im)
    lh = 40
    scroll = 60 * tt + 230 * tt ** 2
    p = 0.01 + 0.55 * sst(1.6, 4.7, tt)
    first = int(scroll // lh)
    fill_all = tt > 4.3
    for i in range(first, first + H // lh + 2):
        y = 60 + i * lh - scroll
        if fill_all:
            s = ("我为什么在写作" * 40)[(i * 3) % 7:]
        else:
            s = RAW_LINES[i % len(RAW_LINES)]
            if p > 0.011:
                s = "".join(CORRUPT[r.integers(len(CORRUPT))] if (r.random() < p and c != " ") else c for c in s)
        x = 90
        k = s.find("我为什么在写作")
        if k >= 0 and not fill_all:
            x0 = 90 + F_MONO.getlength(s[:k])
            d.rectangle([x0 - 4, y - 2, x0 + F_MONO.getlength("我为什么在写作") + 4, y + 34], fill=(150, 120, 230))
        col = (232, 224, 210) if not fill_all else (200, 180, 255)
        d.text((x, y), s, font=F_MONO, fill=col)
    f = np.asarray(im, np.float32) / 255
    return f


# ------------------------------------------------------------------ CODA
SIG2 = "spectacle Lore"


def _anagram_map():
    used = [False] * len(SIG2)
    mp = []
    for ch in SIGNATURE:
        for j, c2 in enumerate(SIG2):
            if not used[j] and c2 == ch:
                used[j] = True
                mp.append(j)
                break
    return mp


ANA = _anagram_map()
SIG_T0 = 207.5
SIG_TIMES = [SIG_T0 + i * 0.125 + 0.03 * math.sin(i * 2.3) for i in range(len(SIGNATURE))]


def coda_frame(t):
    f = np.zeros((H, W, 3), np.float32)
    adv = F_SIG.getlength("M")
    x0a = 960 - adv * len(SIGNATURE) / 2
    x0b = 960 - adv * len(SIG2) / 2
    y = 540 - 34
    g1 = sst(211.6, 212.3, t) * (1 - sst(213.1, 213.8, t))
    fade = 1 - sst(215.2, 218.3, t)
    col = np.array([0.88, 0.85, 0.95], np.float32)
    for i, ch in enumerate(SIGNATURE):
        if t < SIG_TIMES[i]:
            break
        op = smooth((t - SIG_TIMES[i]) / 0.06) * fade
        stg = np.clip(g1 * 1.6 - 0.6 * (i / len(SIGNATURE)), 0, 1)
        e = smooth(stg)
        xa, xb = x0a + i * adv, x0b + ANA[i] * adv
        x = xa + (xb - xa) * e
        yy = y - 38 * math.sin(math.pi * e) * (1 if i % 2 else -1)
        al, gl, _, pad = glyph(ch, "sig")
        fx, fy = x - pad, yy - pad
        ix, iy = int(math.floor(fx)), int(math.floor(fy))
        al2 = shift_sub(al, fx - ix, fy - iy)
        gl2 = shift_sub(gl, fx - ix, fy - iy)
        blit_add(f, gl2, ix, iy, GLOW_COL, 0.25 * op)
        blit_over(f, al2, ix, iy, col, op)
    n_typed = sum(1 for tt in SIG_TIMES if t >= tt)
    if t > SIG_T0 - 1.0 and t < 218.0:
        idle = t - SIG_TIMES[max(0, n_typed - 1)] if n_typed else 9
        on = (idle < 0.4 and n_typed) or ((t * 1.1) % 1.0) < 0.55
        if on and g1 < 0.01:
            cx = int(x0a + n_typed * adv + 4)
            f[y + 8:y + 62, cx:cx + 24] = CURSOR_COL * 0.75 * fade
    return f


# ------------------------------------------------------------------ frame assembly
def world_time(t):
    if FREEZE[0] <= t < FREEZE[1]:
        return FREEZE[0]
    return t


def splat(groups, w, h, scale):
    xs, ys, ws, cs = [], [], [], []
    for pos, col, b in groups:
        xs.append(pos[:, 0] * scale)
        ys.append(pos[:, 1] * scale)
        ws.append(b)
        cs.append(col)
    x, y = np.concatenate(xs), np.concatenate(ys)
    wt, col = np.concatenate(ws), np.concatenate(cs)
    m = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1) & (wt > 1e-4)
    x, y, wt, col = x[m], y[m], wt[m], col[m]
    x0, y0 = x.astype(np.int64), y.astype(np.int64)
    fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    i00 = (y0 * w + x0) * 3
    idx = np.concatenate([i00, i00 + 3, i00 + 3 * w, i00 + 3 * w + 3])
    ww = np.concatenate([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]) * np.tile(wt, 4)
    cc = np.tile(col, (4, 1))
    idx3 = np.concatenate([idx, idx + 1, idx + 2])
    w3 = np.concatenate([ww * cc[:, 0], ww * cc[:, 1], ww * cc[:, 2]])
    acc = np.bincount(idx3, w3, minlength=w * h * 3).astype(np.float32)
    return acc.reshape(h, w, 3)


def hush_frame(t):
    f = np.zeros((H, W, 3), np.float32)
    buf = line_state(5, 5, CUT)
    xs, xend = char_xs(5, 5, [c for c, _ in buf])
    _, yt = line_xy(5, 5)
    if ((t - CUT) % 1.0) < 0.5 and t > CUT + 0.25:
        x0, y0 = int(xend + 4), int(yt + 6)
        f[y0:y0 + 48, x0:x0 + 3] = CURSOR_COL * 0.8
    return f


_gr = np.random.default_rng(77)
GRAIN = [cv2.resize(_gr.standard_normal((H // 2, W // 2)).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) for _ in range(2)]


def post(f, t, fi, ca=1.0, glitch=0.0, grain=0.011):
    f = tonemap(f)
    f = ca_shift(f, ca)
    f *= _VIG
    k = (fi // 3) % 2 if glitch > 0 or grain > 0.015 else 0
    f += (GRAIN[k] * grain * 0.8)[..., None]
    if glitch > 0:
        f = glitch_bands(f, glitch, fi)
    return cv2.convertScaleAbs(f, alpha=255.0)


def render(fi):
    t = fi / FPS
    if CUT <= t < SEC["HUSH"][1]:
        return post(hush_frame(t), t, fi, ca=0, grain=0.02)
    if RAW[0] <= t < RAW[1]:
        tt = t - RAW[0]
        f = raw_frame(t)
        return post(f, t, fi, ca=2 + 8 * sst(2.5, 5.0, tt), glitch=0.15 + 0.85 * sst(2.0, 5.0, tt), grain=0.07)
    if RAW[1] <= t < 176.0:
        return post(np.zeros((H, W, 3), np.float32), t, fi, ca=0, grain=0.02)
    if t >= SEC["CODA"][0]:
        g = 0.0
        if 211.5 < t < 213.9:
            g = 0.25 * window(t, 211.5, 213.9, 0.3, 0.3)
        return post(coda_frame(t), t, fi, ca=1 + 3 * g * 4, glitch=g, grain=0.03)
    wt = world_time(t)
    low = background(wt)
    groups = [swarm(wt)]
    if 152.9 <= t < RAW[0]:
        groups.append(vines(t))
    for si, td in DISSOLVE.items():
        if td <= t < td + 5.5:
            pts, u = TDUST[si]
            r = text_dust(t, pts, u, td)
            if r is not None:
                groups.append(r)
    if TITLE_DISSOLVE <= t < TITLE_DISSOLVE + 6:
        pts, u = TITLE_DUST
        r = text_dust(t, pts, u, TITLE_DISSOLVE, spread=0.35)
        if r is not None:
            r = (r[0], r[1], r[2] * 0.8)
            groups.append(r)
    sharp = splat(groups, W, H, 1.0)
    soft = splat(groups, LW, LHh, 0.25)
    soft = (cv2.GaussianBlur(soft, (0, 0), 1.0) * 0.22 + cv2.GaussianBlur(soft, (0, 0), 3.5) * 0.20
            + cv2.GaussianBlur(soft, (0, 0), 12.0) * 0.16)
    low = low + soft
    sa = stain_alpha(t)
    if sa is not None:
        low = low * (1 - sa[..., None] * 0.85) + STAIN_COL * sa[..., None] * 0.6
    if 160.9 <= t < RAW[0]:
        d = np.hypot(_XX - MOON7[0] / 4, _YY - MOON7[1] / 4) / (MOON7[2] / 4)
        occ = (1 - smooth((d - 0.92) / 0.1)) * sst(160.9, 162.4, t)
        low *= (1 - 0.9 * occ)[..., None]
    f = cv2.resize(low, (W, H), interpolation=cv2.INTER_LINEAR)
    f += sharp * SHARP_GAIN
    if sa is not None:
        sab = cv2.resize(sa, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
        f *= 1 - sab * 0.55
    if 160.9 <= t < RAW[0]:
        f[int(MOON7[1] - 140):int(MOON7[1] + 140), int(MOON7[0] - 140):int(MOON7[0] + 140)] *= 1 - 0.92 * cv2.resize(
            occ, (W, H))[int(MOON7[1] - 140):int(MOON7[1] + 140), int(MOON7[0] - 140):int(MOON7[0] + 140), None]
    # title
    if 3.6 <= t < TITLE_DISSOLVE + 1.2:
        x = 960 - 3 * 112 * 1.45 / 2 + 112 * 0.225
        for i, ch in enumerate(TITLE):
            c = "熏" if (i == 0 and 6.0 <= t < 6.0 + 3 / FPS) else ch
            al, gl, _, pad = glyph(c, "title")
            op = sst(3.8 + i * 0.35, 5.2 + i * 0.35, t) * (1 - smooth((t - TITLE_DISSOLVE - 0.35 * (x + i * 162 - LEFT) / 1200) / 0.3))
            xi, yi = int(x + i * 112 * 1.45 - pad), int(540 - 90 - pad)
            blit_add(f, gl, xi, yi, GLOW_COL, 0.35 * op)
            blit_over(f, al, xi, yi, np.array([0.96, 0.92, 0.88], np.float32), op)
    # 你最好了 grid
    if 71.4 <= t < 73.7:
        rev = (71.4 + (GRID_ROW / NROWS) * 1.6)
        vis = (t > rev).astype(np.float32)[:, None, None]
        f += GRID_A[..., None] * vis * np.array([0.5, 0.38, 0.75], np.float32) * 0.55
    # lavare
    if 141.6 <= t < 145.5:
        op = window(t, 141.6, 145.5, 0.8, 1.0) * 0.55
        xx = 1180
        for ch in "lavāre":
            al, gl, adv, pad = glyph(ch, "lat")
            blit_over(f, al, int(xx - pad), int(860 - pad), np.array([0.78, 0.74, 0.95], np.float32), op)
            xx += adv
    draw_poem(f, t)
    ca, gl_ = 1.0, 0.0
    if FREEZE[0] <= t < FREEZE[1]:
        ca = 5.0
        gl_ = 0.35 if (t - FREEZE[0] < 2 / FPS or FREEZE[1] - t < 2 / FPS) else 0.0
        f = f * 0.85 + f.mean(axis=2, keepdims=True) * 0.15
    if 149.3 <= t < 149.36 or 149.5 <= t < 149.56:
        gl_ = 0.3
    if 170.25 <= t < RAW[0]:
        gl_ = 0.6
        ca = 6
    grain = 0.016 if 152.5 <= t < 176 else 0.011
    return post(f, t, fi, ca=ca, glitch=gl_, grain=grain)


# ------------------------------------------------------------------ main
def _worker(fi):
    return render(fi).tobytes()


def render_all(out, t0=0.0, t1=DURATION, workers=4):
    from multiprocessing import Pool
    n0, n1 = int(round(t0 * FPS)), int(round(t1 * FPS))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "19",
           "-pix_fmt", "yuv420p", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    st = time.time()
    with Pool(workers) as pool:
        for i, buf in enumerate(pool.imap(_worker, range(n0, n1), chunksize=4)):
            p.stdin.write(buf)
            if i % 150 == 0:
                el = time.time() - st
                print(f"frame {n0 + i}/{n1}  {el:.0f}s elapsed  eta {el / (i + 1) * (n1 - n0 - i - 1):.0f}s", flush=True)
    p.stdin.close()
    p.wait()


if __name__ == "__main__":
    if sys.argv[1] == "still":
        os.makedirs("stills", exist_ok=True)
        for a in sys.argv[2:]:
            t = float(a)
            st = time.time()
            img = render(int(round(t * FPS)))
            print(a, f"{(time.time() - st) * 1000:.0f} ms")
            Image.fromarray(img).save(f"stills/t{t:07.2f}.png")
    elif sys.argv[1] == "render":
        out = sys.argv[2]
        t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
        t1 = float(sys.argv[4]) if len(sys.argv) > 4 else DURATION
        render_all(out, t0, t1)
