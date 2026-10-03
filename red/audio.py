"""Synthesised score for 《红色》 — organ, choir, 808 / reese bass, DnB drums, a rhythm-game lead that plays the
exact chart the video shows, and a foley layer for every metaphor.   python audio.py out.wav"""
import sys
import wave

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from common import *

NS = int((DURATION + 2) * SR)
RNG = np.random.default_rng(1977)
TAU = 2 * np.pi
BUSES = ["drums", "bass", "music", "sfx"]
DRY = {b: np.zeros((2, NS), np.float32) for b in BUSES}
SEND = {b: np.zeros((2, NS), np.float32) for b in BUSES}
wt = word_time


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tt(n):
    return np.arange(n) / SR


def add(t, sig, bus="music", pan=0.0, gain=1.0, send=0.25):
    sig = np.asarray(sig, np.float32)
    if sig.ndim == 1:
        a = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        sig = np.stack([sig * np.cos(a), sig * np.sin(a)]) * 1.41
    i = int(round(t * SR))
    if i < 0:
        sig, i = sig[:, -i:], 0
    n = min(sig.shape[1], NS - i)
    if n <= 0:
        return
    DRY[bus][:, i:i + n] += sig[:, :n] * gain
    SEND[bus][:, i:i + n] += sig[:, :n] * gain * send


def lp(x, fc, o=2):
    return sosfilt(butter(o, min(fc, SR * 0.45), "low", fs=SR, output="sos"), x)


def hp(x, fc, o=2):
    return sosfilt(butter(o, fc, "high", fs=SR, output="sos"), x)


def bp(x, f1, f2, o=2):
    return sosfilt(butter(o, [max(f1, 10), min(f2, SR * 0.45)], "band", fs=SR, output="sos"), x)


def env(n, att=0.01, rel=0.1, hold=None):
    e = np.ones(n, np.float32)
    na = max(1, int(att * SR))
    e[:min(na, n)] = np.linspace(0, 1, na)[:n]
    if hold is not None:
        hs, nr = int(hold * SR), int(rel * SR)
        if hs < n:
            m = min(nr, n - hs)
            e[hs:hs + m] *= np.linspace(1, 0, nr)[:m] ** 2
            e[hs + m:] = 0
    return e


_SAW = {}


def saw(f, n, ph=0.0, fm=None):
    K = max(1, int(min(10000, SR * 0.45) / f))
    if K not in _SAW:
        x = np.arange(2048) / 2048
        _SAW[K] = (sum(np.sin(TAU * k * x) / k for k in range(1, K + 1)) * 0.6).astype(np.float32)
    tab = _SAW[K]
    p = (ph + (f * tt(n) if fm is None else np.cumsum(f * fm) / SR)) % 1.0
    return np.interp(p * 2048, np.arange(2049), np.append(tab, tab[0])).astype(np.float32)


def noise(n):
    return RNG.standard_normal(n).astype(np.float32)


def mixl(*arrs):
    n = max(len(a) for a in arrs)
    o = np.zeros(n, np.float32)
    for a in arrs:
        o[:len(a)] += a
    return o


# ================================================================= instruments
def organ(m, dur, vel=1.0, att=0.25, rel=1.2):
    f = mtof(m)
    n = int((dur + rel) * SR)
    t = tt(n)
    vib = 1 + 0.003 * np.sin(TAU * 5.5 * t)
    s = np.zeros(n, np.float32)
    for r, a in [(0.5, 0.5), (1, 1.0), (2, 0.6), (3, 0.35), (4, 0.3), (6, 0.12), (8, 0.1)]:
        if f * r < 12000 and not (r == 0.5 and m >= 50):
            s += a * np.sin(TAU * np.cumsum(f * r * vib) / SR + RNG.random() * 6).astype(np.float32)
    leslie = 1 + 0.12 * np.sin(TAU * 6.3 * t)
    return (s * leslie * env(n, att, rel, dur) * vel * 0.18).astype(np.float32)


def choir(m, dur, vel=1.0, att=0.6, rel=1.5, vowel="a"):
    f = mtof(m)
    n = int((dur + rel) * SR)
    t = tt(n)
    out = np.zeros(n, np.float32)
    for v in range(4):
        vib = 1 + 0.006 * np.sin(TAU * (4.8 + 0.3 * v) * t + v) + 0.002 * RNG.standard_normal()
        out += saw(f * (1 + (v - 1.5) * 0.004), n, RNG.random(), fm=vib)
    F = {"a": [(650, 950, 1.0), (1050, 1350, 0.6), (2400, 2800, 0.25)], "o": [(400, 600, 1.0), (750, 950, 0.5), (2300, 2600, 0.15)],
         "u": [(250, 400, 1.0), (600, 800, 0.35), (2200, 2500, 0.1)]}[vowel]
    s = sum(bp(out, a, b) * g for a, b, g in F)
    return (s * env(n, att, rel, dur) * vel * 0.9).astype(np.float32)


def sub808(m, dur=0.9, vel=1.0):
    f = mtof(m)
    n = int(dur * SR)
    t = tt(n)
    ff = f * (1 + 0.9 * np.exp(-t / 0.025))
    s = np.sin(TAU * np.cumsum(ff) / SR) * np.exp(-t / (dur * 0.5)) * (1 - np.exp(-t / 0.002))
    return (np.tanh(s * 2.2) * 0.8 * vel).astype(np.float32)


def reese(m, dur, vel=1.0, cut=900, lfo=0.0):
    f = mtof(m)
    n = int(dur * SR)
    t = tt(n)
    s = saw(f * 2 ** (11 / 1200), n, 0.0) + saw(f * 2 ** (-11 / 1200), n, 0.37) + 0.6 * np.sin(TAU * f / 2 * t)
    c = cut * (1 + 0.6 * np.sin(TAU * lfo * t)) if lfo else np.full(n, cut)
    out = np.zeros(n, np.float32)
    blk = 1024
    for i in range(0, n, blk):
        out[i:i + blk] = lp(s[max(0, i - 256):i + blk], float(c[i]))[-len(s[i:i + blk]):]
    out = np.tanh(out * 1.8) * env(n, 0.005, 0.03, dur - 0.03)
    return (out * vel * 0.5).astype(np.float32)


def kick(vel=1.0):
    n = int(0.45 * SR)
    t = tt(n)
    f = 45 + 110 * np.exp(-t / 0.03)
    s = np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / 0.22)
    cl = hp(noise(n), 2500) * np.exp(-t / 0.003) * 0.4
    return (np.tanh((s + cl) * 1.6) * vel).astype(np.float32)


def snare(vel=1.0):
    n = int(0.35 * SR)
    t = tt(n)
    body = np.sin(TAU * 185 * t) * np.exp(-t / 0.06) * 0.7
    nz = bp(noise(n), 1200, 9000) * np.exp(-t / 0.11)
    return (np.tanh((body + nz) * 1.4) * vel * 0.8).astype(np.float32)


def hat(vel=1.0, open_=False):
    n = int((0.25 if open_ else 0.06) * SR)
    t = tt(n)
    return (hp(noise(n), 7500) * np.exp(-t / (0.07 if open_ else 0.015)) * vel * 0.35).astype(np.float32)


def crash(vel=1.0, dur=2.5):
    n = int(dur * SR)
    t = tt(n)
    return (hp(noise(n), 3000) * np.exp(-t / (dur * 0.35)) * vel * 0.4).astype(np.float32)


def bell(f, vel=1.0, dur=3.0, ratio=3.5, idx=2.5):
    n = int(dur * SR)
    t = tt(n)
    mod = np.sin(TAU * f * ratio * t) * idx * np.exp(-t / (dur * 0.25))
    return (np.sin(TAU * f * t + mod) * np.exp(-t / (dur * 0.35)) * (1 - np.exp(-t / 0.002)) * vel * 0.3).astype(np.float32)


def lead(m, dur=0.22, vel=1.0):
    f = mtof(m)
    n = int((dur + 0.25) * SR)
    t = tt(n)
    s = sum(saw(f * 2 ** (d / 1200), n, RNG.random()) for d in (-14, -6, 0, 6, 14)) / 5
    s += 0.5 * saw(f * 2, n, 0.1)
    out = np.zeros(n, np.float32)
    blk = 512
    for i in range(0, n, blk):
        fc = 900 + 7000 * np.exp(-i / SR / 0.07)
        out[i:i + blk] = lp(s[max(0, i - 256):i + blk], fc)[-len(s[i:i + blk]):]
    return (out * env(n, 0.002, 0.2, dur) * vel * 0.55).astype(np.float32)


def piano(m, vel=1.0, dur=3.5):
    f = mtof(m)
    n = int(dur * SR)
    t = tt(n)
    s = np.zeros(n)
    for k in range(1, 12):
        fk = k * f * np.sqrt(1 + 0.0004 * k * k)
        if fk > 15000:
            break
        s += np.sin(TAU * fk * t + RNG.random()) / k ** 1.2 * np.exp(-k / 6) * np.exp(-t / (2.6 / (1 + 0.35 * k)))
    s = s * (1 - np.exp(-t / 0.003)) + lp(noise(n) * np.exp(-t / 0.006), 1000) * 0.3
    return (lp(s, 3500) * vel * 0.35).astype(np.float32)


def reed(m, dur, vel=1.0, trem=0.0, duty=0.3, att=0.04):
    f = mtof(m)
    n = int((dur + 0.15) * SR)
    t = tt(n)
    vib = 1 + 0.005 * np.sin(TAU * 5.0 * t)
    s = saw(f, n, 0.0, fm=vib) - saw(f, n, duty, fm=vib)
    s = bp(s, 250, 4000) + 0.6 * bp(s, 900, 1500)
    e = env(n, att, 0.12, dur)
    if trem:
        e = e * (1 - trem * (0.5 + 0.5 * np.sin(TAU * 7 * t)))
    return (s * e * vel * 0.3).astype(np.float32)


def chirp(f0, f1, dur, vel=1.0):
    n = int(dur * SR)
    t = tt(n)
    f = f0 * (f1 / f0) ** (t / dur)
    return (np.sin(TAU * np.cumsum(f) / SR) * np.hanning(n) * vel).astype(np.float32)


def swell(dur, f1=300, f2=9000, shape=2.0, rev=False):
    n = int(dur * SR)
    s = bp(noise(n), f1, f2)
    e = np.linspace(0, 1, n) ** shape
    return (s * (e[::-1] if rev else e)).astype(np.float32)


def riser(dur, f0=200, f1=4000):
    n = int(dur * SR)
    t = tt(n)
    f = f0 * (f1 / f0) ** (t / dur)
    s = np.sin(TAU * np.cumsum(f) / SR) * 0.3 + bp(noise(n), 500, 12000) * (t / dur) ** 2 * 0.6
    s += np.sin(TAU * np.cumsum(f * 1.5) / SR) * 0.15
    return (s * (t / dur) ** 1.5).astype(np.float32)


def impact(t0, size=1.0, metal=False, bus="sfx"):
    n = int(4.0 * SR)
    t = tt(n)
    f = 25 + 55 * np.exp(-t / 0.15)
    boom = np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / 1.0)
    cr = hp(noise(n), 1500) * np.exp(-t / 0.7) * 0.5
    s = np.tanh((boom * 1.4 + cr) * 1.3)
    if metal:
        for r, a in [(1, 1), (2.76, 0.6), (5.4, 0.4), (8.9, 0.25), (13.3, 0.15)]:
            s += a * 0.25 * np.sin(TAU * 220 * r * t) * np.exp(-t / (1.5 / r ** 0.5))
    add(t0, s * size, bus, gain=0.9, send=0.5)
    add(t0 - 1.2, swell(1.2, 400, 12000, 3.0) * 0.6 * size, bus, send=0.4)


def glass_shatter(t0, vel=1.0, n_ping=40):
    add(t0, hp(noise(int(0.6 * SR)), 2500) * np.exp(-tt(int(0.6 * SR)) / 0.12) * 0.6 * vel, "sfx", send=0.4)
    for i in range(n_ping):
        add(t0 + RNG.exponential(0.12), bell(RNG.uniform(2500, 7000), vel * 0.25, 0.6, ratio=2.7, idx=1.0),
            "sfx", pan=RNG.uniform(-1, 1), send=0.5)


def heartbeat(t0, vel=1.0):
    for dt, v in ((0, 1.0), (0.26, 0.6)):
        n = int(0.4 * SR)
        t = tt(n)
        f = 34 + 30 * np.exp(-t / 0.04)
        add(t0 + dt, np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / 0.11) * vel * v, "sfx", send=0.1)


# ================================================================= harmony
CH = {"Dm": [50, 57, 62, 65, 69], "Bb": [46, 53, 58, 62, 65], "Gm": [43, 55, 58, 62, 67], "A": [45, 57, 61, 64, 69],
      "Eb": [51, 58, 63, 67, 70], "C": [48, 55, 60, 64, 67], "Dm9": [50, 57, 64, 65, 69], "Bbmaj7": [46, 53, 57, 62, 65],
      "Gm9": [43, 50, 57, 58, 62], "A7sus": [45, 52, 57, 62, 67], "D": [50, 57, 62, 66, 69], "Bm": [47, 54, 59, 62, 66],
      "G": [43, 55, 59, 62, 67], "Amaj": [45, 57, 61, 64, 69]}


def progression():
    P = {}
    for b in range(8, 32):
        P[b] = ["Dm", "Bb", "Gm", "A"][b % 4]
    for b in range(32, 52):
        P[b] = ["Dm", "Eb", "Dm", "C"][b % 4]
    for b in range(52, 72):
        P[b] = ["Dm", "Bb", "Gm", "A"][b % 4]
    for b in range(72, 76):
        P[b] = ["D", "Bm", "G", "Amaj"][b % 4]
    for b in range(76, 80):
        P[b] = ["Bb", "C", "Dm", "A"][b % 4]
    for b in range(80, 98):
        P[b] = ["Dm", "Bb", "Gm", "A"][b % 4]
    for b in range(100, 116):
        P[b] = ["Dm9", "Bbmaj7", "Gm9", "A7sus"][b % 4]
    for b in range(116, 120):
        P[b] = ["Bb", "C", "Dm", "A"][b % 4]
    for b in range(120, 158):
        P[b] = ["Dm", "Bb", "Gm", "Eb"][b % 4]
    for b in range(160, 164):
        P[b] = "Dm"
    for b in range(164, 170):
        P[b] = ["Bbmaj7", "Bbmaj7", "Gm9", "Gm9", "Dm", "Dm"][b - 164]
    return P


PROG = progression()


# ================================================================= arrangement
def sec_drums():
    for (t, k, v) in DRUMS:
        if k == "k":
            add(t, kick(v), "drums", gain=0.9, send=0.05)
        elif k == "s":
            add(t, snare(v), "drums", pan=0.05, gain=0.75, send=0.18)
        else:
            add(t, hat(v * (0.8 + 0.4 * RNG.random()), open_=False), "drums", pan=0.35, gain=0.55, send=0.05)
    for b in (16, 32, 52, 80, 88, 120, 128, 136, 144, 152):
        add(bar(b), crash(0.8), "drums", pan=-0.2, send=0.3)


def sec_bass():
    for b in range(SECTION_BAR["END"]):
        g = groove(b)
        ch = PROG.get(b)
        if not ch:
            continue
        root = CH[ch][0]
        while root > 45:
            root -= 12
        if g in ("dnb",):
            add(bar(b), reese(root + 12, BAR, 1.0, cut=700 + 500 * ((b % 4) / 3), lfo=1 / (BAR / 2)), "bass", gain=0.85, send=0.03)
            add(bar(b), sub808(root + 12, 0.5, 0.7), "bass", gain=0.6, send=0.0)
            add(bar(b) + 10 * STEP, sub808(root + 12, 0.4, 0.6), "bass", gain=0.6, send=0.0)
        elif g in ("half", "sparse"):
            for s in PATTERNS[g]["k"]:
                add(bar(b) + s * STEP, sub808(root + 12, 1.0 if s == 0 else 0.5, 0.95), "bass", gain=0.85, send=0.02)
        elif b < 16 and b >= 8 or 72 > b >= 68:
            add(bar(b), sub808(root + 12, BAR * 0.95, 0.6), "bass", gain=0.7, send=0.05)
        elif g in ("build", "build2"):
            add(bar(b), sub808(root + 12, 1.2, 0.8), "bass", gain=0.8, send=0.05)


def pads():
    for b, ch in PROG.items():
        notes = CH[ch]
        if 8 <= b < 32 or 52 <= b < 72 or 160 <= b < 170:
            inst = "organ"
        elif 100 <= b < 116 or 72 <= b < 76:
            inst = "choirsoft"
        else:
            inst = "organlow"
        if 72 <= b < 76:
            continue
        for i, m in enumerate(notes):
            if inst == "organ":
                add(bar(b), organ(m, BAR, 0.8 if b < 160 else 1.3, att=0.08, rel=0.6), "music", pan=(i / 4 - 0.5) * 0.8, gain=0.3, send=0.45)
            elif inst == "organlow":
                add(bar(b), organ(m, BAR, 0.6, att=0.05, rel=0.4), "music", pan=(i / 4 - 0.5) * 0.8, gain=0.2, send=0.35)
            else:
                add(bar(b), choir(m + 12, BAR, 0.5, att=0.5, rel=1.0, vowel="o"), "music", pan=(i / 4 - 0.5) * 0.9, gain=0.08, send=0.6)


def sec_intro():
    n = int(bar(6) * SR)
    t = tt(n)
    e = (t / t[-1]) ** 2
    dr = organ(38, bar(6) - 0.5, 1.0, att=3.0, rel=0.5)[:n] + organ(45, bar(6) - 0.5, 0.8, att=4.0, rel=0.5)[:n]
    dr2 = organ(51, bar(6) - 0.5, 0.5, att=6.0, rel=0.5)[:n]
    add(0.0, (dr + dr2) * e[:len(dr)], "music", gain=0.8, send=0.5)
    for k in range(4):
        heartbeat(bar(2) + k * 0.86, 0.6 + 0.1 * k)
    add(bar(4.6), riser(bar(1.4), 300, 3000), "sfx", gain=0.35, send=0.4)
    impact(bar(TITLE_BAR), 1.0)
    for m in CH["Dm"]:
        add(bar(TITLE_BAR), choir(m + 12, bar(2), 1.0, att=0.02, rel=1.5), "music", gain=0.1, send=0.6)
        add(bar(TITLE_BAR), organ(m, bar(2), 1.2, att=0.01, rel=1.0), "music", gain=0.5, send=0.6)


def sec_s1():
    for i in range(24):
        t = bar(12) + RNG.random() * bar(4)
        add(t, bell(mtof(RNG.choice([86, 89, 93, 96, 98, 101])), 0.25, 2.0, ratio=2.0, idx=0.8), "sfx", pan=RNG.uniform(-0.9, 0.9), send=0.6)
    for i in range(60):
        t = bar(16) + RNG.random() * bar(4)
        n = int(0.012 * SR)
        add(t, hp(noise(n), 3000) * np.hanning(n) * 0.3, "sfx", pan=RNG.uniform(-1, 1), send=0.05)
    for m in (50, 57, 62):
        add(bar(20), choir(m, bar(4), 0.9, att=2.0, rel=1.5, vowel="a"), "music", gain=0.07, send=0.6)
    for i in range(11):
        t = bar(24) + 0.5 + i * 0.35
        add(t, swell(0.45, 2000, 12000, 1.0) * np.hanning(int(0.45 * SR)), "sfx", pan=-0.8 + 0.15 * i, gain=0.25, send=0.3)
    ts = wt(0, 4, "剐蹭")
    n = int(0.9 * SR)
    t = tt(n)
    f = 2800 + 1600 * np.sin(TAU * 3.3 * t) + 700 * np.sin(TAU * 17 * t)
    scr = np.sin(TAU * np.cumsum(f) / SR) * (0.6 + 0.4 * np.sin(TAU * 41 * t)) + hp(noise(n), 4000) * 0.4
    add(ts, scr * np.hanning(n) * 0.35, "sfx", pan=0.3, send=0.3)
    glass_shatter(ts + 0.25, 0.6, 20)
    th = wt(0, 5, "雷霆")
    n = int(4 * SR)
    t = tt(n)
    rumble = lp(noise(n), 180) * np.exp(-t / 1.5) * 3 + hp(noise(n), 1500) * np.exp(-t / 0.08) * 1.5
    crack = np.zeros(n)
    for k in range(30):
        i = int(RNG.exponential(0.12) * SR)
        if i < n - 400:
            crack[i:i + 400] += noise(400) * np.exp(-np.arange(400) / 60)
    add(th, np.tanh((rumble + crack) * 0.8) * 0.8, "sfx", send=0.4)
    tl = wt(0, 5, "熔岩")
    for i in range(40):
        add(tl + RNG.random() * (bar(32) - tl - 0.5), chirp(RNG.uniform(80, 160), RNG.uniform(250, 500), RNG.uniform(0.04, 0.1), 0.25),
            "sfx", pan=RNG.uniform(-0.5, 0.5), send=0.2)
    sw = wt(0, 5, "甜蜜")
    for k, m in enumerate([86, 89, 93]):
        add(sw + k * STEP * 2, bell(mtof(m), 0.45, 2.5, ratio=1.0, idx=0.5), "music", pan=0.2, send=0.6)


def sec_s2():
    for k in range(4):
        add(bar(32) + k * BAR, swell(BAR, 1500, 10000, 2.5, rev=False) * 0.3, "sfx", pan=np.sin(k), send=0.4)
        for j in range(8):
            add(bar(32) + k * BAR + j * STEP * 2, bell(mtof([62, 65, 69, 74, 77, 81, 86, 89][j]), 0.12, 1.2, ratio=1.0, idx=0.3), "music", pan=-0.6 + 0.15 * j, send=0.5)
    for k in range(16):
        tb = bar(36) + k * BEAT
        add(tb, bell(mtof(RNG.choice([86, 89, 93, 98, 101])), 0.22, 2.5, ratio=3.01, idx=1.6), "sfx", pan=RNG.uniform(-0.7, 0.7), send=0.7)
    add(bar(36) + 2.6, mixl(np.sin(TAU * 70 * tt(int(0.4 * SR))) * np.exp(-tt(int(0.4 * SR)) / 0.08), lp(noise(int(0.2 * SR)), 1200) * 0.4),
        "sfx", gain=0.9, send=0.2)
    tk = wt(1, 3, "下跪") + 0.35
    n = int(1.6 * SR)
    t = tt(n)
    clank = sum(a * np.sin(TAU * 520 * r * t) * np.exp(-t / (0.9 / r)) for r, a in [(1, 1), (2.4, 0.6), (4.1, 0.4), (6.7, 0.3)])
    add(tk, clank * 0.25, "sfx", send=0.5)
    add(tk, bell(mtof(98), 0.4, 3.0, ratio=1.0, idx=0.2), "sfx", send=0.7)
    glass_shatter(wt(1, 3, "神不是神"), 0.8, 30)
    for i in range(10):
        t0 = bar(48) + 0.3 + i * 0.52
        add(t0, mixl(bell(mtof(88), 0.2, 0.5, 2.0, 0.6), np.pad(bell(mtof(93), 0.2, 0.5, 2.0, 0.6), (int(0.08 * SR), 0))), "sfx",
            pan=-0.5 if i % 2 else 0.5, send=0.3)


VOWELS = [(800, 1200), (400, 2000), (300, 900), (600, 1000), (350, 2300), (500, 1600), (700, 1100)]


def sec_s3():
    for i in range(70):
        t0 = bar(52) + RNG.random() * bar(4)
        f0 = RNG.uniform(110, 260)
        d = RNG.uniform(0.08, 0.2)
        n = int(d * SR)
        src = saw(f0, n, RNG.random())
        a, b_ = VOWELS[RNG.integers(len(VOWELS))]
        s = bp(src, a * 0.85, a * 1.15) + 0.6 * bp(src, b_ * 0.9, b_ * 1.1)
        add(t0, s * np.hanning(n) * 0.5, "sfx", pan=RNG.uniform(-1, 1), send=0.5)
    t = bar(56)
    k = 0
    while t < bar(60):
        n = int(0.05 * SR)
        add(t, bp(noise(n), 1500, 7000) * np.hanning(n) * 0.35, "sfx", pan=0.4 * np.sin(k), send=0.1)
        rate = 3 + 22 * min(1, (t - bar(56)) / 3.5)
        t += 1 / rate
        k += 1
    for k in range(int(bar(4) / (BEAT * 2)) + 1):
        heartbeat(bar(60) + k * BEAT * 2, 1.0)
        n = int(0.3 * SR)
        add(bar(60) + k * BEAT * 2 + 0.1, lp(noise(n), 400) * np.hanning(n) * 0.25, "sfx", send=0.2)
    for m in CH["Bbmaj7"]:
        add(bar(64), choir(m + 12, bar(4), 0.6, att=1.0), "music", gain=0.07, send=0.6)
    tr = wt(2, 4, "不允许")
    n = int(1.15 * SR)
    add(tr, np.sin(TAU * 1000 * tt(n)) * env(n, 0.005, 0.01, 1.1) * 0.22, "sfx", send=0.0)
    # 繁星春水 — clean, unforced, in D major
    mel = [(0, 74, 2), (2, 78, 1), (3, 81, 1), (4, 83, 3), (7, 81, 1), (8, 78, 2), (10, 76, 1), (11, 74, 1), (12, 76, 4)]
    for (bb, m, d) in mel:
        add(bar(72) + bb * BEAT, piano(m, 0.42), "music", pan=0.1, send=0.6)
        add(bar(72) + bb * BEAT, bell(mtof(m + 12), 0.1, 3.0, ratio=1.0, idx=0.2), "music", pan=-0.2, send=0.8)
    for k, ch in enumerate(["D", "Bm", "G", "Amaj"]):
        for m in CH[ch][:3]:
            add(bar(72 + k), piano(m - 12, 0.2, 4.0), "music", pan=-0.3, send=0.6)
    for i in range(30):
        add(bar(72) + RNG.random() * bar(4), bell(RNG.uniform(3000, 6000), 0.04, 1.5, 1.0, 0.1), "sfx", pan=RNG.uniform(-1, 1), send=0.9)
    n = int(bar(4) * SR)
    lap = lp(noise(n), 500) * (0.5 + 0.5 * np.sin(TAU * 0.4 * tt(n))) * np.hanning(n)
    add(bar(72), lap * 0.12, "sfx", send=0.4)
    add(bar(76), bell(mtof(105), 0.5, 4.0, ratio=1.0, idx=0.05), "sfx", send=0.7)
    for i in range(80):
        add(bar(76) + RNG.random() * bar(4), hp(noise(80), 5000) * 0.15, "sfx", pan=RNG.uniform(-0.5, 0.5), send=0.2)
    add(bar(76), riser(bar(4), 150, 5000), "sfx", gain=0.45, send=0.4)
    ta = wt(2, 6, "全下")
    for i in range(40):
        add(ta + RNG.random() * 0.4, mixl(bell(RNG.uniform(2000, 4000), 0.25, 0.15, 1.3, 1.5)), "sfx", pan=RNG.uniform(-0.6, 0.6), send=0.2)
    impact(bar(80), 1.2)


def sec_s4():
    t = bar(80)
    while t < bar(84):
        u = (t - bar(80)) / bar(4)
        add(t, hp(noise(200), 3000) * np.hanning(200) * 0.5, "sfx", pan=0.5, send=0.1)
        t += 0.04 + 0.25 * u ** 2
    for i in range(30):
        tb = bar(84) + 0.2 + i * 0.1
        if tb < bar(84) + 3.2:
            add(tb, np.sign(np.sin(TAU * RNG.uniform(800, 2400) * tt(int(0.05 * SR)))) * 0.05, "sfx", pan=0.4, send=0.1)
    n = int(0.6 * SR)
    add(bar(84) + 3.2, (np.sign(np.sin(TAU * 110 * tt(n))) + np.sign(np.sin(TAU * 117 * tt(n)))) * 0.06 * env(n, 0.005, 0.05, 0.55), "sfx", send=0.1)
    for tc in [wt(3, 2, "第一滴泪"), wt(3, 2, "第一次拥吻"), wt(3, 2, "第一次心碎")]:
        add(tc, mixl(snare(1.0), kick(0.8)), "sfx", gain=0.8, send=0.3)
        add(tc, hp(noise(int(0.08 * SR)), 2000) * np.exp(-tt(int(0.08 * SR)) / 0.02) * 0.5, "sfx", send=0.1)
    mel = [(0, 74, 2), (2, 75, 1), (3, 78, 1), (4, 79, 2), (6, 78, 1), (7, 75, 1), (8, 74, 4), (12, 75, 2), (14, 72, 2)]
    for rep in range(2):
        for (bb, m, d) in mel:
            add(bar(92 + 2 * rep) + bb * BEAT * 0.5, reed(m, d * BEAT * 0.5, 0.9), "music", pan=-0.2, gain=0.6, send=0.5)
    for k in range(4):
        add(bar(96) + k * 0.3, mixl(hp(noise(int(0.04 * SR)), 2500) * np.hanning(int(0.04 * SR)) * 0.6), "sfx", pan=-0.4 + 0.2 * k, send=0.1)
    impact(bar(RIVER_BAR), 1.6, metal=True)


def sec_s5():
    n = int(bar(4) * SR)
    t = tt(n)
    wave = lp(noise(n), 900) * np.clip(t / 2.5, 0, 1) ** 2 * np.exp(-np.clip(t - 3.5, 0, None) / 1.2)
    add(bar(100), wave * 0.5, "sfx", send=0.4)
    add(bar(100), lp(noise(n), 120) * 0.4 * np.hanning(n), "sfx", send=0.2)
    glass_shatter(wt(4, 1, "破碎"), 1.0, 50)
    for k in range(int(bar(4) / (BEAT * 2))):
        tb = bar(108) + k * BEAT * 2
        add(tb, mixl(hp(noise(300), 3000) * np.exp(-np.arange(300) / 40) * 0.4), "sfx", pan=0.3 if k % 2 else -0.3, send=0.1)
    tb_ = wt(4, 3, "洗礼")
    n = int((bar(116) - tb_) * SR)
    pour = bp(noise(n), 400, 5000) * np.clip(tt(n) / 0.6, 0, 1) * (0.7 + 0.3 * np.sin(TAU * 11 * tt(n)))
    add(tb_, pour * 0.25, "sfx", send=0.3)
    n = int(bar(4) * SR)
    fire = lp(noise(n), 300) * (0.6 + 0.4 * np.abs(np.sin(TAU * 1.3 * tt(n))))
    add(bar(112), fire * 0.2, "sfx", send=0.3)
    for m in (50, 57, 62, 65):
        add(bar(112), choir(m + 12, bar(4), 0.7, att=1.5, vowel="u"), "music", gain=0.07, send=0.7)


def sec_s6():
    for i in range(160):
        add(bar(116) + RNG.random() * bar(4), bell(RNG.uniform(1500, 4500), 0.08, 0.4, 1.0, 0.2), "sfx", pan=RNG.uniform(-1, 1), send=0.4)
    add(bar(116), riser(bar(4), 200, 6000), "sfx", gain=0.45, send=0.4)
    impact(bar(120), 1.1)
    for (tn, m, lane) in CHART:
        add(tn, lead(m, STEP * 1.4, 0.85), "music", pan=(lane - 1.5) * 0.25, gain=0.42, send=0.25)
        if tn >= bar(152):
            add(tn, lead(m + 12, STEP * 1.2, 0.5), "music", pan=-(lane - 1.5) * 0.3, gain=0.25, send=0.3)
    for b in range(120, 124):
        for s in range(0, 16, 4):
            n = int(0.12 * SR)
            add(bar(b) + s * STEP, bp(noise(n), 600, 1400) * np.exp(-tt(n) / 0.02) * 0.6 + np.sin(TAU * 820 * tt(n)) * np.exp(-tt(n) / 0.03) * 0.4,
                "sfx", pan=-0.3, send=0.3)
    n = int(6 * SR)
    t = tt(n)
    bowl = sum(np.sin(TAU * 216 * r * t) * a * np.exp(-t / 3) for r, a in [(1, 1), (2.71, 0.5), (5.1, 0.25)]) * (1 + 0.3 * np.sin(TAU * 4 * t))
    add(bar(124), bowl * 0.2, "sfx", send=0.6)
    add(bar(124), choir(38 + 12, bar(4), 0.9, vowel="o", att=0.8), "music", gain=0.1, send=0.5)
    n = int(bar(4) * SR)
    crack = np.zeros(n)
    for k in range(250):
        i = RNG.integers(0, n - 300)
        crack[i:i + 300] += noise(300) * np.exp(-np.arange(300) / 30) * RNG.uniform(0.2, 1)
    add(bar(128), (hp(crack, 1500) * 0.25 + lp(noise(n), 250) * 0.3) * 0.7, "sfx", send=0.2)
    for i in range(24):
        add(bar(132) + RNG.random() * bar(4), chirp(RNG.uniform(500, 900), RNG.uniform(1200, 2000), 0.05, 0.2), "sfx", pan=RNG.uniform(-0.6, 0.6), send=0.4)
    for i in range(20):
        t0 = bar(136) + RNG.random() * bar(3)
        for j in range(3):
            add(t0 + j * 0.07, chirp(RNG.uniform(3000, 4500), RNG.uniform(5000, 6500), 0.05, 0.12), "sfx", pan=RNG.uniform(-1, 1), send=0.4)
    tbr = wt(5, 5, "高枝陷落")
    add(tbr, mixl(hp(noise(int(0.3 * SR)), 800) * np.exp(-tt(int(0.3 * SR)) / 0.05), kick(0.6)), "sfx", gain=0.8, send=0.3)
    for i in range(30):
        add(bar(140) + RNG.random() * bar(2), chirp(RNG.uniform(4000, 6000), RNG.uniform(2500, 3500), 0.08, 0.1), "sfx", pan=RNG.uniform(-1, 1), send=0.4)
    glass_shatter(wt(5, 6, "砸碎"), 0.9, 40)
    for k, m in enumerate([62, 65, 69, 67]):
        add(bar(144) + k * BAR, reed(m, BAR * 0.9, 0.7, trem=0.15, duty=0.45, att=0.2), "music", pan=-0.4, gain=0.5, send=0.6)
    toy = [81, 79, 77, 76, 77, 79, 81, 81, 79, 77, 76, 74]
    for k, m in enumerate(toy):
        add(bar(148) + k * BEAT, bell(mtof(m + 12), 0.25, 1.2, ratio=5.0, idx=1.2), "music", pan=0.4, send=0.5)
    for i in range(8):
        n = int(0.5 * SR)
        add(bar(152) + 0.3 + i * 0.6, bp(noise(n), 300, 3000) * np.hanning(n) * 0.5, "sfx", pan=-0.8 + 0.2 * i, send=0.3)
    for m in CH["Dm"]:
        add(bar(156), choir(m + 12, bar(2), 1.0, att=0.3), "music", gain=0.12, send=0.6)
    add(bar(156.5), riser(bar(1.5), 400, 8000), "sfx", gain=0.5, send=0.3)
    n = int(1.2 * SR)
    t = tt(n)
    scratch = np.sin(TAU * np.cumsum(900 * (1 - t / 1.2) ** 2 + 60) / SR) * np.exp(-t / 0.5)
    add(bar(MISS_BAR), scratch * 0.4 + np.sign(np.sin(TAU * 98 * t)) * 0.08 * np.exp(-t / 0.4), "sfx", send=0.2)


def sec_s7():
    impact(bar(160), 1.5)
    n = int(bar(4) * SR)
    t = tt(n)
    roar = lp(noise(n), 1500) * (0.4 + 0.6 * np.exp(-t / 2.5))
    add(bar(160), roar * 0.4, "sfx", send=0.4)
    for m in CH["Dm"] + [74, 77, 81]:
        add(bar(160), choir(m + 12 if m < 70 else m, bar(4), 1.0, att=0.05, rel=2.0), "music", gain=0.08, send=0.6)
    paris = [(0, 74, 1), (1, 76, 1), (2, 77, 2), (4, 79, 1), (5, 77, 1), (6, 76, 2)]
    for (bb, m, d) in paris:
        tb = bar(165) + bb * BEAT * 1.4
        if tb < wt(6, 1, "巴黎") + 0.4:
            add(tb, lp(reed(m, d * BEAT * 1.3, 0.6, trem=0.35, duty=0.25), 2500), "music", pan=0.6, gain=0.4, send=0.6)
    for k, ts in enumerate([bar(170), bar(172), bar(174), bar(176)]):
        add(ts, piano(38, 0.6, 5.0), "music", send=0.7)
        add(ts + 0.05, piano(57 + (5 if k == 3 else 0), 0.35, 5.0), "music", send=0.7)
    for k in range(10):
        heartbeat(bar(176) + k * 0.9, 0.5 * (1 - k / 12))
    add(bar(176), organ(50, bar(4), 0.6, att=2.0, rel=3.0), "music", gain=0.4, send=0.7)
    add(bar(176), organ(69, bar(4), 0.25, att=3.0, rel=3.0), "music", gain=0.4, send=0.7)


def sec_coda():
    n = int(bar(1.5) * SR)
    add(bar(180), swell(bar(1.5), 800, 9000, 1.0) * np.hanning(n), "sfx", gain=0.3, send=0.5)
    n = int(9 * SR)
    t = tt(n)
    gong = sum(a * np.sin(TAU * f * t + np.sin(TAU * 0.3 * t) * 2) * np.exp(-t / d) for f, a, d in
               [(55, 1, 6), (82, 0.7, 5), (143, 0.5, 4), (217, 0.4, 3), (391, 0.3, 2.5), (612, 0.2, 2)])
    gong = gong * (1 - np.exp(-t / 0.4))
    add(bar(181.5), gong * 0.25, "sfx", send=0.6)
    n = int((bar(190) - bar(181.5)) * SR)
    t = tt(n)
    hum = (np.sin(TAU * 36.7 * t) + 0.4 * np.sin(TAU * 73.4 * t) + 0.15 * np.sin(TAU * 110 * t)) * np.clip(t / 3, 0, 1) * np.clip((t[-1] - t) / 3, 0, 1)
    add(bar(181.5), hum * 0.12, "music", send=0.3)
    for i in range(12):
        add(bar(181.5) + i * 0.18, bell(mtof(98 + (i % 4) * 2), 0.1, 3.0, ratio=1.0, idx=0.3), "sfx", pan=-0.6 + 0.1 * i, send=0.8)


# ================================================================= render
def make_ir(rt=2.8, length=4.0):
    n = int(length * SR)
    t = tt(n)
    pre = int(0.018 * SR)
    ir = np.zeros((2, n + pre))
    for c in range(2):
        nz = noise(n)
        ir[c, pre:] = lp(nz, 3500) * np.exp(-6.91 * t / rt) + 0.4 * hp(nz, 3500) * np.exp(-6.91 * t / (rt * 0.35))
        for d in RNG.uniform(0.004, 0.06, 12):
            ir[c, pre + int(d * SR)] += RNG.uniform(-0.6, 0.6)
    ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
    return ir


def gate(x, a, b, fade=0.004):
    i0, i1, f = int(a * SR), int(b * SR), int(fade * SR)
    x[:, i0:i1] = 0
    x[:, max(0, i0 - f):i0] *= np.linspace(1, 0, min(f, i0))
    if i1 + f < x.shape[1]:
        x[:, i1:i1 + f] *= np.linspace(0, 1, f)


def tapestop(x, t0, dur):
    i0, n = int(t0 * SR), int(dur * SR)
    sp = (1 - np.arange(n) / n) ** 1.5
    pos = i0 + np.cumsum(sp)
    for c in range(2):
        x[c, i0:i0 + n] = np.interp(pos, np.arange(x.shape[1]), x[c]) * np.linspace(1, 0, n) ** 0.6
    x[:, i0 + n:i0 + n + int(0.01 * SR)] = 0


def bitcrush(x, a, b, hold=24, levels=24):
    i0, i1 = int(a * SR), int(b * SR)
    seg = x[:, i0:i1]
    idx = (np.arange(seg.shape[1]) // hold) * hold
    seg2 = np.round(seg[:, idx] * levels) / levels
    ramp = np.clip(np.arange(seg.shape[1]) / (0.6 * SR), 0, 1)
    x[:, i0:i1] = seg * (1 - ramp) + seg2 * ramp


def duck(x, kicks, depth=0.5, rel=0.18):
    g = np.ones(x.shape[1], np.float32)
    n = int(0.4 * SR)
    curve = 1 - depth * np.exp(-tt(n) / rel * 3)
    for t in kicks:
        i = int(t * SR)
        m = min(n, len(g) - i)
        if m > 0:
            g[i:i + m] = np.minimum(g[i:i + m], curve[:m])
    return x * g


def compress(x, thr_db=-18, ratio=2.0):
    blk = int(0.005 * SR)
    nb = x.shape[1] // blk
    m = np.sqrt((x[:, :nb * blk].reshape(2, nb, blk) ** 2).mean(axis=(0, 2)) + 1e-12)
    e = np.zeros(nb)
    v = 0.0
    for i in range(nb):
        a = 0.6 if m[i] > v else 0.02
        v += (m[i] - v) * a
        e[i] = v
    db = 20 * np.log10(e + 1e-9)
    gr = np.where(db > thr_db, (thr_db - db) * (1 - 1 / ratio), 0.0)
    g = np.interp(np.arange(x.shape[1]), np.arange(nb) * blk + blk / 2, 10 ** (gr / 20))
    return x * g


def build():
    sec_drums(); sec_bass(); pads(); sec_intro(); sec_s1(); sec_s2(); sec_s3(); sec_s4(); sec_s5(); sec_s6(); sec_s7(); sec_coda()
    ir = make_ir()
    out = {}
    for b in BUSES:
        wet = np.stack([fftconvolve(SEND[b][c], ir[c])[:NS] for c in range(2)]).astype(np.float32)
        out[b] = DRY[b] + wet * 0.5
        print("bus", b, flush=True)
    kick_dnb = [t for t in KICKS if groove(int(t / BAR)) in ("dnb", "half")]
    out["music"] = duck(out["music"], kick_dnb, 0.45)
    out["bass"] = duck(out["bass"], [t for t in KICKS if groove(int(t / BAR)) == "dnb"], 0.6)
    tr = wt(2, 4, "不允许")
    for b in ("drums", "bass", "music"):
        bitcrush(out[b], tr + 0.2, bar(72) - 0.02)
        gate(out[b], bar(72), bar(72) + 0.01)
        gate(out[b], bar(RIVER_BAR) + 0.01, bar(100) - 0.02, fade=0.003)
        tapestop(out[b], bar(MISS_BAR), 0.5)
        gate(out[b], bar(MISS_BAR) + 0.5, bar(160) - 0.01, fade=0.003)
    for b in ("drums", "bass"):
        gate(out[b], bar(170), DURATION + 1)
    mix = out["drums"] * 1.0 + out["bass"] * 1.0 + out["music"] * 1.0 + out["sfx"] * 1.0
    mix = compress(mix)
    mix = np.tanh(mix / np.abs(mix).max() * 1.4) / np.tanh(1.4)
    mix *= 0.9 / np.abs(mix).max()
    end = int(DURATION * SR)
    mix = mix[:, :end]
    mix[:, :int(0.3 * SR)] *= np.linspace(0, 1, int(0.3 * SR))
    mix[:, -int(1.0 * SR):] *= np.linspace(1, 0, int(1.0 * SR))
    return mix


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "score.wav"
    mix = build()
    pcm = (np.clip(mix.T, -1, 1) * 32767).astype("<i2")
    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print("wrote", out, mix.shape[1] / SR, "s")
