"""Synthesised score for 《薰衣草》 — every sound is generated here from numpy.  python audio.py out.wav"""
import sys
import wave

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from common import *

NS = int((DURATION + 1) * SR)
RNG = np.random.default_rng(2024)
TAU = 2 * np.pi
SEG_BOUNDS = [FREEZE[0], CUT, RAW[1]]          # events are routed into 4 independent buses
DRY = [np.zeros((2, NS), np.float32) for _ in range(4)]
SEND = [np.zeros((2, NS), np.float32) for _ in range(4)]


def seg_of(t):
    return sum(t >= b for b in SEG_BOUNDS)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tt(n):
    return np.arange(n) / SR


def add(t, sig, pan=0.0, gain=1.0, send=0.3, seg=None):
    sig = np.asarray(sig, np.float32)
    if sig.ndim == 1:
        a = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        sig = np.stack([sig * np.cos(a), sig * np.sin(a)]) * 1.41
    k = seg_of(t) if seg is None else seg
    i = int(round(t * SR))
    if i < 0:
        sig, i = sig[:, -i:], 0
    n = min(sig.shape[1], NS - i)
    if n <= 0:
        return
    DRY[k][:, i:i + n] += sig[:, :n] * gain
    SEND[k][:, i:i + n] += sig[:, :n] * gain * send


def env_ar(n, att, rel_start=None, rel=0.3):
    e = np.ones(n, np.float32)
    na = max(1, int(att * SR))
    e[:na] = np.linspace(0, 1, na) ** 1.5 if na < n else np.linspace(0, 1, n)[:n]
    if rel_start is not None:
        rs = int(rel_start * SR)
        nr = int(rel * SR)
        if rs < n:
            seg = np.linspace(1, 0, nr) ** 2
            m = min(nr, n - rs)
            e[rs:rs + m] *= seg[:m]
            e[rs + m:] = 0
    return e


def lp(x, fc, order=2):
    return sosfilt(butter(order, min(fc, SR * 0.45), "low", fs=SR, output="sos"), x)


def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, "high", fs=SR, output="sos"), x)


def bp(x, f1, f2, order=2):
    return sosfilt(butter(order, [f1, min(f2, SR * 0.45)], "band", fs=SR, output="sos"), x)


_SAW = {}


def bl_saw(f, n, ph=0.0, fm=None):
    """band-limited saw via wavetable; fm = optional per-sample frequency multiplier"""
    K = max(1, int(min(9000, SR * 0.45) / f))
    if K not in _SAW:
        x = np.arange(2048) / 2048
        _SAW[K] = sum(np.sin(TAU * k * x) / k for k in range(1, K + 1)).astype(np.float32) * 0.6
    tab = _SAW[K]
    if fm is None:
        p = (ph + f * tt(n)) % 1.0
    else:
        p = (ph + np.cumsum(f * fm) / SR) % 1.0
    return np.interp(p * 2048, np.arange(2049), np.append(tab, tab[0])).astype(np.float32)


# ------------------------------------------------------------------ instruments
def musicbox(m, vel=1.0, cents=0.0, length=None):
    f = mtof(m) * 2 ** (cents / 1200)
    T = length or (3.4 if f < 700 else 2.6)
    n = int(T * SR)
    t = tt(n)
    ph = RNG.random() * TAU
    dec = 1.9 if f < 700 else (1.25 if f < 1200 else 0.9)
    s = (np.sin(TAU * f * t) + 0.65 * np.sin(TAU * f * 1.0016 * t + ph)) * np.exp(-t / dec)
    s += 0.20 * np.sin(TAU * 2 * f * t + ph) * np.exp(-t / 0.32)
    if 5.93 * f < 20000:
        s += 0.10 * np.sin(TAU * 5.93 * f * t) * np.exp(-t / 0.06)
    if 9.1 * f < 20000:
        s += 0.04 * np.sin(TAU * 9.1 * f * t) * np.exp(-t / 0.025)
    click = hp(RNG.standard_normal(n) * np.exp(-t / 0.0015), 3000) * 0.10
    s = (s * (1 - np.exp(-t / 0.0012)) + click) * vel
    return s.astype(np.float32)


def glock(f, vel=1.0, dec=1.3):
    n = int((dec * 2.5) * SR)
    t = tt(n)
    s = (np.sin(TAU * f * t) * np.exp(-t / dec) + 0.35 * np.sin(TAU * 2.756 * f * t) * np.exp(-t / (dec * 0.3))
         + 0.12 * np.sin(TAU * 5.404 * f * t) * np.exp(-t / (dec * 0.1)))
    return (s * (1 - np.exp(-t / 0.001)) * vel).astype(np.float32)


def pad_note(m, dur, att=1.0, rel=1.6, fc=1300, det=8, vel=1.0, trem=None):
    f = mtof(m)
    n = int((dur + rel) * SR)
    s = sum(bl_saw(f * 2 ** (d / 1200), n, RNG.random()) for d in (-det, 0, det)) / 3
    s = lp(s, fc)
    e = env_ar(n, att, dur, rel)
    if trem is not None:
        e = e * trem[:n] if len(trem) >= n else e
    return (s * e * vel).astype(np.float32)


def pad_chord(t0, notes, dur, gain=0.044, **kw):
    for i, m in enumerate(notes):
        add(t0, pad_note(m, dur, **kw), pan=(i / max(1, len(notes) - 1) - 0.5) * 0.9, gain=gain, send=0.35)


def choir_note(m, dur, att=1.4, rel=2.0, vel=1.0, detune=0.0, reverse=False):
    f = mtof(m) * 2 ** (detune / 1200)
    n = int((dur + rel) * SR)
    t = tt(n)
    out = np.zeros(n, np.float32)
    for v in range(4):
        vib = 1 + 0.006 * np.sin(TAU * (4.6 + 0.4 * v) * t + v) + 0.003 * (RNG.random() - 0.5)
        out += bl_saw(f * (1 + (v - 1.5) * 0.004), n, RNG.random(), fm=vib)
    form = bp(out, 600, 900) * 1.0 + bp(out, 1050, 1350) * 0.55 + bp(out, 2400, 2800) * 0.22
    e = env_ar(n, att, dur, rel)
    if reverse:
        e = np.linspace(0, 1, n) ** 3
    return (form * e * vel).astype(np.float32)


def felt_piano(m, vel=1.0, dur=3.5):
    f = mtof(m)
    n = int(dur * SR)
    t = tt(n)
    s = np.zeros(n)
    for k in range(1, 13):
        fk = k * f * np.sqrt(1 + 0.00035 * k * k)
        if fk > 16000:
            break
        s += np.sin(TAU * fk * t + RNG.random()) * (1 / k ** 1.25) * np.exp(-k / 5) * np.exp(-t / (3.0 / (1 + 0.4 * k)))
    th = lp(RNG.standard_normal(n) * np.exp(-t / 0.006), 900) * 0.4
    s = (s * (1 - np.exp(-t / 0.003)) + th) * vel
    return lp(s, 2600).astype(np.float32)


def reed(f, dur, vel=1.0, att=0.06, rel=0.12, duty=0.28):
    n = int((dur + rel) * SR)
    vib = 1 + 0.004 * np.sin(TAU * 5.2 * tt(n))
    a = bl_saw(f, n, 0.0, fm=vib)
    b = bl_saw(f, n, duty, fm=vib)
    s = a - b
    s = bp(s, 300, 3800) + 0.7 * bp(s, 950, 1350)
    return (s * env_ar(n, att, dur, rel) * vel * 0.5).astype(np.float32)


def click(vel=1.0, low=False):
    n = int(0.06 * SR)
    t = tt(n)
    nz = bp(RNG.standard_normal(n), 1200 if not low else 600, 5000 if not low else 2500) * np.exp(-t / 0.004)
    th = np.sin(TAU * (190 if not low else 120) * t) * np.exp(-t / 0.018)
    return ((nz * 0.6 + th * 0.5) * vel).astype(np.float32)


def ratchet(vel=1.0):
    n = int(0.05 * SR)
    t = tt(n)
    nz = bp(RNG.standard_normal(n), 2000, 7000) * np.exp(-t / 0.002)
    pg = np.sin(TAU * 2350 * t) * np.exp(-t / 0.012)
    return ((nz + 0.5 * pg) * vel).astype(np.float32)


def thump(f0=60, f1=34, dec=0.18, vel=1.0):
    n = int(dec * 5 * SR)
    t = tt(n)
    f = f1 + (f0 - f1) * np.exp(-t / 0.05)
    s = np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / dec) * (1 - np.exp(-t / 0.003))
    return (s * vel).astype(np.float32)


def noise_swell(dur, f1=400, f2=6000, shape=2.0, rev=False):
    n = int(dur * SR)
    s = bp(RNG.standard_normal(n), f1, f2)
    e = np.linspace(0, 1, n) ** shape
    if rev:
        e = e[::-1]
    return (s * e).astype(np.float32)


def grain(f, dur, q=0.08):
    n = int(dur * SR)
    s = bp(RNG.standard_normal(n), f * (1 - q), f * (1 + q))
    return (s * np.hanning(n)).astype(np.float32)


def mixl(*arrs):
    n = max(len(a) for a in arrs)
    out = np.zeros(n, np.float32)
    for a in arrs:
        out[:len(a)] += a
    return out


def chirp(f0, f1, dur, vel=1.0):
    n = int(dur * SR)
    t = tt(n)
    f = f0 * (f1 / f0) ** (t / dur)
    return (np.sin(TAU * np.cumsum(f) / SR) * np.hanning(n) * vel).astype(np.float32)


# ------------------------------------------------------------------ score data
MEL_A = [(0, 81, 2), (2, 84, 1), (3, 82, 1), (4, 81, 1), (5, 79, 1), (6, 77, 2), (8, 81, 1), (9, 79, 3),
         (12, 81, 2), (14, 84, 1), (15, 86, 1), (16, 84, 1), (17, 82, 1), (18, 79, 1.5), (19.5, 81, 0.5),
         (20, 76, 1), (21, 77, 3)]
MEL_B = [(0, 86, 2), (2, 84, 1), (3, 81, 2), (5, 77, 1), (6, 82, 1), (7, 84, 1), (8, 86, 1), (9, 84, 3),
         (12, 86, 1), (13, 84, 1), (14, 81, 1), (15, 82, 1), (16, 81, 1), (17, 79, 1), (18, 84, 1), (19, 82, 1),
         (20, 76, 1), (21, 77, 3)]
CH_A = ["F", "C7", "F", "C", "F", "Bb", "C7", "F"]
CH_B = ["Dm", "Dm", "Bb", "C", "Dm", "Gm", "C7", "F"]
ACC = {"F": (65, [69, 72]), "C7": (60, [64, 70]), "C": (60, [64, 67]), "Bb": (58, [62, 65]),
       "Dm": (62, [65, 69]), "Gm": (55, [62, 70])}
PADV = {"F": [53, 57, 60, 65], "C7": [48, 55, 58, 64], "C": [48, 55, 60, 64], "Bb": [46, 53, 58, 62],
        "Dm": [50, 57, 62, 65], "Gm": [43, 55, 58, 62], "Bbmaj7": [46, 53, 57, 62], "Fadd9": [53, 57, 60, 67],
        "Dm9": [50, 57, 60, 64], "Csus4": [48, 55, 60, 65], "Cdirty": [48, 49, 55, 61, 64]}
ARP = {"F": [65, 69, 72, 77], "C7": [60, 64, 67, 70], "C": [60, 64, 67, 72], "Bb": [58, 62, 65, 70],
       "Dm": [62, 65, 69, 74], "Gm": [55, 62, 67, 70]}


def play_mel(mel, t0, beat=BEAT, vel=0.26, transpose=0, inst="mb", pan=0.1, send=0.45, cents_fn=None,
             skip=(), ghost=None):
    for i, (b, m, d) in enumerate(mel):
        if i in skip:
            continue
        t = t0 + b * beat
        c = cents_fn(t, i) if cents_fn else 0.0
        if inst == "mb":
            s = musicbox(m + transpose, vel * (0.9 + 0.2 * RNG.random()), c)
        else:
            s = felt_piano(m + transpose, vel * (0.85 + 0.25 * RNG.random()))
            t += RNG.normal(0, 0.02)
        add(t, s, pan=pan + RNG.normal(0, 0.05), send=send)
        if ghost and ghost[0] <= t < ghost[1]:
            add(t + 0.037, musicbox(m + transpose, vel * 0.7, c + 28), pan=-pan - 0.4, send=send)


def play_acc(chords, t0, beat=BEAT, vel=0.10, arp=False, fast=None):
    for k, ch in enumerate(chords):
        tb = t0 + k * 3 * beat
        if not arp:
            r, up = ACC[ch]
            add(tb, musicbox(r, vel), pan=-0.3, send=0.45)
            for j, m in enumerate(up):
                add(tb + (j + 1) * beat, musicbox(m, vel * 0.7), pan=0.3 - 0.2 * j, send=0.45)
        else:
            notes = ARP[ch]
            seq = notes + notes[-2:0:-1]
            nsub = 6
            if fast and fast[0] <= tb < fast[1]:
                nsub = 12
                seq = (notes + notes[::-1]) * 2
            for j in range(nsub):
                add(tb + j * 3 * beat / nsub, glock(mtof(seq[j % len(seq)] + 12), vel * 0.55, dec=0.5),
                    pan=np.sin(j * 0.9) * 0.6, send=0.5)


def pads(chords, t0, dur_each=BAR, **kw):
    for k, ch in enumerate(chords):
        pad_chord(t0 + k * dur_each, PADV[ch], dur_each, **kw)


# ------------------------------------------------------------------ sections
def sec_prologue():
    for turn, ts in enumerate([0.5, 1.75, 3.0]):
        for i in range(9):
            add(ts + i * 0.105 * (1 - 0.02 * i), ratchet(0.16 - 0.007 * i), pan=0.25, send=0.15)
    for (t, m, v) in [(5.0, 81, 0.20), (6.67, 84, 0.17), (7.5, 82, 0.14), (8.33, 81, 0.12)]:
        add(t, musicbox(m, v), pan=0.05, send=0.8)
    add(8.4, noise_swell(1.6, 300, 4000, 2.5), gain=0.04, send=0.6)


def sec_s1():
    t0 = SEC["S1"][0]
    play_mel(MEL_A, t0)
    play_acc(CH_A, t0)
    pads(CH_A, t0)
    for (t, m, v) in [(30.0, 81, 0.18), (31.67, 84, 0.16), (32.5, 83, 0.17), (33.33, 81, 0.13), (34.17, 79, 0.11)]:
        add(t, musicbox(m, v), pan=-0.1, send=0.6)
    pads(["F", "C7"], 30.0)
    add(28.8, noise_swell(0.9, 200, 3000, 3), gain=0.05, send=0.4)
    add(29.62, thump(55, 30, 0.6, 0.55), send=0.3)


def sec_s2():
    t0 = SEC["S2"][0]
    play_mel(MEL_B, t0)
    play_acc(CH_B, t0)
    for k, ch in enumerate(CH_B):
        tb = t0 + k * BAR
        n = int((BAR + 1.6) * SR)
        trem = None
        if 39.5 < tb < 45.5:
            tg = tb + tt(n)
            depth = np.clip((tg - 40.3) / 0.8, 0, 1) * np.clip((45.5 - tg) / 1.0, 0, 1)
            trem = (1 - 0.7 * depth * (0.5 + 0.5 * np.sin(TAU * 9 * tg))).astype(np.float32)
        pad_chord(tb, PADV[ch], BAR, trem=trem)
    for f, g in [(mtof(41), 0.07), (mtof(48), 0.05)]:
        s = reed(f, 12.0, 1.0, att=3.0, rel=2.5)
        add(44.5, s, pan=-0.2 if f < 100 else 0.2, gain=g, send=0.3)
    for (b, m, d) in [(0, 69, 3), (3, 70, 2), (5, 69, 1), (6, 67, 3), (9, 65, 3)]:
        t = 45.0 + b * BEAT
        add(t - 0.07, reed(mtof(m - 2), 0.06, att=0.005, rel=0.03), gain=0.05, pan=0.35, send=0.3)
        add(t, reed(mtof(m), d * BEAT - 0.05), gain=0.06, pan=0.35, send=0.35)
    for k in range(5):
        tb = 45.0 + k * BAR
        add(tb, thump(125, 75, 0.22, 0.35), pan=-0.1, send=0.15)
        add(tb, hp(RNG.standard_normal(int(0.15 * SR)), 2500) * np.exp(-tt(int(0.15 * SR)) / 0.04) * 0.08, send=0.2)
        add(tb + 2 * BEAT, thump(140, 90, 0.12, 0.15), pan=0.1, send=0.15)
    for tb in (55.72, 179.0):
        add(tb, chirp(900, 1350, 0.12, 0.05), pan=0.4, send=0.4)


def sec_s3():
    t0 = SEC["S3"][0]
    for i in range(14):
        add(57.6 + i * 0.06, glock(mtof(96 - i * 2 - (i % 3)), 0.12, 0.8), pan=np.sin(i) * 0.7, send=0.6)
    add(57.4, noise_swell(1.3, 1500, 9000, 0.5, rev=True), gain=0.04, send=0.5)
    play_mel(MEL_A, t0, ghost=(71.4, 73.7))
    play_acc(CH_A, t0, arp=True, fast=(74.9, 80.0))
    pads(CH_A, t0)
    for i in range(9):
        add(62.5 + RNG.random() * 3.2, glock(mtof(RNG.choice([89, 91, 93, 96, 98, 101])), 0.10, 1.0),
            pan=RNG.uniform(-0.8, 0.8), send=0.6)
    for j in range(int(4.0 / (BEAT / 4))):
        tb = 75.0 + j * BEAT / 4
        n = int(0.05 * SR)
        add(tb, hp(RNG.standard_normal(n), 5000) * np.exp(-tt(n) / 0.012) * (0.05 if j % 2 else 0.08), send=0.1, pan=0.3)
    t1 = 77.5
    play_mel([(b - 12, m, d) for (b, m, d) in MEL_A if b >= 12], t1)
    play_acc(CH_A[4:], t1, arp=True, fast=(74.9, 80.0))
    pads(CH_A[4:], t1)
    add(78.8, noise_swell(2.0, 500, 8000, 1.5), gain=0.035, send=0.5)
    for i in range(10):
        add(85.3 + i * 0.13, glock(mtof(98 - i * 3), 0.06, 0.9), pan=0.5 - i * 0.1, send=0.7)


def sec_s4():
    pad_chord(87.5, PADV["Bbmaj7"], 6.0, gain=0.05, att=1.5)
    pad_chord(87.5, [74, 77, 81], 6.0, gain=0.025, att=2.0, fc=2500)
    t = 88.0
    while t < FREEZE[0] - 0.05:
        dens = 1.0 + 13 * np.clip((t - 88) / 5.2, 0, 1) ** 1.6
        add(t, glock(mtof(RNG.choice([89, 91, 93, 96, 98, 100, 101, 103])), 0.05 + 0.04 * RNG.random(), 0.7),
            pan=RNG.uniform(-0.9, 0.9), send=0.7)
        t += RNG.exponential(1 / dens)
    tm = 88.6 + np.cumsum(0.55 * 0.86 ** np.arange(16))
    for i, x in enumerate(tm):
        if x < FREEZE[0]:
            add(x, noise_swell(0.7, 1500 + 200 * i, 7000, 0.6, rev=True) * np.linspace(1, 0.3, int(0.7 * SR)),
                gain=0.02, pan=0.7 - 0.08 * i, send=0.5)
    # 好吧好吧
    st = STANZAS[3][1][1]
    hb = [find_time(3, 1, "好", k) for k in range(2)]
    for k, th in enumerate(hb):
        v = 0.15 if k == 0 else 0.10
        add(th, musicbox(81, v), pan=-0.1, send=0.6)
        add(th + 0.2, musicbox(77, v * 0.9), pan=0.1, send=0.6)
    t1 = 96.25
    play_mel([(b - 12, m, d) for (b, m, d) in MEL_A if b >= 12], t1, beat=0.55, vel=0.2)
    pad_chord(96.25, PADV["F"], 4.6, gain=0.04, att=1.5)
    for i in range(7):
        f0 = RNG.uniform(300, 2200)
        n = int(1.5 * SR)
        tg = tt(n)
        f = 880 + (f0 - 880) * np.exp(-tg / 0.35)
        add(100.75, np.sin(TAU * np.cumsum(f) / SR) * np.hanning(n) * 0.04, pan=RNG.uniform(-0.7, 0.7), send=0.6)
    n = int(5.0 * SR)
    tg = tt(n)
    add(101.6, np.sin(TAU * 880 * tg) * np.exp(-tg / 1.6) * (1 - np.exp(-tg / 0.2)) * 0.05, send=0.6)


def sec_s5():
    t0 = SEC["S5"][0]
    for k, ch in enumerate(["Dm", "Bb", "F", "C"]):
        for i, m in enumerate(PADV[ch]):
            add(t0 + k * 5.0, choir_note(m + 12 if m < 55 else m, 5.0), gain=0.030, pan=(i / 3 - 0.5) * 0.8, send=0.55)
    play_mel(MEL_B, t0, inst="piano", transpose=-12, vel=0.28, send=0.4)
    n = int(14.5 * SR)
    tg = tt(n)
    rain = bp(RNG.standard_normal((2, n)), 700, 7000) * (np.clip(tg / 1.8, 0, 1) * np.clip((14.5 - tg) / 2.0, 0, 1))
    add(107.4, rain * 0.035, send=0.15)
    for i in range(70):
        add(107.6 + RNG.random() * 13.5, chirp(RNG.uniform(2500, 5000), RNG.uniform(1800, 3000), 0.03, 0.03),
            pan=RNG.uniform(-1, 1), send=0.3)
    tw = find_time(4, 2, "木")
    add(tw, mixl(grain(900, 0.06, 0.25) * 3, thump(400, 300, 0.05, 0.3)), gain=0.6, send=0.4)
    add(find_time(4, 3, "象"), felt_piano(84, 0.35), send=0.5)
    add(find_time(4, 3, "璞"), glock(mtof(100), 0.22, 2.5), pan=0.2, send=0.7)
    tm = find_time(4, 3, "大")
    add(tm, mixl(grain(2600, 0.03, 0.3) * 2.5, thump(90, 50, 0.3, 0.3)), send=0.4)
    add(120.0, thump(48, 28, 0.5, 0.45), send=0.2)


def warp_times(t0, t_end_target, nbeats):
    """music-box spring winding down: tempo slows; returns beat->time map such that beat 21 lands after the cut"""
    def sim(k):
        t, b, out = t0, 0.0, []
        dt = 0.002
        bt = [t0]
        while b < nbeats:
            sp = max(0.12, 1 - k * ((t - t0) / (CUT - t0)) ** 1.6)
            b += dt * sp / BEAT
            t += dt
            if b >= len(bt):
                bt.append(t)
        return bt
    lo, hi = 0.0, 0.95
    for _ in range(30):
        mid = (lo + hi) / 2
        bt = sim(mid)
        if bt[21] < t_end_target:
            lo = mid
        else:
            hi = mid
    return sim((lo + hi) / 2)


def sec_s6():
    t0 = 122.9
    bt = warp_times(t0, CUT + 0.35, 24)

    def btime(b):
        i = int(b)
        return bt[i] + (bt[i + 1] - bt[i]) * (b - i)

    for (b, m, d) in MEL_A:
        t = btime(b)
        if t >= CUT:
            continue
        sag = -75 * ((t - 122.5) / 27.5) ** 1.7
        v = 0.25 * (1 - 0.35 * (t - 122.5) / 27.5)
        add(t, musicbox(m, v, sag + RNG.normal(0, 4)), pan=0.1, send=0.5)
    for k, ch in enumerate(CH_A):
        t = btime(k * 3)
        if t >= CUT:
            continue
        sag = -75 * ((t - 122.5) / 27.5) ** 1.7
        r, up = ACC[ch]
        add(t, musicbox(r, 0.09, sag), pan=-0.3, send=0.5)
        if t < 140:
            add(btime(k * 3 + 1), musicbox(up[0], 0.06, sag), pan=0.25, send=0.5)
    for (a, b_, ch) in [(122.5, 131.0, "F"), (131.0, 137.6, "Bb"), (137.6, 141.0, "Cdirty"), (141.0, 144.9, "Gm"),
                        (144.9, 150.5, "Csus4")]:
        pad_chord(a, PADV[ch], b_ - a, gain=0.045 if ch != "Cdirty" else 0.05, att=1.2)
    n = int(4.0 * SR)
    tg = tt(n)
    drone = (bl_saw(mtof(37), n) + bl_saw(mtof(38) * 1.003, n)) * np.sin(np.pi * tg / 4.0) ** 2
    add(137.4, lp(drone, 400) * 0.12, send=0.3)
    for i in range(26):
        add(141.1 + RNG.random() * 3.6, chirp(RNG.uniform(250, 600), RNG.uniform(600, 1200), 0.04, 0.05),
            pan=RNG.uniform(-0.8, 0.8), send=0.4)
    n = int(4.0 * SR)
    slosh = lp(RNG.standard_normal(n), 700) * (0.5 + 0.5 * np.sin(TAU * 0.9 * tt(n))) * np.sin(np.pi * tt(n) / 4.0)
    add(141.0, slosh * 0.09, send=0.3)
    add(145.0, noise_swell(4.9, 3000, 12000, 1.0) * 0.5, gain=0.03, send=0.6)
    # the typing of the confession
    for li in range(6):
        for (te, kind, ch) in OPS[5][li]:
            if te < CUT:
                v = 0.10 if kind == "type" else 0.20
                add(te, click(v, low=(kind == "back")), pan=-0.35, send=0.08)


def sec_s7():
    a = 152.55
    n = int((RAW[0] - a) * SR)
    tg = tt(n)
    sw = np.clip(tg / 0.4, 0, 1)
    drone = (np.sin(TAU * mtof(29) * tg) * 0.5 + lp(bl_saw(mtof(29), n), 180) * 0.6 + 0.35 * np.sin(TAU * mtof(35) * tg))
    drone *= sw * (0.7 + 0.3 * np.clip((tg - 8) / 8, 0, 1))
    add(a, drone * 0.11, send=0.15)
    t = 153.0
    while t < 170.25:
        bpm = 62 + 90 * np.clip((t - 153) / 17, 0, 1) ** 1.4
        add(t, thump(62, 36, 0.12, 0.5), send=0.08)
        add(t + 0.26 * 60 / bpm, thump(55, 34, 0.10, 0.3), send=0.08)
        t += 60 / bpm
    t = 153.0
    while t < 170.0:
        d = 4 + 45 * np.clip((t - 153) / 4.5, 0, 1) - 25 * np.clip((t - 160) / 5, 0, 1)
        add(t, grain(RNG.uniform(260, 1700), RNG.uniform(0.025, 0.09), RNG.uniform(0.03, 0.12)),
            gain=RNG.uniform(0.05, 0.16), pan=RNG.uniform(-1, 1), send=0.25)
        t += RNG.exponential(1 / max(d, 1))
    tmp = np.zeros(int(12 * SR), np.float32)
    for (b, m, d) in MEL_A:
        if b < 12:
            s = musicbox(m - 1, 0.25)
            i = int(b * BEAT * SR)
            tmp[i:i + len(s)] += s[:len(tmp) - i]
    add(155.2, tmp[::-1].copy(), gain=0.5, pan=0.15, send=0.6)
    for i in range(12):
        add(157.0 + RNG.random() * 2.2, glock(mtof(RNG.choice([84, 87, 91])), 0.08, 0.5), pan=RNG.uniform(-0.8, 0.8), send=0.5)
    n = int(4.2 * SR)
    wind = bp(RNG.standard_normal(n), 250, 1600) * np.sin(np.pi * tt(n) / 4.2) ** 2
    add(160.3, wind * 0.10, pan=-0.2, send=0.3)
    for i in range(25):
        add(161.4 + RNG.random() * 6, lp(RNG.standard_normal(int(0.12 * SR)), 3000) * np.hanning(int(0.12 * SR)) * 0.05,
            pan=RNG.uniform(-1, 1), send=0.3)
    for i, m in enumerate([62, 65, 69, 74]):
        add(161.5, choir_note(m, 2.0, att=0.1, rel=0.8, detune=RNG.uniform(-40, 40), reverse=True)[: int(2.8 * SR)],
            gain=0.06, pan=(i / 3 - 0.5), send=0.6)
    n = int(2.8 * SR)
    tg = tt(n)
    river = bp(RNG.standard_normal(n), 180, 1200) * (0.6 + 0.4 * np.sin(TAU * 3.1 * tg)) * np.sin(np.pi * tg / 2.8)
    pan = np.linspace(-0.9, 0.9, n)
    ang = (pan + 1) * np.pi / 4
    add(164.3, np.stack([river * np.cos(ang), river * np.sin(ang)]) * 0.14, send=0.2)
    n = int((170.3 - 166.9) * SR)
    tg = tt(n)
    k = 1 + 14 * (tg / tg[-1]) ** 2
    tone = np.tanh(k * np.sin(TAU * 1046.5 * tg)) / np.tanh(k) * (0.3 + 0.7 * tg / tg[-1])
    tone = lp(tone, 9000) * np.clip(tg / 0.08, 0, 1)
    add(166.9, tone * 0.07, send=0.2)


def sec_raw():
    a, b = RAW
    # source material for stutters: the sweet music box of stanza 1
    src = DRY[0][:, int(10.0 * SR):int(30.0 * SR)].mean(axis=0)
    t = a
    while t < b - 0.02:
        x = (t - a) / (b - a)
        add(t, click(RNG.uniform(0.08, 0.2)), pan=RNG.uniform(-0.6, 0.6), send=0.05, seg=2)
        t += RNG.exponential(1 / (14 + 30 * x))
    t = a + 0.2
    while t < b - 0.05:
        x = (t - a) / (b - a)
        kind = RNG.integers(0, 3)
        if kind == 0:
            gl = int(RNG.uniform(0.03, 0.11) * SR)
            st = RNG.integers(0, len(src) - gl)
            g = src[st:st + gl] * np.hanning(gl)
            rep = int(RNG.integers(2, 7))
            s = np.tile(g, rep)
            q = 2 ** RNG.integers(3, 6)
            s = np.round(s * q) / q
            add(t, s * 1.4, pan=RNG.uniform(-0.8, 0.8), send=0.1, seg=2)
        elif kind == 1:
            n = int(RNG.uniform(0.02, 0.15) * SR)
            s = np.sign(np.sin(TAU * RNG.uniform(300, 3000) * tt(n))) * 0.04
            add(t, s * np.hanning(n) * (0.5 + x), pan=RNG.uniform(-1, 1), send=0.05, seg=2)
        else:
            n = int(RNG.uniform(0.02, 0.12) * SR)
            s = RNG.standard_normal(n)
            s = np.round(s * 4) / 4 * 0.05
            add(t, s, pan=RNG.uniform(-1, 1), send=0.05, seg=2)
        t += RNG.exponential(1 / (5 + 40 * x ** 1.5))
    n = int((b - a) * SR)
    hum = np.sign(np.sin(TAU * 50 * tt(n))) * 0.02 * np.linspace(0.2, 1, n)
    add(a, lp(hum, 800), send=0.0, seg=2)


def sec_s8():
    add(175.5, noise_swell(1.4, 600, 9000, 2.5), gain=0.04, send=0.7, seg=3)
    prog = [(176.0, "Fadd9", 4), (180, "Dm9", 4), (184, "Bbmaj7", 4), (188, "Csus4", 4), (192, "Fadd9", 5),
            (197, "Bbmaj7", 4), (201, "F", 4.5)]
    for (a, ch, d) in prog:
        pad_chord(a, PADV[ch], d, gain=0.05, att=1.4, rel=2.2, fc=1800)
        pad_chord(a, [m + 24 for m in PADV[ch][1:3]], d, gain=0.018, att=2.0, fc=4000)

    def cents(t, i):
        sp = 8 + 40 * np.clip((t - 177) / 22, 0, 1)
        return RNG.uniform(-sp, sp)

    play_mel(MEL_A, 177.0, beat=1.0, vel=0.22, cents_fn=cents, skip=(11, 14), send=0.55)
    for k in range(7):
        tp = 178.2 + k * 1.8
        n = int(1.6 * SR)
        add(tp, np.sin(TAU * 70 * tt(n)) * np.sin(np.pi * tt(n) / 1.6) ** 2 * 0.05, send=0.3)
    add(185.4, noise_swell(4.0, 3000, 14000, 0.7) * np.hanning(int(4.0 * SR)), gain=0.05, send=0.6)
    w = [find_time(7, 3, "晚", k) for k in range(2)] + [find_time(7, 3, "安", k) for k in range(2)]
    for k, (tw, ta) in enumerate([(w[0], w[2]), (w[1], w[3])]):
        v = 0.12 if k == 0 else 0.08
        add(tw, glock(mtof(84), v, 2.2), pan=0.3, send=0.8)
        add(ta, glock(mtof(81), v, 2.6), pan=0.35, send=0.8)
    a, b = GRAVITY, 205.6
    n = int((b - a) * SR)
    tg = tt(n)
    K = 9
    out = np.zeros(n)
    for k in range(K):
        x = (k + tg / 5.5) % K
        f = 35 * 2 ** x
        amp = np.exp(-((x - K / 2) / (K / 4.5)) ** 2)
        out += amp * np.sin(TAU * np.cumsum(f) / SR)
    e = np.clip(tg / 3.5, 0, 1) * np.clip((b - a - tg) / 4.5, 0, 1)
    add(a, out * e * 0.035, send=0.5)
    add(197.4, noise_swell(6.0, 400, 6000, 1.0) * np.hanning(int(6.0 * SR)), gain=0.03, send=0.6)


SIG_TIMES = [207.5 + i * 0.125 + 0.03 * np.sin(i * 2.3) for i in range(len(SIGNATURE))]


def sec_coda():
    for i, t in enumerate(SIG_TIMES):
        add(t, click(0.16), pan=-0.1 + 0.015 * i, send=0.1)
        add(t + 0.005, chirp(1800 + 90 * i, 1200, 0.03, 0.012), send=0.2)
    for i in range(13):
        add(211.6 + i * 0.04, chirp(600 + 40 * i, 1500 + 60 * i, 0.25, 0.012), pan=-0.6 + 0.1 * i, send=0.5)
        add(213.1 + i * 0.04, chirp(1500 + 60 * i, 600 + 40 * i, 0.25, 0.010), pan=0.6 - 0.1 * i, send=0.5)
    add(214.5, musicbox(77, 0.22, -30, length=4.0), send=0.9)


def vinyl(a, b, level):
    n = int((b - a) * SR)
    hiss = hp(RNG.standard_normal((2, n)), 3000) * 0.004
    cr = np.zeros((2, n))
    k = int(9 * (b - a))
    idx = RNG.integers(0, n - 200, k)
    for i in idx:
        L = RNG.integers(20, 120)
        cr[RNG.integers(0, 2), i:i + L] += RNG.standard_normal(L) * np.exp(-np.arange(L) / 15) * RNG.uniform(0.02, 0.15)
    s = hiss + hp(cr, 1200)
    e = np.ones(n)
    f = int(1.0 * SR)
    e[:f] = np.linspace(0, 1, f)
    e[-f:] = np.linspace(1, 0, f)
    add(a, s * e * level, send=0.0)


# ------------------------------------------------------------------ render
def make_ir(rt=3.4, length=5.0):
    n = int(length * SR)
    t = tt(n)
    pre = int(0.022 * SR)
    ir = np.zeros((2, n + pre))
    for c in range(2):
        nz = RNG.standard_normal(n)
        lo = lp(nz, 2200) * np.exp(-6.91 * t / rt)
        hi = hp(nz, 2200) * np.exp(-6.91 * t / (rt * 0.4))
        ir[c, pre:] = lo + 0.5 * hi
        for d in RNG.uniform(0.005, 0.07, 10):
            ir[c, pre + int(d * SR)] += RNG.uniform(-0.6, 0.6)
    ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
    return ir


def tapestop(x, t0, dur):
    i0, n = int(t0 * SR), int(dur * SR)
    sp = (1 - np.arange(n) / n) ** 1.6
    pos = i0 + np.cumsum(sp)
    for c in range(2):
        seg = np.interp(pos, np.arange(len(x[c])), x[c])
        x[c, i0:i0 + n] = seg * np.linspace(1, 0, n) ** 0.7
    return x


def gate(x, a, b, fade=0.004):
    i0, i1, f = int(a * SR), int(b * SR), int(fade * SR)
    x[:, i0:i1] = 0
    x[:, i0 - f:i0] *= np.linspace(1, 0, f)
    if i1 + f < x.shape[1]:
        x[:, i1:i1 + f] *= np.linspace(0, 1, f)


def compress(x, thr_db=-22, ratio=2.2):
    blk = int(0.01 * SR)
    nb = x.shape[1] // blk
    m = np.sqrt((x[:, :nb * blk].reshape(2, nb, blk) ** 2).mean(axis=(0, 2)) + 1e-12)
    env = np.zeros(nb)
    e = 0.0
    for i in range(nb):
        a = 0.5 if m[i] > e else 0.03
        e += (m[i] - e) * a
        env[i] = e
    db = 20 * np.log10(env + 1e-9)
    gr = np.where(db > thr_db, (thr_db - db) * (1 - 1 / ratio), 0.0)
    g = 10 ** (gr / 20)
    g = np.interp(np.arange(x.shape[1]), np.arange(nb) * blk + blk / 2, g)
    return x * g


def build():
    vinyl(0.3, 149.9, 1.0)
    vinyl(176.5, 205.5, 0.5)
    sec_prologue(); sec_s1(); sec_s2(); sec_s3(); sec_s4(); sec_s5(); sec_s6(); sec_s7(); sec_raw(); sec_s8()
    sec_coda()
    ir = make_ir()
    outs = []
    for k in range(4):
        wet = np.stack([fftconvolve(SEND[k][c], ir[c])[:NS] for c in range(2)]).astype(np.float32)
        outs.append(DRY[k] + wet * 0.55)
        print("bus", k, "done", flush=True)
    seg0, seg1, seg2, seg3 = outs
    tapestop(seg0, FREEZE[0] - 0.02, 0.26)
    seg0[:, int((FREEZE[0] + 0.24) * SR):] = 0
    gate(seg1, CUT, DURATION + 1, fade=0.003)
    gate(seg2, RAW[1], DURATION + 1, fade=0.002)
    mix = seg0 + seg1 + seg2 + seg3
    gate(mix, FREEZE[0] + 0.24, FREEZE[1] - 0.02, fade=0.01)
    t_dead = find_time(7, 3, "安", 1) + 0.55
    gate(mix, t_dead, t_dead + 2 / FPS, fade=0.003)
    mix = compress(mix)
    mix = np.tanh(mix / np.abs(mix).max() * 1.15) / np.tanh(1.15)
    mix *= 0.89 / np.abs(mix).max()
    end = int(DURATION * SR)
    mix = mix[:, :end]
    mix[:, -int(0.5 * SR):] *= np.linspace(1, 0, int(0.5 * SR))
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
