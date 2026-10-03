"""《红色》 renderer.   python video.py still t1 t2 ...   |   python video.py render out.mp4 [t0 t1]"""
import math
import os
import subprocess
import sys
import time

import numpy as np

from common import *
import draw as D
import typo

S = dict(black=0, silk=1, slit=2, lava=3, skirt=4, gem=5, cloth=6, tower=7, flesh=8, blobs=9, water=10, felt=11, sea=12,
         rings=13, bluefire=14, tears=15, stream=16, fire=17, heaven=18, smoke=19, redwave=20, haze=21, corona=22,
         altar=23, gradient=24)
SCALE = dict(tower=0.6, gem=0.7, altar=0.7, cloth=0.8)
sst, sm = D.sst, D.sm
wt = word_time

T_SCRATCH = wt(0, 4, "剐蹭")
T_THUNDER = wt(0, 5, "雷霆")
T_LAVA = wt(0, 5, "熔岩")
T_KNEEL = wt(1, 3, "下跪")
T_GODNOT = wt(1, 3, "神不是神")
T_REDACT = wt(2, 4, "不允许")
T_ALLIN = wt(2, 6, "全下")
T_CARDS = [wt(3, 2, "第一滴泪"), wt(3, 2, "第一次拥吻"), wt(3, 2, "第一次心碎")]
T_RIVER = bar(RIVER_BAR)
T_SUNBREAK = wt(4, 1, "破碎")
T_BAPTISM = wt(4, 3, "洗礼")
T_BRANCH = wt(5, 5, "高枝陷落")
T_SMASH = wt(5, 6, "砸碎")
T_MISS = bar(MISS_BAR)


def heart(t, period=2 * BAR / 4):
    ph = (t % period) / period
    return math.exp(-ph * 9) + 0.6 * math.exp(-abs(ph - 0.22) * 14)


def shot(b0, b1, scene, P=(0, 0, 0, 0), Q=(0, 0, 0, 0), draws=(), xf=0.0, wipe=0.0):
    return dict(b0=b0, b1=b1, scene=scene, P=P, Q=Q, draws=list(draws), xf=xf, wipe=wipe)


SHOTS = [
    shot(0, 6, "slit", P=lambda t, lt: (0.06 + 0.45 * sst(bar(2), bar(4.5), t) + 1.7 * sst(bar(5.2), bar(6), t),
                                        sst(0.3, 2.4, t), 0, 0)),
    shot(6, 8, "silk", P=(1.25, 1.2, 1.0, 0)),
    # ---- S1
    shot(8, 12, "silk", P=(1.0, 1.0, 1.0, 0)),
    shot(12, 16, "silk", P=(1.0, 1.6, 1.1, 3.0), draws=[lambda L, t, lt: D.willow(L, t)]),
    shot(16, 20, "silk", P=(0.6, 1.0, 1.3, 7.0), draws=[D.spiders, D.dwarves]),
    shot(20, 24, "altar", P=(1, 0, 0, 0), draws=[D.arms]),
    shot(24, 28, "gradient", P=(0.12, 0.0, 0.02, 0), Q=(0.01, 0, 0, 0), draws=[D.spear_shards]),
    shot(28, T_LAVA / BAR, "black", draws=[lambda L, t, lt: D.lightning(L, t, T_THUNDER)]),
    shot(T_LAVA / BAR, 32, "lava", P=lambda t, lt: (sst(0.0, 1.6, lt), sst(1.0, 3.5, lt), 0, 0)),
    # ---- S2
    shot(32, 36, "skirt", P=lambda t, lt: (1.6, sst(0.5, 5.0, lt), 1.0 + 0.1 * lt / 5, 0)),
    shot(36, 40, "gem", P=lambda t, lt: (0.0, 1.0, 0, 0), draws=[D.certificate]),
    shot(40, 48, "gradient", P=(0.16, 0.0, 0.03, 0), Q=(0.0, 0, 0, 0),
         draws=[lambda L, t, lt: D.halo(L, t, D.kneeler(L, t, lt, T_KNEEL), T_GODNOT)]),
    shot(48, 52, "cloth"),
    # ---- S3
    shot(52, 56, "tower", draws=[D.babel_words]),
    shot(56, 60, "gradient", P=(0.1, 0.0, 0.015, 0), Q=(0.0, 0, 0, 0), draws=[D.pages]),
    shot(60, 64, "flesh", P=lambda t, lt: (heart(t), 0, 0, 0)),
    shot(64, 68, "blobs", P=lambda t, lt: (0.78 - 0.5 * sst(0, 5.5, lt), 0, 0, 0)),
    shot(68, 72, "blobs", P=lambda t, lt: (0.28 - 0.2 * sst(0, 3, lt), 0, 0, 0)),
    shot(72, 76, "water", P=(1, 0, 0, 0), draws=[lambda L, t, lt: D.willow(L, t, D.JADE, 0.55, 0.5, True)]),
    shot(76, 80, "gradient", P=(0.16, 0.0, 0.02, 0), Q=(0.02, 0, 0, 0),
         draws=[D.wineglass, lambda L, t, lt: D.chips(L, t, T_ALLIN)]),
    # ---- S4
    shot(80, 84, "felt", P=(0.55, 0.0, 0, 0), draws=[D.roulette, D.counters]),
    shot(84, 88, "felt", P=(0.55, 0.0, 0, 0), draws=[D.price_rose]),
    shot(88, 92, "felt", P=(0.0, 0.0, 0, 0), draws=[lambda L, t, lt: D.cards3(L, t, T_CARDS)]),
    shot(92, 96, "gradient", P=(0.08, 0.0, 0.01, 0), Q=(0.0, 0, 0, 0), draws=[D.anubis_scale, D.equations]),
    shot(96, 100, "felt", P=(0.0, 0.0, 0, 0), draws=[lambda L, t, lt: D.poker(L, t, lt, T_RIVER, a=1 - sst(T_RIVER + 1.4, bar(100), t))]),
    # ---- S5
    shot(100, 104, "sea", P=lambda t, lt: (0.0, 0.0, sst(0.3, 4.5, lt), 1.0), draws=[D.palms], xf=0.6),
    shot(104, 108, "sea", P=lambda t, lt: (1 - sst(T_SUNBREAK, T_SUNBREAK + 0.05, t), 0.0, 0, 0),
         draws=[lambda L, t, lt: D.palms(L, t, lt, fingers=sst(0, 2, lt)), D.gap, lambda L, t, lt: D.sun_shards(L, t, T_SUNBREAK)]),
    shot(108, 112, "rings", P=lambda t, lt: (1.0, 0.05 + 1.3 * sst(0, 4.5, lt), 0, 0), draws=[D.veins], xf=0.5),
    shot(112, 116, "bluefire", P=lambda t, lt: (1.0, sst(T_BAPTISM, T_BAPTISM + 1.5, t), 0, 0), xf=0.5),
    # ---- S6 (the rhythm game)
    shot(116, 120, "tears", P=lambda t, lt: (1.0, 0.25 + 0.5 * sst(0, 5, lt), 0, 0)),
    shot(120, 124, "stream", draws=[D.monk]),
    shot(124, 128, "gradient", P=(0.2, 0.02, 0.02, 0), Q=(0.02, 0, 0, 0), draws=[D.mandala]),
    shot(128, 132, "fire", P=lambda t, lt: (0.35 + 0.35 * sst(0, 5, lt), 0, 0, 0), draws=[D.burning_verses, D.wadang]),
    shot(132, 136, "gradient", P=(0.12, 0.05, 0.02, 0), Q=(0.0, 0, 0, 0), draws=[D.caffeine]),
    shot(136, 140, "haze", P=(0.4, 0, 0, 0), draws=[lambda L, t, lt: D.birds_branch(L, t, lt, T_BRANCH)]),
    shot(140, 144, "sea", P=(0.0, 0.0, 0.0, 0.0), draws=[D.swallows, lambda L, t, lt: D.rose_shatter(L, t, T_SMASH)]),
    shot(144, 148, "smoke", draws=[D.silhouette]),
    shot(148, 152, "silk", P=(0.9, 1.0, 1.6, 11.0), draws=[D.naive]),
    shot(152, 156, "gradient", P=(0.05, 0.0, 0.01, 0), Q=(0.0, 0, 0, 0), draws=[D.brush]),
    shot(156, 158, "heaven", P=lambda t, lt: (0.3 + 0.3 * sst(0, 2.5, lt), 0, 0, 0), draws=[D.knights]),
    shot(158, 160, "black"),
    # ---- S7
    shot(160, 164, "redwave", P=lambda t, lt: (sst(0.1, 4.6, lt), 0, 0, 0)),
    shot(164, 170, "haze", P=lambda t, lt: (1.0 - 0.35 * sst(4, 8, lt), 0, 0, 0), draws=[D.eiffel], xf=1.2),
    shot(170, 180, "black", xf=1.0),
    # ---- CODA
    shot(180, 181.5, "silk", P=(0.7, 0.6, 0.5, 0), xf=1.2),
    shot(181.5, 190, "corona", P=lambda t, lt: (sst(0.5, 4.0, lt) * (1 - sst(9.0, 11.0, lt)), 0, 0, 0), xf=1.6, wipe=1.0),
]
for i, s in enumerate(SHOTS):
    s["idx"] = i

IMPACTS = [(bar(TITLE_BAR), 1.0, (1, 1, 1)), (T_THUNDER, 1.0, (0.9, 0.88, 1.0)), (bar(32), 0.45, (1, 0.2, 0.2)),
           (T_ALLIN, 0.5, (1, 0.3, 0.3)), (bar(80), 1.0, (1, 1, 1)), (T_RIVER, 1.3, (1, 1, 1)), (T_SUNBREAK, 0.6, (1, 0.6, 0.3)),
           (bar(120), 0.7, (1, 1, 1)), (T_MISS, 1.0, (1, 0.1, 0.15)), (bar(REDWAVE_BAR), 0.8, (1, 0.1, 0.12)),
           (T_KNEEL + 0.35, 0.35, (1, 0.8, 0.6)), (T_GODNOT, 0.5, (1, 0.85, 0.5)), (T_SMASH, 0.5, (1, 0.5, 0.5)),
           (T_BRANCH, 0.4, (1, 1, 1))]
for si in (3, 5):
    for b in LINE_BAR[si][1:]:
        IMPACTS.append((bar(b), 0.25, (1, 1, 1)))
KICK_ARR = np.array(KICKS)
SNARE_ARR = np.array(SNARES)


def find_shot(b):
    cur = SHOTS[0]
    for s in SHOTS:
        if s["b0"] <= b:
            cur = s
    return cur


def params(s, t, lt, key):
    v = s[key]
    return v(t, lt) if callable(v) else v


def last_before(arr, t):
    i = np.searchsorted(arr, t, side="right") - 1
    return t - arr[i] if i >= 0 else 99.0


def fx(t):
    b = t / BAR
    g = groove(int(b))
    out = dict(zoom=1.0, shake=(0.0, 0.0), ca=1.0, flash=(1, 1, 1, 0), shock=(0.5, 0.5, 0.0, 0.0), pix=0.0, crack=0.0,
               glitch=0.0, bloom=0.65, vig=0.38, gain=1.0, thr=0.55)
    if g != "none":
        dk = last_before(KICK_ARR, t)
        amp = {"dnb": 0.024, "half": 0.016, "sparse": 0.01}.get(g, 0.012)
        out["zoom"] += amp * math.exp(-dk / 0.075)
        ds = last_before(SNARE_ARR, t)
        out["ca"] += 3.5 * math.exp(-ds / 0.09)
    sx = sy = 0.0
    for (ti, st, col) in IMPACTS:
        dt = t - ti
        if 0 <= dt < 1.2:
            fa = st * 0.85 * math.exp(-dt / 0.11)
            if fa > out["flash"][3]:
                out["flash"] = (col[0], col[1], col[2], min(0.9, fa))
            r = np.random.default_rng(int(t * 600))
            sx += st * 0.012 * math.exp(-dt / 0.16) * r.normal()
            sy += st * 0.012 * math.exp(-dt / 0.16) * r.normal()
            if st >= 0.6:
                out["shock"] = (0.5, 0.5, dt * 1.3, st * 0.035 * math.exp(-dt / 0.35))
                out["ca"] += 6 * st * math.exp(-dt / 0.2)
    out["shake"] = (sx, sy)
    if T_SCRATCH <= t < bar(32):
        out["crack"] = sst(T_SCRATCH, T_SCRATCH + 0.25, t)
    if T_REDACT <= t < bar(72):
        out["pix"] = 2 + 46 * sst(T_REDACT, T_REDACT + 1.2, t)
    if bar(72) <= t < bar(76):
        out.update(bloom=0.45, vig=0.3, ca=0.6, zoom=1.0)
    for (ti, dur, amt) in [(bar(80), 0.25, 0.4), (T_RIVER, 0.35, 0.6), (T_MISS, 1.4, 0.9), (bar(158.8), 0.8, 0.5)]:
        if ti <= t < ti + dur:
            out["glitch"] = max(out["glitch"], amt * (1 - (t - ti) / dur))
    if t < 1.0:
        out["gain"] = t
    if t > bar(189.2):
        out["gain"] = max(0.0, 1 - (t - bar(189.2)) / (bar(190) - bar(189.2)))
    if bar(160) <= t < bar(164):
        out["bloom"] = 0.9
    return out


# ------------------------------------------------------------------ engine glue
ENG = None
ZERO_U8 = None
STATE = dict(over_zero=True, glow_zero=True)


def init():
    global ENG, ZERO_U8
    from engine import Engine
    ENG = Engine()
    ENG.upload(aux=D.chat_texture()[:, :, :], crack=D.crack_texture())
    ZERO_U8 = np.zeros((H, W, 4), np.uint8)


def run_scene(which, s, t):
    lt = t - bar(s["b0"])
    sc = SCALE.get(s["scene"], 1.0)
    return ENG.scene(which, S[s["scene"]], t, lt, params(s, t, lt, "P"), params(s, t, lt, "Q"), sc)


def render(fi):
    t = fi / FPS
    b = t / BAR
    s = find_shot(b)
    lt = t - bar(s["b0"])
    prev = SHOTS[s["idx"] - 1] if s["idx"] > 0 else None
    mix = 0.0
    if s["xf"] > 0 and lt < s["xf"] and prev is not None:
        sa = run_scene(0, prev, t)
        sb = run_scene(1, s, t)
        mix = sm(lt / s["xf"])
    else:
        sa = run_scene(0, s, t)
        sb = (1.0, 1.0)
    L = D.Layer()
    for d in s["draws"]:
        d(L, t, lt)
    if mix > 0 and prev is not None and mix < 1:
        for d in prev["draws"]:
            plt = t - bar(prev["b0"])
            tmp = D.Layer()
            d(tmp, t, plt)
            if tmp.used:
                L.over = np.maximum(L.over, (tmp.over.astype(np.float32) * (1 - mix)).astype(np.uint8))
                L.used = True
    if bar(116) <= t < T_MISS + 0.8:
        D.rhythm_ui(L, t, a=sst(bar(116), bar(119.5), t))
    over = L.over if L.used else None
    glow = L.glow if L.gused else None
    if over is None and not STATE["over_zero"]:
        over = ZERO_U8
    if glow is None and not STATE["glow_zero"]:
        glow = ZERO_U8
    STATE["over_zero"] = not L.used
    STATE["glow_zero"] = not L.gused
    T = typo.render_text(t)
    ENG.upload(over=over, glow=glow, text=T)
    f = fx(t)
    return ENG.frame(sa, sb, mixAB=mix, wipe=s["wipe"] if mix > 0 else 0.0, zoom=f["zoom"], shake=f["shake"], pix=f["pix"],
                     crack=f["crack"], shock=f["shock"], flash=f["flash"], ca=f["ca"], bloom=f["bloom"], grain=0.045,
                     glitch=f["glitch"], time=t, vig=f["vig"], gain=f["gain"], thr=f["thr"])


def _winit():
    os.environ["LP_NUM_THREADS"] = "1"
    init()


def _work(fi):
    return render(fi).tobytes()


def render_all(out, t0=0.0, t1=DURATION, workers=4):
    from multiprocessing import get_context
    n0, n1 = int(round(t0 * FPS)), int(round(t1 * FPS))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    st = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(workers, initializer=_winit) as pool:
        for i, buf in enumerate(pool.imap(_work, range(n0, n1), chunksize=2)):
            p.stdin.write(buf)
            if i % 150 == 0:
                el = time.time() - st
                print(f"frame {n0 + i}/{n1}  {el:.0f}s  eta {el / (i + 1) * (n1 - n0 - i - 1):.0f}s", flush=True)
    p.stdin.close()
    p.wait()


if __name__ == "__main__":
    if sys.argv[1] == "still":
        from PIL import Image
        init()
        os.makedirs("stills", exist_ok=True)
        for a in sys.argv[2:]:
            t = float(a)
            st = time.time()
            img = render(int(round(t * FPS)))
            print(a, f"{(time.time() - st) * 1000:.0f} ms", flush=True)
            Image.fromarray(img).save(f"stills/t{t:07.2f}.png")
    elif sys.argv[1] == "render":
        t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
        t1 = float(sys.argv[4]) if len(sys.argv) > 4 else DURATION
        render_all(sys.argv[2], t0, t1)
