"""《红色》 — poem, beat grid and arrangement shared by audio.py and video.py."""
import math

W, H, FPS, SR = 1920, 1080, 30, 48000
FONT_DIR = "fonts"
BPM = 174.0
BEAT = 60.0 / BPM            # 0.3448 s
BAR = 4 * BEAT               # 1.3793 s
STEP = BEAT / 4              # one 16th


def bar(b):
    return b * BAR


TITLE = "红色"
SIGNATURE = "Apokal-Panra"

STANZAS = [
    ["红色，红色在你魔鬼的腰畔", "随着金柳下的枝桠荡出。这些", "谄媚的侏儒，小兽般爬行的蜘蛛",
     "厌弃之高台上有更多的痛苦在渴求", "如朗基努斯之残片，剐蹭玻璃又", "惩罚雷霆，熔岩蛋糕一样甜蜜的"],
    ["你笑容渗透下裙摆的弧线，一颗", "国际的鸽血尖晶石，为什么不镶嵌在", "膝盖顶部，这样再也没人被允许",
     "下跪，神不是神，手帕上记录了", "小道消息和一整夜的聊天记录，什么是"],
    ["眉目传情，语言魔法的巴别塔大学博士生", "毕业论文写了三千页的你，可他并不", "如我吧，至少我有肉体，所以我想你了",
     "让我们先拥抱，再做一些只有", "中文现代诗不允许的举措，我要纯洁的", "语言，美好如青柳的修辞，繁星春水",
     "在杯中笑出的酒花，这一局我要全下"],
    ["积分制的弊端不过如此了，你又能堵上什么", "这些价值怎么定义，你送我的第一朵蔷薇", "第一滴泪，第一次拥吻，第一次心碎",
     "情感和劳动怎么等价交换，阿努比斯天平", "下的随意平均数，River clash。"],
    ["水崩山夜，棕榈树如分岔路口般的", "手指深入我们的间隙，阳光将破碎在", "今天晚上，预告的年轮精密如静脉",
     "你深海色的火焰城堡需要一次洗礼"],
    ["或许是用世上全部的泪水? 这痛苦如", "口腔溃疡之溪，取经又静默而心中", "诵读出每种超越性观点，你真的得到了",
     "幸福，焚烧了所有玫瑰诗句，乐未央", "紧接着提取出爱意，某种豆类饮料", "林鸟也在此跪坐，高枝陷落",
     "燕辞水洛，砸碎的弗洛伊德蔷薇", "反射我的懦弱，烟雾缭绕的娼妓", "别她莫辩之错，天真但从不感伤的",
     "画师，不成比例的魔法，爆裂音游", "终极的天堂之火，无限的弃绝骑士"],
    ["这时全部的红色向你袭来，你终于", "得到了幸福，你不想死，也不想去巴黎", "不用天堂，不用我，我祝你永远失去，永远活着"],
]

# words printed in red inside each line, and the giant poster word behind it
KEYWORDS = [
    ["红色", "金柳", "蜘蛛", "痛苦", "朗基努斯", "雷霆"],
    ["裙摆", "鸽血尖晶石", "膝盖", "神", "聊天记录"],
    ["巴别塔", "三千页", "肉体", "拥抱", "不允许", "繁星春水", "全下"],
    ["积分", "价值", "第一", "等价交换", "River clash"],
    ["水崩山夜", "破碎", "年轮", "洗礼"],
    ["泪水", "溃疡", "超越", "幸福", "豆类饮料", "陷落", "蔷薇", "懦弱", "天真", "爆裂音游", "弃绝骑士"],
    ["红色", "幸福", "永远活着"],
]
POSTER = [
    ["红", "柳", "蛛", "痛", "LONGINUS", "雷"],
    ["弧", "SPINEL", "跪", "神", "00:00"],
    ["BABEL", "3000", "肉", "抱", "禁", "春", "ALL IN"],
    ["积分", "¥?", "1ST", "W=c+v+m", "RIVER"],
    ["崩", "碎", "轮", "洗"],
    ["泪", "溃", "超", "幸", "C8H10N4O2", "陷", "碎", "懦", "天真", "爆裂", "∞"],
    ["红", "福", "活"],
]

# first bar of each line
LINE_BAR = [
    [8, 12, 16, 20, 24, 28],
    [32, 36, 40, 44, 48],
    [52, 56, 60, 64, 68, 72, 76],
    [80, 84, 88, 92, 96],
    [100, 104, 108, 112],
    [116 + 4 * i for i in range(11)],
    [160, 164, 170],
]
SECTION_BAR = dict(INTRO=0, S1=8, S2=32, S3=52, S4=80, S5=100, S6=116, S7=160, CODA=180, END=190)
DURATION = bar(SECTION_BAR["END"])           # 262.07 s
TITLE_BAR = 6
RIVER_BAR = 98                               # 「River clash」 impact
MISS_BAR = 158                               # combo breaks
REDWAVE_BAR = 160
PURE_BARS = (72, 76)                         # 繁星春水 — the only clean, un-red moment


def chunks(text):
    out, cur = [], ""
    for ch in text:
        cur += ch
        if ch in "，。?":
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def _schedule():
    """time each character appears; chunks start on beats, characters stream in 16ths"""
    T = []
    for si, st in enumerate(STANZAS):
        rows = []
        for li, text in enumerate(st):
            t0 = bar(LINE_BAR[si][li])
            per = STEP * (1.0 if si != 6 else 1.6)
            times = []
            if si == 6 and li == 2:
                starts = [bar(170), bar(172), bar(174), bar(176)]
                for ck, s in zip(chunks(text), starts):
                    for j, _ in enumerate(ck):
                        times.append(s + j * STEP * 2.2)
                rows.append(times)
                continue
            t = t0
            for ck in chunks(text):
                for j, _ in enumerate(ck):
                    times.append(t + j * per)
                end = t + len(ck) * per
                t = math.ceil((end + BEAT * 0.5 - t0) / BEAT - 1e-6) * BEAT + t0
            rows.append(times)
        T.append(rows)
    return T


CHAR_T = _schedule()


def word_time(si, li, word):
    k = STANZAS[si][li].find(word)
    return CHAR_T[si][li][max(k, 0)]


# ------------------------------------------------------------------ arrangement (bars -> groove)
def groove(b):
    if b < 16 or 72 <= b < 76 or 98 <= b < 100 or 158 <= b < 180 or b >= 180:
        return "none"
    if 76 <= b < 80:
        return "build"
    if 80 <= b < 98 or 120 <= b < 158:
        return "dnb"
    if 116 <= b < 120:
        return "build2"
    if 100 <= b < 116:
        return "sparse"
    return "half"


PATTERNS = {
    "half": dict(k=[0, 10], s=[8], h=list(range(0, 16, 2))),
    "sparse": dict(k=[0, 11], s=[8], h=[2, 6, 10, 14]),
    "dnb": dict(k=[0, 10], s=[4, 12], h=list(range(0, 16, 2)), g=[7, 15]),
    "build": dict(k=[0, 8], s=[], h=list(range(0, 16, 2))),
    "build2": dict(k=[0, 8], s=[], h=list(range(0, 16, 2))),
    "none": dict(k=[], s=[], h=[]),
}


def drum_events():
    """(time, kind, velocity) for every drum hit in the piece"""
    ev = []
    for b in range(SECTION_BAR["END"]):
        g = groove(b)
        p = PATTERNS[g]
        t0 = bar(b)
        for s in p["k"]:
            ev.append((t0 + s * STEP, "k", 1.0))
        for s in p["s"]:
            ev.append((t0 + s * STEP, "s", 1.0))
        for s in p["h"]:
            ev.append((t0 + s * STEP, "h", 0.6 if s % 4 else 0.9))
        for s in p.get("g", []):
            ev.append((t0 + s * STEP, "s", 0.25))
        if g in ("build", "build2"):
            k = (b - (76 if g == "build" else 116))
            n = [4, 8, 8, 16][min(k, 3)]
            for i in range(n):
                ev.append((t0 + i * BAR / n, "s", 0.35 + 0.6 * (i / n) * (k + 1) / 4))
        if g == "dnb" and b % 8 == 7:
            for i in range(8):
                ev.append((t0 + (8 + i) * STEP, "s", 0.5 + 0.06 * i))
    return sorted(ev)


DRUMS = drum_events()
KICKS = [t for t, k, v in DRUMS if k == "k"]
SNARES = [t for t, k, v in DRUMS if k == "s" and v > 0.9]

# ------------------------------------------------------------------ rhythm-game chart (S6) — shared with the lead synth
# a 2-bar lead riff in D phrygian (16ths), MIDI; None = rest
RIFF = [74, None, 77, 74, 81, None, 79, 77, 74, None, 75, 77, 79, None, 82, 81,
        74, None, 77, 74, 81, None, 84, 82, 81, None, 79, 77, 75, 77, 74, None]


def chart():
    notes = []
    for b in range(120, 158):
        half = (b - 120) % 2
        dens = 0.55 if b < 128 else (0.8 if b < 144 else 1.0)
        for i in range(16):
            m = RIFF[half * 16 + i]
            if m is None:
                continue
            if (i % 2 == 1) and ((b * 7 + i * 13) % 100) / 100 > dens:
                continue
            notes.append((bar(b) + i * STEP, m, (m * 5 + i) % 4))
    return notes


CHART = chart()
