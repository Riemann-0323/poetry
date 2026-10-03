"""Shared poem text + master timeline for the 《薰衣草》 film (audio and video both read this)."""
import numpy as np

W, H, FPS = 1920, 1080, 30
SR = 48000
DURATION = 219.0
FONT_DIR = "fonts"

TITLE = "薰衣草"
SIGNATURE = "Lacepocterlse"

# (leading spaces, text) exactly as the poem was laid out
STANZAS = [
    [(0, "薰衣草之蓝，甜甜圈的魔法"), (0, "抹茶千层一般的沉默，甜蜜而"),
     (0, "苦涩，你说安睡之神这时"), (0, "会不会也梦到罪恶的我")],
    [(5, "忧郁的骑士轻嗅玫瑰，被否定的"), (6, "无理就已经是现实，不要拒绝"),
     (8, "我的全部，此刻普罗旺斯的风笛"), (11, "悠扬，你知道的，抱着小熊的我")],
    [(10, "就拥有了整片星空，草地上的露水"), (8, "好讨厌，你能不能帮我擦擦脸，用"),
     (6, "手就行啦，我不讲究的，我知道的"), (5, "你最好了，你从不拒绝别人，所以"),
     (4, "来片巧克力曲奇吧，就现在，快点呀"), (5, "我们出发探险吧，天上只有那么多星星")],
    [(7, "但许愿的人越来越多，你想许什么愿望"), (5, "什么? 告诉我就不灵了，好吧好吧"),
     (0, "我都听你的，但我还想告诉你我的心意")],
    [(0, "恩底弥翁的幸运色，奶酪棒的缠绵"), (0, "唯一的钢琴谱飘散在雨中，其实油画"),
     (0, "最重要的是画框，你喜欢木头"), (0, "象牙，璞玉，还是大理石呢")],
    [(8, "一直在一起么，我就知道你会这么说"), (6, "但我要告诉你，在我停止爱你之前"),
     (4, "我会一直爱着这个世界，这个有你的现实"), (3, "欸怎么你衣服脏了，你知不知道呀，薰衣草"),
     (1, "其实是古罗马人洗衣服时用的，哎呀就算知道"), (0, "也听我说完嘛，不会很久的，薰衣草的花语是")],
    [(0, "藤蔓攀登的速度比绝望还快"), (0, "仅仅思考到种子成熟，就已经"),
     (0, "秋天了吗，为什么还看不到月亮"), (0, "为什么河水还在向东，重音符号"),
     (0, "为什么这么长，我为什么在写作")],
    [(2, "小熊变成了水母降落伞，离别的曲子"), (4, "怎么还有点跑调，明明是我最后的歌"),
     (6, "蔚蓝的空气毯子会是柔软的吗，未来"), (8, "我不知道，重力被我遗忘，晚安，晚安"),
     (9, "我相信薰衣草会飞往"), (33, "飞往，或许是遥远的远方")],
]

RAW_SOURCE = "* " + TITLE + "<br/>" + "<br/><br/>".join(
    "<br/>".join(" " * ind + txt for ind, txt in st) for st in STANZAS) + "<br/>" + SIGNATURE

# ---------------------------------------------------------------- sections (seconds)
BAR = 2.5            # 72 bpm, 3/4  -> one bar = 2.5 s
BEAT = BAR / 3
SEC = dict(
    P=(0.0, 10.0), S1=(10.0, 35.0), S2=(35.0, 57.5), S3=(57.5, 87.5), S4=(87.5, 102.5),
    S5=(102.5, 122.5), S6=(122.5, 150.0), HUSH=(150.0, 152.5), S7=(152.5, 176.0),
    S8=(176.0, 206.0), CODA=(206.0, DURATION))

FREEZE = (93.40, 94.20)        # 「什么?」 — time stops
CUT = 150.0                    # 「薰衣草的花语是」 — never answered
RAW = (170.40, 175.40)         # 「我为什么在写作」 — the page shows its source
GRAVITY = 192.6                # 「重力被我遗忘」
DISSOLVE = {0: 33.0, 1: 55.3, 2: 85.3, 3: 100.9, 4: 120.6}
TITLE_DISSOLVE = 7.4

# Each line: start time + either a typing rate (chars/s) or an explicit op list.
LINE_PLAN = [
    [(11.5, 5.5), (16.6, 5.5), (21.6, 5.5), (26.6, 5.0)],
    [(36.5, 5.5), (41.4, 5.5), (46.3, 5.5), (51.0, 5.5)],
    [(58.2, 5.5), (62.6, 6.0), (66.9, 6.5), (71.0, 7.0), (75.0, 8.5), (78.9, 8.5)],
    [(88.2, 6.0), (92.6, [("type", "什么?", 3.2), ("pause", 0.95), ("type", " 告诉我就不灵了，好吧好吧", 7.5)]), (97.0, 6.5)],
    [(103.2, 5.0), (107.9, 5.0), (112.6, 5.0), (116.6, 4.0)],
    [(123.2, 5.5), (127.6, 5.0),
     (131.8, [("type", "我会一直爱着你", 5.0), ("pause", 0.75), ("back", 1, 0.0), ("pause", 0.55),
              ("type", "这个世界，这个有你的现实", 5.5)]),
     (137.0, 6.0), (141.0, 6.5),
     (144.9, [("type", "也听我说完嘛，不会很久的，薰衣草的花语是", 7.5), ("pause", 0.5),
              ("type", "等待", 5.0), ("pause", 0.3), ("back", 2, 0.2)])],
    [(153.0, 6.5), (156.8, 6.5), (160.4, 6.5), (164.4, 6.5), (168.0, 6.5)],
    [(177.0, 5.0), (181.4, 5.0), (185.8, 5.0), (190.2, 5.0), (195.2, 4.0), (198.4, 3.5)],
]

PAUSE_AFTER = {"，": 0.24, "?": 0.30, " ": 0.06}


def build_ops(si, li):
    """Typing events for one line: list of (t, 'type'|'back', char)."""
    t0, plan = LINE_PLAN[si][li]
    text = STANZAS[si][li][1]
    rng = np.random.default_rng(1000 + si * 37 + li)
    if not isinstance(plan, list):
        plan = [("type", text, plan)]
    ops, t = [], t0
    for seg in plan:
        if seg[0] == "type":
            _, s, rate = seg
            for ch in s:
                ops.append((t, "type", ch))
                t += (1.0 / rate) * (0.65 + 0.7 * rng.random()) + PAUSE_AFTER.get(ch, 0.0)
        elif seg[0] == "pause":
            t += seg[1]
        elif seg[0] == "back":
            _, n, iv = seg
            for _ in range(n):
                ops.append((t, "back", ""))
                t += iv
    return ops


OPS = [[build_ops(si, li) for li in range(len(st))] for si, st in enumerate(STANZAS)]


def line_state(si, li, t):
    """Characters currently on the line: list of (char, typed_time)."""
    buf = []
    for (te, kind, ch) in OPS[si][li]:
        if te > t:
            break
        if kind == "type":
            buf.append((ch, te))
        elif buf:
            buf.pop()
    return buf


def line_end(si, li):
    return OPS[si][li][-1][0]


def char_time(si, li, idx):
    """Time the idx-th char of the *final* line text is typed."""
    buf = []
    for (te, kind, ch) in OPS[si][li]:
        if kind == "type":
            buf.append(te)
        else:
            buf.pop()
    return buf[idx]


def find_time(si, li, sub, occurrence=0):
    text = STANZAS[si][li][1]
    k = -1
    for _ in range(occurrence + 1):
        k = text.index(sub, k + 1)
    return char_time(si, li, k)


if __name__ == "__main__":
    for si, st in enumerate(STANZAS):
        for li, (ind, txt) in enumerate(st):
            ops = OPS[si][li]
            print(f"S{si+1}.{li+1}  {ops[0][0]:7.2f} -> {ops[-1][0]:7.2f}  {txt}")
    print(RAW_SOURCE)
