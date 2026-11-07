# -*- coding: utf-8 -*-
"""
个性化笔迹签名采集器 (PySide6) · 新口径: 只选「没见过」的 + 「难」的
=====================================================================
硬规则: user_data 里已经有的字, 一个都不再抽。
        (旧版会优先补 1~3 张的老字、也会从常用字池随机抽到写过的字, 已取消)

「没见过」= 启动时 user_data 里 0 张的 charset 字符(当前 3759 类中约 3481 个)。
「难」    = 按用户给的画像「带偏旁、笔画较多」做客观判定, 全部指标来自楷体
            字形本身, 不编造字义、不靠主观字表:
  S1 笔画较多   墨迹像素量 >= 全字符表 p70(自适应分位数, 当前 1702 px)
  S2 带偏旁     连通部件数 >= 4(偏旁与主体分离) 或 4x4 网格铺满 16 格
                或命中按可见偏旁整理的 FAMILIES 组
  S1+S2 组合    用户举的典型难字, 额外加权, 永远排在最前
  S3 难例同族   历史难例(hard_history / paradigm_gap / 优化报告)所在偏旁组
                里的未见字。难例字本身若在库里则跳过, 只把难度扩散给新字
  S4 模型证据   扩类二级字(索引 >= 3755) / HWDB 通用域按类召回最差的字
  S5 稀疏易混   笔画量位于全表后 15%, 辅助信号(单独不算难字)

三种口径(下拉切换, 三者都执行上面的硬规则):
  未见 · 难字优先   全部未见字合格, 按难度分降序, 带偏旁+笔画多的字排最前(默认)
  未见 · 仅难字     未见字 ∩ 重点难字(S1+S2 组合 / S3 / S4), 只出难字
  未见 · 难例同族   未见字 ∩ S3, 最聚焦

断点续采: 上次没采满 4 张的那个字会被保留并优先采完(它属于「已经在采的字」,
          不算重新抽老字), 之后新抽的字一律遵循硬规则。

4 张的意义: 覆盖你自己的书写方差(端正 / 自然 / 连笔倾斜 / 不同大小),
微调后形成该字的个性化签名; 单张只会让模型记住一个偶然写法。

轨迹旁路记录(本版新增):
  位图照旧存 hanzi/user_data/<字>_<ms>.png; 同时把鼠标轨迹写到
  hanzi/traj_data/<字>_<YYYYMMDD_HHMMSS>.npz —— x/y/t/pen/stroke_idx,
  同目录再写一个同名 .json 侧车(device/source/writer_id/点数/时长)。
  格式见 hanzi/traj/TRAJ_FORMAT.md; 把 TRAJ_ON 改成 False 即退回纯位图采集。

  注意: 左下角显示的是「区域序」而不是笔画数。连笔会让 pen 段数与真实笔画脱钩,
  笔画数没有结构参考价值; 真正稳定的是笔尖在 4x4 网格里停留的先后顺序
  (hanzi/traj/region_order.py)。stroke_idx 仍然逐点记录, 只作数据质检用。

用法: D:/Anaconda/envs/pytorch/python.exe 个性化签名采集.py
采集完运行: python 一键优化训练.py            (位图 -> 个性化微调)
轨迹攒够后: python hanzi/traj/train.py --real hanzi/traj_data
"""
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter

import cv2
import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import (QColor, QFont, QImage, QKeySequence, QPainter, QPen,
                           QPixmap, QShortcut)
from PySide6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QLabel,
                               QMainWindow, QPushButton, QTextEdit, QVBoxLayout,
                               QWidget)

ROOT = os.path.dirname(os.path.abspath(__file__))
HANZI = os.path.join(ROOT, 'hanzi')
USER_DIR = os.path.join(HANZI, 'user_data')
STATE_FILE = os.path.join(HANZI, 'sign_progress.json')
META_FILE = os.path.join(HANZI, 'sign_difficulty.json')
CANVAS_SIZE = 400
PEN_WIDTH = 18
NEED = 4

POOL_KEYS = ['unseen_hard_first', 'unseen_hard_only', 'unseen_family']
POOL_LABELS = ['未见 · 难字优先（推荐）', '未见 · 仅难字', '未见 · 难例同族']
POOL_DESC = {
    'unseen_hard_first': '未见字全集，按难度分降序；带偏旁、笔画多的字排最前',
    'unseen_hard_only': '未见字 ∩ 重点难字（带偏旁+笔画较多 / 难例同族 / 模型实证）',
    'unseen_family': '未见字 ∩ 历史难例所在偏旁家族（最聚焦）',
}
LEGACY_POOL = {'common': 'unseen_hard_first',
               'full': 'unseen_hard_first',
               'ext': 'unseen_family'}

# 按「可见偏旁 / 字形共性」整理, 用于把历史难例的难度扩散到未见字。
# 不做字义解释, 只是结构分组; NON_RADICAL 组不参与「带偏旁」判定。
FAMILIES = {
    '穴字头': '穴空窍究窃窗帘穿突窄窝窘窈窕',
    '隹 / 且旁': '雎雀焦集售准谁难雁雕狙疽阻组租',
    '鸟旁': '鸥鸦鸭鹅鹊鹃鹤鸡鸣鸠鸩',
    '厂字头': '严产芦户声厅厉压厦厨',
    '辶旁': '这运进迸遂迹过达递迷遭遥逢避邀',
    '氵旁': '清泌滚漫溃演消没设灌溉潭溪渊溯',
    '艹头': '芦茄苗花茶草荒葱藕菠莓葡萍萤萧蔬藏蔓荔苔藻菌蕊蕨藓',
    '纟旁': '绒红细织绞给结绕绳线绸缎绷绿纲纺纷纸纹纯绵缉缮',
    '木旁': '根林村枝柱柜桃梅棉椅桂梢栈桥梁梓棱棵棚棕橡榴槛',
    '攵旁': '政改攻放故效救教敏败敌收散敦敷整',
    '宀头': '宝字家室客安完定宜实务寂寞寡寓宙审宿寨',
    '冖头': '写军冤冠冥幂',
    '力旁': '劲功动助勇勤势勉勘勒勋',
    '歹 / 死': '死残歼殊殉殖殡',
    '尸 / 户': '户肩扁扇启屉届屋居层屏屠',
    '讠旁': '识证语读话讲议评诉诚诊译诵谱谢谭',
    '钅旁': '铜铁钢铝锡铅锌镍镜铃锣错镇锐镀铸',
    '忄 / 心': '情怕恨悔恼懒慌惜惊惧惭愧忱悸悼',
    '扌旁': '打把推拉拖提捧抱扔抛摔擦描捏拣搔',
    '竹头': '笔笛筐篮箩箕筷筒簿籍笨笋簇簌',
    '疒旁': '疾病疼痛痒疲瘦瘪瘫瘾痴痰',
    '少笔画易混': '己已巳末未拨拔兔免戊戌戍哀衷',
}
NON_RADICAL = {'少笔画易混'}

STYLE_HINTS = [
    '第 1 张：端正慢写，笔画写完整',
    '第 2 张：自然速度，平时怎么写就怎么写',
    '第 3 张：稍快 / 带点连笔或倾斜',
    '第 4 张：换个大小或在画布里换个位置',
]


def _load_json(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return dict() if default is None else default


def _base_n():
    try:
        with open(os.path.join(HANZI, 'ext_chars.json'), encoding='utf-8') as f:
            return int(json.load(f)['base_classes'])
    except Exception:
        return 3755


BASE_N = _base_n()


def load_chars():
    with open(os.path.join(HANZI, 'charset.json'), encoding='utf-8') as f:
        return json.load(f)['chars']


def counts_of_user_data():
    c = Counter()
    for fn in os.listdir(USER_DIR) if os.path.isdir(USER_DIR) else []:
        m = re.match(r'(.+)_\d+\.png', fn)
        if m and not m.group(1).startswith('_'):
            c[m.group(1)] += 1
    return c


# ---------------------------------------------------------------- 字形难度索引
_FONT = None
_GLYPH = {}


def _get_font():
    global _FONT
    if _FONT is None:
        try:
            from PIL import ImageFont
            fp = r'C:\Windows\Fonts\simkai.ttf'
            _FONT = ImageFont.truetype(fp, 90) if os.path.exists(fp) else False
        except Exception:
            _FONT = False
    return _FONT or None


def glyph_stats(ch, font):
    """单个字的客观字形指标: 墨迹像素量 / 连通部件数 / 4x4 网格覆盖 / 密度"""
    if font is None:
        return None
    try:
        from PIL import Image, ImageDraw
        im = Image.new('L', (128, 128), 255)
        d = ImageDraw.Draw(im)
        bb = d.textbbox((0, 0), ch, font=font)
        d.text(((128 - (bb[2] - bb[0])) / 2 - bb[0],
                (128 - (bb[3] - bb[1])) / 2 - bb[1]), ch, font=font, fill=0)
        ink = ((255 - np.asarray(im, np.uint8)) > 40).astype(np.uint8)
        ys, xs = np.where(ink)
        if len(xs) == 0:
            return None
        c = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        _, _, st, _ = cv2.connectedComponentsWithStats(c, 8)
        parts = int((st[1:, cv2.CC_STAT_AREA] >= 12).sum())
        h, w = c.shape
        cells = 0
        for i in range(4):
            for j in range(4):
                sub = c[i * h // 4:(i + 1) * h // 4, j * w // 4:(j + 1) * w // 4]
                if sub.mean() > 0.04:
                    cells += 1
        return {'px': int(c.sum()), 'parts': parts, 'cells': cells,
                'dens': round(float(c.mean()), 4)}
    except Exception:
        return None


def glyph_index(chars):
    """一次性算全表字形指标并落盘缓存; 字符表变化(扩类)时自动重算。约 1.5s。"""
    sig = hashlib.md5(''.join(chars).encode('utf-8')).hexdigest()
    cache = _load_json(META_FILE)
    if (cache.get('charset_sig') == sig
            and len(cache.get('glyphs') or {}) >= len(chars)):
        glyphs, pcts = cache['glyphs'], cache['pcts']
    else:
        font = _get_font()
        glyphs = {}
        for ch in chars:
            glyphs[ch] = glyph_stats(ch, font) or {'px': 0, 'parts': 0,
                                                    'cells': 0, 'dens': None}
        px = np.array([glyphs[c]['px'] for c in chars], dtype=float)
        order = np.argsort(px)
        rank = {}
        for r, i in enumerate(order):
            rank[chars[i]] = round(100.0 * r / max(1, len(chars) - 1), 1)
        pcts = {'px_p70': int(np.percentile(px, 70)),
                'px_p15': int(np.percentile(px, 15)),
                'px_rank': rank,
                'note': 'px=楷体90号渲染的墨迹像素量(笔画多少的客观代理); '
                        'parts=面积>=12px的连通部件数(偏旁与主体分离)'}
        with open(META_FILE, 'w', encoding='utf-8') as f:
            json.dump({'charset_sig': sig, 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                       'glyphs': glyphs, 'pcts': pcts}, f, ensure_ascii=False)
    _GLYPH.update(glyphs)
    return glyphs, pcts


def ink_density(ch):
    """保留旧接口: 笔画密度(供界面「模型视角」一行使用)"""
    g = _GLYPH.get(ch)
    if g is not None:
        return g.get('dens')
    s = glyph_stats(ch, _get_font())
    return s['dens'] if s else None


def fam_index():
    fam = {}
    for name, s in FAMILIES.items():
        for ch in dict.fromkeys(s):
            fam.setdefault(ch, []).append(name)
    return fam


def _coerce_ch(v, chars):
    """优化报告里既有字符也有类别索引, 统一转成字符"""
    if isinstance(v, str) and len(v) == 1 and not v.isdigit():
        return v
    try:
        i = int(v)
        return chars[i] if 0 <= i < len(chars) else ''
    except Exception:
        return ''


def hard_seeds(chars):
    """汇总客观「难」证据 -> ({字: 证据列表}, 通用域最差类集合)。仅作难度种子。"""
    seeds, worst = {}, set()

    def add(ch, why):
        if not ch or len(ch) != 1:
            return
        cur = seeds.setdefault(ch, [])
        if why not in cur:
            cur.append(why)

    for ch in (_load_json(os.path.join(HANZI, 'hard_history.json'),
                          {'tags': {}}).get('tags') or {}):
        add(ch, '累积难例记忆')
    gap = _load_json(os.path.join(HANZI, 'paradigm_gap.json'))
    for ch in (gap.get('weights') or {}):
        add(ch, '范式对照加权')
    for key, label in [('blind_tags', '深度盲区'), ('hard_tags', '共同难例'),
                       ('low_conf_tags', '低置信'), ('ext_tags', '扩类新字'),
                       ('history_tags', '历史难例')]:
        for ch in (gap.get(key) or []):
            add(ch, label)
    rep = _load_json(os.path.join(HANZI, 'optimization_report.json'))
    for v in (rep.get('bad_tags') or []):
        add(_coerce_ch(v, chars), '门槛未过标签')
    dom = rep.get('domains') or {}
    for key, label in [('fit', '拟合域'), ('val', '留出域')]:
        d = dom.get(key) or {}
        for k in ('wrong', 'low'):
            for v in (d.get(k) or []):
                add(_coerce_ch(v, chars), label + ('错例' if k == 'wrong' else '低置信'))
    for key in ('hw', 'hw_full'):
        for row in ((dom.get(key) or {}).get('worst') or []):
            if isinstance(row, (list, tuple)) and row:
                ch = _coerce_ch(row[0], chars)
                add(ch, '通用域最差类')
                if ch:
                    worst.add(ch)
    return seeds, worst


def difficulty_index(chars, unseen, seeds, worst, fam, glyphs, pcts):
    """给每个未见字算难度分 + 可解释理由"""
    seed_fams = set()
    for ch in seeds:
        seed_fams.update(fam.get(ch, []))
    fam_seeds = {}
    for name, s in FAMILIES.items():
        hit = [c for c in dict.fromkeys(s) if c in seeds]
        if hit:
            fam_seeds[name] = hit
    idx_of = {c: i for i, c in enumerate(chars)}
    p70, p15 = pcts['px_p70'], pcts['px_p15']
    rank = pcts['px_rank']
    meta = {}
    for ch in unseen:
        g = glyphs.get(ch) or {}
        px, parts, cells = g.get('px', 0), g.get('parts', 0), g.get('cells', 0)
        score, reasons, sig = 0.0, [], []
        rad = [n for n in fam.get(ch, []) if n not in NON_RADICAL]
        # 组内必须真的有「别的」难例字才算同族扩散, 避免种子就是自己时自指加分
        fam_hit = [n for n in fam.get(ch, [])
                   if n in seed_fams
                   and any(x != ch for x in fam_seeds.get(n, []))]

        s1 = px >= p70
        s2 = parts >= 4 or cells >= 16 or bool(rad)
        if s1:
            score += 3.0
            sig.append('S1')
            reasons.append(f'笔画较多：墨迹 {px} px，位于全表前 '
                           f'{100 - rank.get(ch, 50):.0f}%（阈值 {p70} px）')
        if s2:
            score += 3.0
            sig.append('S2')
            bits = []
            if parts >= 4:
                bits.append(f'连通部件 {parts} 块（偏旁与主体分离）')
            if cells >= 16:
                bits.append('4×4 网格铺满 16 格（部件分布满）')
            if rad:
                bits.append('命中偏旁组「' + '、'.join(rad) + '」')
            reasons.append('带偏旁 / 合体结构：' + '；'.join(bits))
        if s1 and s2:
            score += 2.0
            sig.append('S1+S2')
            reasons.append('组合命中：带偏旁 + 笔画较多（指定的典型难字，优先采集）')
        if fam_hit:
            score += 2.5
            sig.append('S3')
            ex = []
            for n in fam_hit[:2]:
                others = [x for x in fam_seeds.get(n, []) if x != ch][:3]
                ex.append(f'{n}（{"、".join(others)}）' if others else n)
            reasons.append('难例同族：同组出现过模型难例 —— ' + '；'.join(ex))
        is_ext = idx_of.get(ch, -1) >= BASE_N
        if is_ext or ch in worst:
            score += 2.0
            sig.append('S4')
            bits = []
            if is_ext:
                bits.append(f'扩类二级字（索引 ≥ {BASE_N}，原模型无输出节点）')
            if ch in worst:
                bits.append('HWDB 通用域按类召回最差')
            reasons.append('模型证据：' + '；'.join(bits))
        if px and px <= p15:
            score += 0.5
            sig.append('S5')
            reasons.append(f'笔画稀疏（墨迹 {px} px），易与形近字撞')
        # 连续项: 同档内再按客观复杂度排序, 避免大量字同分
        score += 1.5 * min(px / 2200.0, 1.0) + 1.0 * min(parts / 8.0, 1.0)
        meta[ch] = {'score': round(score, 2), 'reasons': reasons,
                    'signals': sig,
                    'is_hard': bool({'S1', 'S2', 'S3', 'S4'} & set(sig)),
                    # 重点难字: 明确「带偏旁+笔画多」的组合, 或有模型实证的字。
                    # 只沾到单个弱信号的字不算, 避免「仅难字」口径被稀释。
                    'is_focus': bool({'S1+S2', 'S3', 'S4'} & set(sig))}
    return meta, seed_fams, fam_seeds


# ------------------------------------------------------------------ 轨迹
TRAJ_DIR = os.path.join(HANZI, 'traj_data')
TRAJ_ON = True            # False 则退回纯位图采集, 一个 npz 都不写
REC_MIN_DT_MS = 2         # 同位置最小间隔, 低于此值丢点(去抖/去重)

# 区域停留序列：结构与抬不抬笔无关，连笔不会破坏它（见 hanzi/traj/region_order.py）
try:
    if os.path.join(HANZI, 'traj') not in sys.path:
        sys.path.insert(0, os.path.join(HANZI, 'traj'))
    import region_order as _region
except Exception:                                   # 库缺失不影响采集
    _region = None


class TrajRecorder:
    """旁路记录鼠标轨迹: x, y, t(ms), pen(0=抬笔/1=落笔), stroke_idx。

    完全独立于位图: 墨迹仍由 DrawCanvas 的 QPixmap 画, 这里只按事件采点。
    坐标=画布像素(左上原点, y 向下), 与 hanzi/traj/TRAJ_FORMAT.md 一致。
    stroke_idx 每次 pen 由 0->1 自增, 保存后可与标准笔顺逐笔比对。
    """

    def __init__(self, canvas_size=CANVAS_SIZE):
        self.cw = self.ch = int(canvas_size)
        self.reset()

    def reset(self):
        self.x, self.y, self.t, self.pen, self.si = [], [], [], [], []
        self._t0 = None
        self._last_t = -1
        self._last_xy = None
        self._stroke = -1
        self._down = False
        self._i0 = 0          # 本笔第一个点的下标

    # ---------- 时间 ----------
    def _now(self):
        if self._t0 is None:
            self._t0 = time.perf_counter()
        return int((time.perf_counter() - self._t0) * 1000.0)

    def _push(self, x, y, pen, sidx):
        t = self._now()
        if t <= self._last_t:          # t 必须严格单调递增
            t = self._last_t + 1
        self.x.append(float(x))
        self.y.append(float(y))
        self.t.append(t)
        self.pen.append(int(pen))
        self.si.append(int(sidx))
        self._last_t = t
        self._last_xy = (x, y)

    # ---------- 事件 ----------
    def down(self, x, y):
        if self._down:                 # 上一笔漏了 release, 先补抬笔
            self._close()
        self._stroke += 1
        self._down = True
        self._i0 = len(self.x)
        self._last_xy = None
        self._push(x, y, 1, self._stroke)

    def move(self, x, y):
        if not self._down:
            return
        if self._last_xy == (x, y) and (self._now() - self._last_t) < REC_MIN_DT_MS:
            return
        self._push(x, y, 1, self._stroke)

    def up(self, x, y):
        if not self._down:
            return
        if (x, y) != self._last_xy:
            self._push(x, y, 1, self._stroke)
        self._close()

    def _close(self):
        """收笔: 补一个 pen=0 的抬笔点(与末点同坐标, stroke_idx=-1)。

        抬笔点不归属任何笔画, 与 轨迹采集.py / dataio.stroke_run_ids 的语义一致;
        pen==1 的连续段数 == 笔画数, 这是唯一口径。
        """
        if len(self.x) - self._i0 < 2:
            # 只点了一下: 原地补点, 保证每笔 >= 2 点
            self._push(self.x[-1], self.y[-1], 1, self._stroke)
        t = self._now()
        if t <= self._last_t:
            t = self._last_t + 1
        self.x.append(float(self.x[-1]))
        self.y.append(float(self.y[-1]))
        self.t.append(t)
        self.pen.append(0)
        self.si.append(-1)
        self._last_t = t
        self._down = False

    # ---------- 输出 ----------
    def stats(self):
        return len(self.x), (self._stroke + 1 if self._stroke >= 0 else 0), (
            self.t[-1] if self.t else 0)

    def arrays(self):
        return {
            'x': np.asarray(self.x, np.float32),
            'y': np.asarray(self.y, np.float32),
            't': np.asarray(self.t, np.int64),
            'pen': np.asarray(self.pen, np.uint8),
            'stroke_idx': np.asarray(self.si, np.int32),
            'canvas_w': np.int32(self.cw),
            'canvas_h': np.int32(self.ch),
        }

    def save_npz(self, path, char, writer_id='u0', device='mouse'):
        n, ns, dur = self.stats()
        if n < 4 or ns < 1:
            return None
        arr = self.arrays()
        arr['char'] = np.array(char)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:        # 中文路径必须传文件对象
            np.savez_compressed(f, **arr)
        meta = {
            'char': char, 'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': device, 'source': 'real', 'writer_id': writer_id,
            'canvas_w': self.cw, 'canvas_h': self.ch,
            'n_points': n, 'n_strokes': ns, 'duration_ms': int(dur),
            'app': '个性化签名采集.py v1.1', 'format': 'traj-v1',
        }
        with open(os.path.splitext(path)[0] + '.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        return path


# ------------------------------------------------------------------ 画板
class DrawCanvas(QWidget):
    """鼠标书写画布, 导出白底黑字灰度图; 同时旁路记录轨迹"""
    def __init__(self, size=CANVAS_SIZE):
        super().__init__()
        self.setFixedSize(size, size)
        self.setStyleSheet('border:1px solid #999; background:white;')
        self._last = None
        self.rec = TrajRecorder(size)
        self.on_rec = None           # 回调: (点数, 笔数, 时长ms)
        self.clear()

    def _notify(self):
        if self.on_rec:
            self.on_rec(*self.rec.stats())

    def clear(self):
        self._pixmap = QPixmap(self.size())
        self._pixmap.fill(QColor('white'))
        self._last = None
        self.rec.reset()
        self._notify()
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            p = e.position().toPoint()
            self._last = p
            self.rec.down(p.x(), p.y())
            self._notify()

    def mouseMoveEvent(self, e):
        if self._last is None:
            return
        p = e.position().toPoint()
        self.rec.move(p.x(), p.y())
        painter = QPainter(self._pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor('black'), PEN_WIDTH)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(self._last, p)
        painter.end()
        self._last = p
        self.update()
        self._notify()

    def mouseReleaseEvent(self, e):
        if self._last is not None:
            p = e.position().toPoint()
            self.rec.up(p.x(), p.y())
            self._notify()
        self._last = None

    def paintEvent(self, _):
        QPainter(self).drawPixmap(0, 0, self._pixmap)

    def to_gray(self):
        img = self._pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
        w, h = img.width(), img.height()
        return np.frombuffer(img.constBits(), np.uint8, count=w * h).reshape(h, w).copy()


def preprocess64(canvas_img):
    """与识别画板完全一致: 反色 -> 外接框 -> 最长边62 -> 64居中"""
    a = 255 - canvas_img
    ys, xs = np.where(a > 40)
    if len(xs) == 0:
        return None
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    s = 62.0 / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((64, 64), np.uint8)
    ox, oy = (64 - nw) // 2, (64 - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = a
    return out


# ------------------------------------------------------------------ 主窗口
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('个性化笔迹签名采集 · 只选没见过的与难的')
        self.chars = load_chars()
        self.pool = POOL_KEYS[0]
        self.queue = []
        self.current = ''
        self.saved = 0
        self.saved_files = []
        self.saved_trajs = []
        self._rec_t = 0.0
        self.thumbs = []
        self.skipped = set()
        self.exempt = set()
        self._load_state()

        # 一次性构建字形难度索引(带缓存, 约 1.5s)
        self.glyphs, self.pcts = glyph_index(self.chars)
        self.fam = fam_index()
        self.seeds, self.worst = hard_seeds(self.chars)

        self.canvas = DrawCanvas()
        self.canvas.on_rec = self._on_rec
        self.traj_lab = QLabel('轨迹：0 点 / 0.0s')
        self.traj_lab.setFont(QFont('Consolas', 9))
        self.traj_lab.setStyleSheet('color:#666;')
        self.traj_lab.setWordWrap(True)
        self.traj_lab.setMinimumHeight(34)
        self.target_lab = QLabel('–')
        self.target_lab.setFont(QFont('SimSun', 96))
        self.target_lab.setAlignment(Qt.AlignCenter)
        self.target_lab.setFixedHeight(130)

        self.explain = QTextEdit()
        self.explain.setReadOnly(True)
        self.explain.setFont(QFont('Microsoft YaHei', 10))
        self.explain.setMinimumHeight(230)
        self.explain.setPlainText('正在构建字形难度索引（首次约 2 秒）…')
        QtWidgets.QApplication.processEvents()

        self.style_lab = QLabel('')
        self.style_lab.setFont(QFont('Microsoft YaHei', 11))
        self.style_lab.setStyleSheet('color:#185FA5;')

        self.progress_lab = QLabel('')
        self.progress_lab.setFont(QFont('Microsoft YaHei', 11, QFont.Bold))
        self.stat_lab = QLabel('')
        self.stat_lab.setFont(QFont('Microsoft YaHei', 10))
        self.stat_lab.setStyleSheet('color:#555;')
        self.done_lab = QLabel('')
        self.done_lab.setFont(QFont('Microsoft YaHei', 10))
        self.done_lab.setStyleSheet('color:#1D9E75;')

        self.pool_box = QComboBox()
        self.pool_box.addItems(POOL_LABELS)
        self.pool_box.blockSignals(True)
        self.pool_box.setCurrentIndex(POOL_KEYS.index(self.pool)
                                      if self.pool in POOL_KEYS else 0)
        self.pool_box.blockSignals(False)
        self.pool_box.currentIndexChanged.connect(self._on_pool_change)

        btn_save = QPushButton('保存这张  (Enter)')
        btn_clear = QPushButton('清空画布  (C)')
        btn_undo = QPushButton('撤销上一张')
        btn_skip = QPushButton('换一个字  (S)')
        btn_reshuffle = QPushButton('重排队列')
        for b in (btn_save, btn_clear, btn_undo, btn_skip, btn_reshuffle):
            b.setFont(QFont('Microsoft YaHei', 10))
        btn_save.setStyleSheet('QPushButton{background:#1D9E75;color:white;padding:7px;}')
        btn_save.clicked.connect(self.save_current)
        btn_clear.clicked.connect(self.canvas.clear)
        btn_undo.clicked.connect(self.undo_last)
        btn_skip.clicked.connect(self.skip_char)
        btn_reshuffle.clicked.connect(self.rebuild_queue)
        QShortcut(QKeySequence('Return'), self).activated.connect(self.save_current)
        QShortcut(QKeySequence('C'), self).activated.connect(self.canvas.clear)
        QShortcut(QKeySequence('S'), self).activated.connect(self.skip_char)

        thumb_row = QHBoxLayout()
        for _ in range(NEED):
            lab = QLabel()
            lab.setFixedSize(72, 72)
            lab.setAlignment(Qt.AlignCenter)
            lab.setStyleSheet('border:1px solid #bbb; background:#fafafa;')
            lab.setText('·')
            thumb_row.addWidget(lab)
            self.thumbs.append(lab)

        right = QVBoxLayout()
        right.addWidget(self.pool_box)
        right.addWidget(self.target_lab)
        row = QHBoxLayout()
        row.addWidget(QLabel('本字进度:'))
        row.addWidget(self.progress_lab)
        row.addStretch(1)
        right.addLayout(row)
        right.addLayout(thumb_row)
        right.addWidget(self.style_lab)
        right.addWidget(self.explain)
        for b in (btn_save, btn_clear, btn_undo, btn_skip, btn_reshuffle):
            right.addWidget(b)
        right.addWidget(self.stat_lab)
        right.addWidget(self.done_lab)
        right.addStretch(1)

        left = QVBoxLayout()
        left.addWidget(self.canvas)
        left.addWidget(self.traj_lab)

        layout = QHBoxLayout()
        layout.addLayout(left)
        layout.addLayout(right)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # 断点续采: 上次没采满 4 张的字, 若它仍是「未见字」则先采完。
        carry = None
        if (self.current and 0 < self.saved < NEED
                and self.current in self.chars):
            self.exempt = {self.current}
            self._compute_pools()
            if self.current in self.unseen:
                carry = (self.current, self.saved)
        if carry:
            self._resume(*carry)
        else:
            self.rebuild_queue()

    # ---------- 候选池 ----------
    def _compute_pools(self):
        """硬规则: 库里已有的字(除正在采的那个)全部排除"""
        self.baseline = counts_of_user_data()
        excluded = {c for c, n in self.baseline.items()
                    if n > 0 and c not in self.exempt}
        self.unseen = [c for c in self.chars if c not in excluded]
        self.meta, self.seed_fams, self.fam_seeds = difficulty_index(
            self.chars, self.unseen, self.seeds, self.worst,
            self.fam, self.glyphs, self.pcts)
        self.pool_all = list(self.unseen)
        self.pool_hard = [c for c in self.unseen if self.meta[c]['is_hard']]
        self.pool_focus = [c for c in self.unseen if self.meta[c]['is_focus']]
        self.pool_family = [c for c in self.pool_hard
                            if 'S3' in self.meta[c]['signals']]

    def _pool_for_mode(self):
        if self.pool == 'unseen_hard_only':
            return list(self.pool_focus)
        if self.pool == 'unseen_family':
            return list(self.pool_family)
        return list(self.pool_all)

    def _ordered_candidates(self):
        """难字按难度分降序(同分随机抖动) -> 普通未见字 -> 跳过的字"""
        rng = random.Random(int(time.time()))
        cand = self._pool_for_mode()
        hard = [c for c in cand if self.meta[c]['is_hard']]
        plain = [c for c in cand if not self.meta[c]['is_hard']]
        rng.shuffle(hard)
        rng.shuffle(plain)
        hard.sort(key=lambda c: self.meta[c]['score'] + rng.random() * 0.6,
                  reverse=True)

        def split(lst):
            return ([c for c in lst if c not in self.skipped],
                    [c for c in lst if c in self.skipped])

        hm, hs = split(hard)
        pm, ps = split(plain)
        return hm + pm + hs + ps

    def rebuild_queue(self):
        # 重排一律按实时笔迹库重算, 不豁免任何字
        self.exempt = set()
        self._compute_pools()
        self.queue = self._ordered_candidates()
        self.current = ''
        self.saved = 0
        self.saved_files = []
        self.saved_trajs = []
        if self.queue:
            self._next_char()
        else:
            self.target_lab.setText('✓')
            self.explain.setPlainText(
                '该口径下已经没有「没见过 + 难」的字了。\n'
                '可切换到「未见 · 难字优先」继续采其余未见字，'
                '或运行 一键优化训练.py 把新笔迹用起来。')
            self.stat_lab.setText(f'笔迹库 {sum(self.baseline.values())} 张 / '
                                  f'{len(self.baseline)} 字；未见字 '
                                  f'{len(self.pool_all)} 个 / 重点难字 '
                                  f'{len(self.pool_focus)} 个')
            self._save_state()

    def _next_char(self):
        while self.queue:
            ch = self.queue.pop(0)
            # 双保险: 只接受当前口径下的未见字, 且已在采/已采完的字不重抽
            if ch not in self.unseen or ch in self.exempt:
                continue
            self.current = ch
            self.saved = 0
            self.saved_files = []
            self.saved_trajs = []
            for lab in self.thumbs:
                lab.clear()
                lab.setText('·')
            self.canvas.clear()
            self._refresh()
            return
        self.rebuild_queue()

    def skip_char(self):
        if self.current:
            self.skipped.add(self.current)
            self.done_lab.setText(
                f'跳过「{self.current}」，下次重排时排到队尾'
                '（库里已有的字仍不会重抽）')
        self._next_char()

    def _on_pool_change(self, idx):
        self.pool = POOL_KEYS[idx]
        self.rebuild_queue()

    # ---------- 状态 ----------
    def _save_state(self):
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump({'version': 2, 'pool': self.pool,
                       'queue': self.queue[:200],
                       'current': self.current, 'saved': self.saved}, f,
                      ensure_ascii=False, indent=1)

    def _load_state(self):
        """队列一律按新口径重建, 旧 state 的 queue(旧口径随机序)作废"""
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, encoding='utf-8') as f:
                s = json.load(f)
            p = s.get('pool', POOL_KEYS[0])
            self.pool = p if p in POOL_KEYS else LEGACY_POOL.get(p, POOL_KEYS[0])
            self.current = s.get('current', '')
            self.saved = int(s.get('saved', 0))
            self.queue = []
        except Exception:
            pass

    def _resume(self, ch, n):
        """恢复上次没采满 4 张的字(它属于已经在采的字, 不算重抽老字)"""
        self.current, self.saved = ch, n
        self.queue = [c for c in self._ordered_candidates() if c != ch]
        self.saved_files = []
        self.saved_trajs = []
        files = sorted(glob.glob(os.path.join(USER_DIR, f'{ch}_*.png')))[-n:]
        for i, fn in enumerate(files[:NEED]):
            try:
                a = cv2.imdecode(np.fromfile(fn, np.uint8), cv2.IMREAD_GRAYSCALE)
                if a is not None and a.shape == (64, 64):
                    self._set_thumb(i, a)
                    self.saved_files.append(fn)
            except Exception:
                pass
        self.canvas.clear()
        self._refresh()
        self.done_lab.setText(f'续采上次未完成的「{ch}」（已有 {n}/{NEED} 张）')

    # ---------- 说明 ----------
    def _refresh(self):
        if not self.current:
            return
        ch = self.current
        cnt = counts_of_user_data()
        have = cnt.get(ch, 0)
        m = self.meta.get(ch, {'score': 0.0, 'reasons': [], 'signals': []})
        idx = self.chars.index(ch) if ch in self.chars else -1
        is_ext = idx >= BASE_N

        if ch in self.exempt:
            head = (f'「{ch}」是上次没采满 4 张、断点续采的字，不算重新抽老字；'
                    f'采完这 {NEED - self.saved} 张之后，'
                    '后面的字一律从没见过的字里抽。')
            head += f'\n已写 {self.saved} / {NEED} 张。'
        else:
            n_excluded = sum(1 for c, v in self.baseline.items()
                             if v > 0 and c not in self.exempt)
            head = (f'「{ch}」在启动时的笔迹库里是 0 张 → 新字。'
                    f'库里已有的 {n_excluded} 个字全部排除，不再重复抽。')
            if self.saved:
                head += f'\n本次会话已写 {self.saved} / {NEED} 张。'
        if m['reasons']:
            diff = '【为什么算「难」】难度分 ' + f"{m['score']:.1f}" + \
                   '　命中：' + '、'.join(m['signals']) + '\n· ' + \
                   '\n· '.join(m['reasons'])
        else:
            diff = ('【难度】未命中难度信号（难度分 '
                    f"{m['score']:.1f}），排在难字之后。\n"
                    '当前口径允许全部未见字；切到「未见 · 仅难字」可只出难字。')

        dens = ink_density(ch)
        if dens is None:
            dens_line = '笔画密度：无法估算（缺楷体字体）'
        else:
            level = '偏高' if dens > 0.28 else ('偏低' if dens < 0.15 else '中等')
            tip = ('笔画多、容易糊在一起，写的时候注意留白' if dens > 0.28 else
                   '笔画少、特征稀疏，容易和形近字撞' if dens < 0.15 else
                   '结构适中，保持笔画完整即可')
            dens_line = f'笔画密度 {dens*100:.0f}%（{level}）：{tip}'

        fams = self.fam.get(ch, [])
        if fams:
            fam_line = '偏旁 / 形近组：' + '、'.join(fams)
            hot = [f for f in fams if f in self.seed_fams]
            if hot:
                fam_line += '；其中 ' + '、'.join(hot) + ' 组出现过历史难例'
        else:
            fam_line = '偏旁 / 形近组：未命中预设组，按你的自然写法书写即可'

        text = (
            f'【为什么抽到「{ch}」】\n{head}\n\n'
            f'{diff}\n\n'
            f'【模型视角】\n'
            f'· 字符表索引 {idx} / {len(self.chars)}；'
            f'{"扩类新增的二级字（原模型没有它）" if is_ext else "GB2312 一级规范字"}\n'
            f'· 抽取口径：{POOL_DESC[self.pool]}\n'
            f'· {dens_line}\n'
            f'· {fam_line}\n\n'
            f'【4 张为什么要写得不一样】\n'
            f'单张只会让模型记住一个偶然写法。4 张分别覆盖端正、自然、连笔倾斜、'
            f'不同大小，等于给这个字建一份"书写方差样本"，微调后就是你的个人签名。\n\n'
            f'当前：第 {self.saved + 1} / {NEED} 张'
        )
        self.explain.setPlainText(text)
        self.target_lab.setText(ch)
        self.progress_lab.setText(f'{self.saved} / {NEED}')
        self.style_lab.setText(STYLE_HINTS[min(self.saved, NEED - 1)])
        self.stat_lab.setText(
            f'笔迹库：{sum(cnt.values())} 张 / {len(cnt)} 字（已有字不重抽）　|　'
            f'未见字 {len(self.pool_all)} 个 / 重点难字 {len(self.pool_focus)} 个'
            f'　|　队列剩余 {len(self.queue)} 字')
        self._save_state()

    def _set_thumb(self, i, small):
        big = cv2.resize(small, (70, 70), interpolation=cv2.INTER_NEAREST)
        q = QImage(big.data, 70, 70, 70, QImage.Format_Grayscale8).copy()
        self.thumbs[i].setPixmap(QPixmap.fromImage(q))

    # ---------- 采集 ----------
    def _on_rec(self, n, ns, dur):
        now = time.perf_counter()
        seq = ''
        # 区域序按 ~8Hz 刷新，别让每个 move 事件都重算
        if _region is not None and n >= 4 and (now - self._rec_t) > 0.12:
            self._rec_t = now
            try:
                r = _region.region_sequence(self.canvas.rec.x, self.canvas.rec.y,
                                            self.canvas.rec.t, self.canvas.rec.pen)
                if r['ids'].size:
                    ids = r['ids']
                    # 5x5 网格最多 25 格，整串会超过标签宽度；只显示起手 8 格 + 总格数
                    txt = '→'.join(str(int(c) + 1) for c in ids[:8])
                    if len(ids) > 8:
                        txt += f'…(共{len(ids)}格)'
                    seq = f'  ·  区域序 {txt}'
            except Exception:
                pass
        extra = ''
        if TRAJ_ON:
            saved = sum(1 for p in self.saved_trajs if p)
            extra = f'  ·  已存 {saved}'
        self.traj_lab.setText(f'轨迹：{n} 点 / {dur / 1000.0:.1f}s{seq}{extra}')

    def _save_traj(self, ts_ms):
        """与 PNG 同名同时间戳写轨迹 npz + 侧车 json; 失败不阻断位图采集。"""
        if not TRAJ_ON:
            return None
        try:
            ts_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(ts_ms / 1000.0))
            tp = os.path.join(TRAJ_DIR, f'{self.current}_{ts_str}.npz')
            k = 2
            while os.path.exists(tp):
                tp = os.path.join(TRAJ_DIR, f'{self.current}_{ts_str}_{k}.npz')
                k += 1
            return self.canvas.rec.save_npz(tp, self.current)
        except Exception as ex:                      # 轨迹是旁路, 不能拖垮采集
            self.traj_lab.setText(f'轨迹写入失败：{ex}')
            return None

    def save_current(self):
        if not self.current:
            return
        small = preprocess64(self.canvas.to_gray())
        if small is None or int((small > 40).sum()) < 100:
            self.style_lab.setText('画布是空的（或笔画太少），请先写这个字')
            return
        os.makedirs(USER_DIR, exist_ok=True)
        ts = int(time.time() * 1000)
        fn = os.path.join(USER_DIR, f'{self.current}_{ts}.png')
        while os.path.exists(fn):
            ts += 1
            fn = os.path.join(USER_DIR, f'{self.current}_{ts}.png')
        ok, buf = cv2.imencode('.png', small)      # 中文路径必须 imencode + tofile
        if not ok:
            self.style_lab.setText('图像编码失败')
            return
        buf.tofile(fn)
        self.saved_files.append(fn)
        self.saved_trajs.append(self._save_traj(ts))
        self._set_thumb(self.saved, small)
        self.saved += 1
        self.canvas.clear()
        if self.saved >= NEED:
            self.done_lab.setText(f'✓ 已完成「{self.current}」的 4 张个性化签名')
            self.style_lab.setText(f'✓「{self.current}」4 张已凑齐，自动换下一个字')
            self._save_state()
            QtWidgets.QApplication.processEvents()
            self.exempt = set()
            self._next_char()
        else:
            self._refresh()

    def undo_last(self):
        if not self.saved_files:
            self.style_lab.setText('本次会话还没有可撤销的样本')
            return
        fn = self.saved_files.pop()
        try:
            os.remove(fn)
        except OSError:
            pass
        # 轨迹与位图同进同退, 否则会留下没有对应 PNG 的孤儿 npz
        if len(self.saved_trajs) > len(self.saved_files):
            tp = self.saved_trajs.pop()
            if tp:
                for p in (tp, os.path.splitext(tp)[0] + '.json'):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
        self.saved = max(0, self.saved - 1)
        for i in range(self.saved, NEED):
            self.thumbs[i].clear()
            self.thumbs[i].setText('·')
        self.canvas.clear()
        self._refresh()
        self.style_lab.setText('已撤销上一张，重新写这一张')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
