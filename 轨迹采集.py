# -*- coding: utf-8 -*-
"""
笔迹轨迹采集器 (PySide6) · 独立文件，不 import 主画板
=====================================================
产出：hanzi/traj_data/<字>_<时间戳>.npz（格式见 hanzi/traj/TRAJ_FORMAT.md）

与 个性化签名采集.py 的关系：只复用它的**纯函数**候选池口径
（未见字 / 难字：counts_of_user_data / hard_seeds / difficulty_index / glyph_index），
不 import 它的 GUI，也不写 hanzi/user_data/，两套数据完全隔离。

关键点：
  * move 事件全采；同一位置且距上一点 < 2 ms 的重复事件丢弃
  * pen 由 0→1 判定新笔画，逐点写 stroke_idx（抬笔点 = -1），
    便于日后把真实笔迹的笔画数/顺序与 stroke_medians.json 的标准笔顺逐笔比对
  * 断点续采：state 存 hanzi/traj/sign_progress.json
  * 保存用 numpy.savez_compressed + `open(path,'wb')` 传文件对象（中文路径必需）

用法：
  D:/Anaconda/envs/pytorch/python.exe 轨迹采集.py
  D:/Anaconda/envs/pytorch/python.exe 轨迹采集.py --pool hard --need 6
  QT_QPA_PLATFORM=offscreen python 轨迹采集.py --smoke     # 离屏冒烟，不产生任何文件
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
HANZI = os.path.join(ROOT, 'hanzi')
TRAJ_DIR = os.path.join(HANZI, 'traj')
TRAJ_DATA = os.path.join(HANZI, 'traj_data')
STATE_FILE = os.path.join(TRAJ_DIR, 'sign_progress.json')
CHARSET = os.path.join(HANZI, 'charset.json')
USER_DIR = os.path.join(HANZI, 'user_data')

CANVAS = 420
MIN_DT_MS = 2.0          # 同位置去重的最小间隔
SAME_POS_PX = 0.5
POOL_CHOICES = {'hard_first': '未见·难字优先', 'hard': '未见·仅难字', 'all': '未见·全部'}


# ------------------------------------------------------------------ 候选池
def _load_sign_module():
    """按路径载入 个性化签名采集.py 的纯函数（不触发 GUI）。"""
    fp = os.path.join(ROOT, '个性化签名采集.py')
    if not os.path.exists(fp):
        return None
    try:
        spec = importlib.util.spec_from_file_location('_sign_collector', fp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:            # noqa: BLE001
        print(f'[!] 候选池模块载入失败，退回简易池: {e}')
        return None


def _glyph_index_readonly(mod, chars):
    """只读版 glyph_index：命中缓存直接用，未命中就地计算，**绝不写盘**。"""
    sig = hashlib.md5(''.join(chars).encode('utf-8')).hexdigest()
    cache = mod._load_json(mod.META_FILE)
    if cache.get('charset_sig') == sig and len(cache.get('glyphs') or {}) >= len(chars):
        return cache['glyphs'], cache['pcts']
    font = mod._get_font()
    glyphs = {c: (mod.glyph_stats(c, font) or {'px': 0, 'parts': 0, 'cells': 0, 'dens': None})
              for c in chars}
    px = np.array([glyphs[c]['px'] for c in chars], float)
    order = np.argsort(px)
    rank = {chars[i]: round(100.0 * r / max(1, len(chars) - 1), 1) for r, i in enumerate(order)}
    pcts = {'px_p70': int(np.percentile(px, 70)), 'px_p15': int(np.percentile(px, 15)),
            'px_rank': rank, 'note': 'readonly-computed'}
    return glyphs, pcts


def build_queue(pool='hard_first', n=300, seed=0, need=6):
    """返回 [(字, 层级, 难度分, 理由)]。层级 focus > hard > other。"""
    mod = _load_sign_module()
    with open(CHARSET, encoding='utf-8') as f:
        chars = json.load(f)['chars']
    if mod is None:
        cnt = {}
        for fn in os.listdir(USER_DIR) if os.path.isdir(USER_DIR) else []:
            if fn.endswith('.png') and '_' in fn and not fn.startswith('_'):
                cnt[fn.rsplit('_', 1)[0]] = cnt.get(fn.rsplit('_', 1)[0], 0) + 1
        unseen = [c for c in chars if cnt.get(c, 0) == 0]
        rng = random.Random(seed)
        rng.shuffle(unseen)
        return [(c, 'other', 0.0, '退回简易池') for c in unseen[:n]]

    cnt = mod.counts_of_user_data()
    unseen = [c for c in chars if cnt.get(c, 0) < need]
    seeds, worst = mod.hard_seeds(chars)
    fam = mod.fam_index()
    glyphs, pcts = _glyph_index_readonly(mod, chars)
    meta, _, _ = mod.difficulty_index(chars, unseen, seeds, worst, fam, glyphs, pcts)

    tiers = {'focus': [], 'hard': [], 'other': []}
    for c in unseen:
        m = meta.get(c) or {}
        t = 'focus' if m.get('is_focus') else ('hard' if m.get('is_hard') else 'other')
        tiers[t].append((c, t, float(m.get('score', 0.0)),
                         '；'.join(m.get('reasons') or [])[:120]))
    rng = random.Random(seed)
    for k in tiers:
        rng.shuffle(tiers[k])
        tiers[k].sort(key=lambda x: -x[2])
    if pool == 'hard':
        pool_list = tiers['focus'] + tiers['hard']
    elif pool == 'all':
        pool_list = tiers['focus'] + tiers['hard'] + tiers['other']
    else:                                   # hard_first：难字优先，其余补足
        n_focus = int(n * 0.5)
        n_hard = int(n * 0.3)
        pool_list = (tiers['focus'][:max(n_focus, 1)] + tiers['hard'][:max(n_hard, 1)]
                     + tiers['other'])
        rest = [x for x in tiers['focus'][max(n_focus, 1):]
                + tiers['hard'][max(n_hard, 1):]]
        pool_list += rest
    return pool_list[:n]


# ------------------------------------------------------------------ 轨迹缓冲
class TrajBuffer:
    """纯 Python 轨迹缓冲，不依赖 Qt，便于离屏冒烟直接测。"""

    def __init__(self):
        self.pts = []          # list of dict(x,y,t,pen,stroke)
        self.stroke_id = -1
        self.pen_down = False
        self.t0 = None

    # -- 时间
    def _now_ms(self):
        if self.t0 is None:
            self.t0 = time.perf_counter()
            return 0
        return int(round((time.perf_counter() - self.t0) * 1000.0))

    def _last(self):
        return self.pts[-1] if self.pts else None

    # -- 事件
    def down(self, x, y):
        if self.pen_down:
            return
        self.pen_down = True
        self.stroke_id += 1
        t = self._now_ms()
        if self.pts and t <= self.pts[-1]['t']:
            t = self.pts[-1]['t'] + 1
        self.pts.append({'x': float(x), 'y': float(y), 't': t, 'pen': 1,
                         'stroke': self.stroke_id})

    def move(self, x, y):
        if not self.pen_down:
            return
        t = self._now_ms()
        last = self._last()
        if last is not None:
            d = ((x - last['x']) ** 2 + (y - last['y']) ** 2) ** 0.5
            if d < SAME_POS_PX and (t - last['t']) < MIN_DT_MS:
                return
            if t <= last['t']:
                t = last['t'] + 1
        self.pts.append({'x': float(x), 'y': float(y), 't': t, 'pen': 1,
                         'stroke': self.stroke_id})

    def up(self, x=None, y=None):
        if not self.pen_down:
            return
        self.pen_down = False
        last = self._last()
        px = last['x'] if (x is None and last) else (x if x is not None else 0)
        py = last['y'] if (y is None and last) else (y if y is not None else 0)
        t = self._now_ms()
        if last is not None and t <= last['t']:
            t = last['t'] + 1
        self.pts.append({'x': float(px), 'y': float(py), 't': t, 'pen': 0, 'stroke': -1})

    def undo(self):
        """撤掉最后一笔（含其后的抬笔点）。"""
        while self.pts and self.pts[-1]['pen'] == 0:
            self.pts.pop()
        sid = self.pts[-1]['stroke'] if self.pts else -1
        while self.pts and self.pts[-1]['stroke'] == sid and self.pts[-1]['pen'] == 1:
            self.pts.pop()
        self.stroke_id = max(-1, sid - 1)
        self.pen_down = False
        if not self.pts:
            self.t0 = None

    def clear(self):
        self.pts = []
        self.stroke_id = -1
        self.pen_down = False
        self.t0 = None

    # -- 统计
    @property
    def n_points(self):
        return len(self.pts)

    @property
    def n_strokes(self):
        ps = [p['pen'] for p in self.pts]
        if not ps:
            return 0
        return int(sum(1 for i, v in enumerate(ps) if v == 1 and (i == 0 or ps[i - 1] == 0)))

    @property
    def duration_ms(self):
        if not self.pts:
            return 0
        return int(self.pts[-1]['t'] - self.pts[0]['t'])

    def to_arrays(self):
        if not self.pts:
            return None
        x = np.asarray([p['x'] for p in self.pts], np.float32)
        y = np.asarray([p['y'] for p in self.pts], np.float32)
        t = np.asarray([p['t'] for p in self.pts], np.int64)
        pen = np.asarray([p['pen'] for p in self.pts], np.uint8)
        sid = np.asarray([p['stroke'] for p in self.pts], np.int32)
        return x, y, t, pen, sid

    def problems(self):
        """保存前的自洽检查，返回问题列表（空 = 可保存）。"""
        out = []
        if not self.pts:
            return ['没有笔迹']
        x, y, t, pen, sid = self.to_arrays()
        if not np.all(np.diff(t) > 0):
            out.append('t 非严格递增')
        if pen[0] != 1:
            out.append('首点不是落笔')
        if pen[-1] != 0:
            out.append('末点不是抬笔（请松手后再保存）')
        if x.min() < 0 or y.min() < 0 or x.max() > CANVAS or y.max() > CANVAS:
            out.append('坐标越界')
        # 每笔点数
        runs = np.flatnonzero((pen[1:] == 1) & (pen[:-1] == 0)) + 1
        idx = np.concatenate([[0], runs]) if pen[0] == 1 else runs
        for i, a in enumerate(idx):
            b = idx[i + 1] if i + 1 < len(idx) else len(pen)
            if pen[a] == 1 and (b - a) < 2:
                out.append(f'第 {i + 1} 笔只有 1 个点')
                break
        if self.duration_ms < 200:
            out.append('用时过短（<200ms），可能误触')
        return out


def save_traj(buf, ch, out_dir=TRAJ_DATA, device='mouse', writer_id='u0'):
    """按 TRAJ_FORMAT 落盘，返回 (npz 路径, 问题列表)。"""
    probs = buf.problems()
    if probs:
        return None, probs
    os.makedirs(out_dir, exist_ok=True)
    x, y, t, pen, sid = buf.to_arrays()
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f'{ch}_{ts}.npz')
    payload = {
        'x': x, 'y': y, 't': t, 'pen': pen, 'stroke_idx': sid,
        'pressure': np.full(len(x), 0.5, np.float32),
        'canvas_w': np.int32(CANVAS), 'canvas_h': np.int32(CANVAS),
        'char': np.asarray([ch], dtype='<U1'),
        'source': np.asarray(['real'], dtype='<U4'),
    }
    with open(path, 'wb') as f:              # 中文路径必须传文件对象
        np.savez_compressed(f, **payload)
    meta = {
        'char': ch, 'ts': time.strftime('%Y-%m-%d %H:%M:%S'), 'device': device,
        'source': 'real', 'writer_id': writer_id,
        'canvas_w': CANVAS, 'canvas_h': CANVAS,
        'n_points': int(len(x)), 'n_strokes': int(buf.n_strokes),
        'duration_ms': int(buf.duration_ms), 'app': '轨迹采集.py v1.0',
        'format': 'traj-v1',
    }
    with open(os.path.splitext(path)[0] + '.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return path, []


# ------------------------------------------------------------------ GUI
def make_window(pool='hard_first', seed=0, need=6, n=300, smoke=False):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QColor, QFont, QPainter, QPen
    from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMainWindow, QPushButton,
                                   QVBoxLayout, QWidget)

    class Canvas(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setFixedSize(CANVAS, CANVAS)
            self.setAttribute(Qt.WA_OpaquePaintEvent, True)
            self.buf = TrajBuffer()
            self.on_change = None

        def paintEvent(self, ev):                      # noqa: N802
            p = QPainter(self)
            p.fillRect(self.rect(), QColor(255, 255, 255))
            p.setPen(QPen(QColor(232, 232, 228), 1))
            p.drawRect(0, 0, CANVAS - 1, CANVAS - 1)
            p.setPen(QPen(QColor(20, 20, 20), 14, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            cur = None
            for q in self.buf.pts:
                if q['pen'] == 1:
                    if cur is None:
                        cur = [q]
                    else:
                        cur.append(q)
                else:
                    if cur:
                        self._stroke(p, cur); cur = None
            if cur:
                self._stroke(p, cur)
            p.end()

        @staticmethod
        def _stroke(p, pts):
            if len(pts) == 1:
                p.drawPoint(int(pts[0]['x']), int(pts[0]['y']))
                return
            for a, b in zip(pts[:-1], pts[1:]):
                p.drawLine(int(a['x']), int(a['y']), int(b['x']), int(b['y']))

        def _notify(self):
            if self.on_change:
                self.on_change()

        def mousePressEvent(self, e):                  # noqa: N802
            if e.button() == Qt.LeftButton:
                pos = e.position()
                self.buf.down(pos.x(), pos.y())
                self.update(); self._notify()

        def mouseMoveEvent(self, e):                   # noqa: N802
            if self.buf.pen_down:
                pos = e.position()
                self.buf.move(pos.x(), pos.y())
                self.update(); self._notify()

        def mouseReleaseEvent(self, e):                # noqa: N802
            if e.button() == Qt.LeftButton:
                pos = e.position()
                self.buf.up(pos.x(), pos.y())
                self.update(); self._notify()

        def clear(self):
            self.buf.clear(); self.update(); self._notify()

        def undo(self):
            self.buf.undo(); self.update(); self._notify()

    class Win(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('笔迹轨迹采集 · 标准笔顺数据集')
            self.smoke = smoke
            self.pool = pool
            self.need = need
            self.queue = []
            self.state = {'queue': [], 'idx': 0, 'saved': {}, 'skipped': [], 'seed': seed}
            if not smoke:
                self._load_state()
                if not self.state['queue']:
                    self.queue = build_queue(pool, n, seed, need)
                    self.state['queue'] = [q[0] for q in self.queue]
                    self.state['tiers'] = {q[0]: {'tier': q[1], 'score': q[2], 'why': q[3]}
                                           for q in self.queue}
                    self._save_state()
            else:
                self.queue = [('测', 'focus', 1.0, 'smoke')]
            self.canvas = Canvas(self)
            self.canvas.on_change = self._refresh

            central = QWidget()
            lay = QVBoxLayout(central)
            self.lab_char = QLabel('-')
            self.lab_char.setAlignment(Qt.AlignCenter)
            f = QFont('KaiTi', 64, QFont.Bold)
            self.lab_char.setFont(f)
            self.lab_char.setStyleSheet('color:#b3261e;')
            self.lab_hud = QLabel('点数 0 | 笔数 0 | 用时 0.0s')
            self.lab_hud.setAlignment(Qt.AlignCenter)
            self.lab_hud.setStyleSheet('font-size:15px;color:#333;')
            self.lab_why = QLabel('')
            self.lab_why.setWordWrap(True)
            self.lab_why.setAlignment(Qt.AlignCenter)
            self.lab_why.setStyleSheet('font-size:12px;color:#666;')
            lay.addWidget(self.lab_char)
            lay.addWidget(self.lab_why)
            lay.addWidget(self.canvas, 0, Qt.AlignHCenter)
            lay.addWidget(self.lab_hud)

            row = QHBoxLayout()
            self.b_undo = QPushButton('撤销 (Ctrl+Z)')
            self.b_clear = QPushButton('清空 (Ctrl+Backspace)')
            self.b_skip = QPushButton('跳过 (S)')
            self.b_save = QPushButton('保存并下一个 (Enter)')
            for b in (self.b_undo, self.b_clear, self.b_skip, self.b_save):
                row.addWidget(b)
            lay.addLayout(row)
            self.setCentralWidget(central)

            self.b_undo.clicked.connect(self.canvas.undo)
            self.b_clear.clicked.connect(self.canvas.clear)
            self.b_skip.clicked.connect(self.skip)
            self.b_save.clicked.connect(self.save_next)

            self.timer = QTimer(self)
            self.timer.timeout.connect(self._refresh)
            self.timer.start(100)
            self._show_char()

        # -- 状态
        def _load_state(self):
            try:
                with open(STATE_FILE, encoding='utf-8') as f:
                    st = json.load(f)
                if st.get('queue'):
                    self.state = st
                    self.queue = [(c, (st.get('tiers', {}).get(c) or {}).get('tier', 'other'),
                                   (st.get('tiers', {}).get(c) or {}).get('score', 0.0),
                                   (st.get('tiers', {}).get(c) or {}).get('why', ''))
                                  for c in st['queue']]
            except Exception:
                pass

        def _save_state(self):
            if self.smoke:
                return
            os.makedirs(TRAJ_DIR, exist_ok=True)
            self.state['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=1)

        def cur_char(self):
            i = self.state.get('idx', 0)
            return self.queue[i][0] if 0 <= i < len(self.queue) else None

        def _show_char(self):
            ch = self.cur_char()
            i = self.state.get('idx', 0)
            self.lab_char.setText(ch or '—')
            if ch and i < len(self.queue):
                _, tier, score, why = self.queue[i]
                self.lab_why.setText(f'[{POOL_CHOICES.get(self.pool, self.pool)}] '
                                     f'{i + 1}/{len(self.queue)} · {tier} · 难度 {score:.1f}'
                                     + (f'\n{why}' if why else ''))
            self.setWindowTitle(f'笔迹轨迹采集 · {ch or "-"} · {i + 1}/{len(self.queue)}')
            self._refresh()

        def _refresh(self):
            b = self.canvas.buf
            self.lab_hud.setText(f'点数 {b.n_points} | 笔数 {b.n_strokes} | '
                                 f'用时 {b.duration_ms / 1000.0:.1f}s')

        def skip(self):
            ch = self.cur_char()
            if ch and ch not in self.state['skipped']:
                self.state['skipped'].append(ch)
            self.canvas.clear()
            self.state['idx'] = min(self.state.get('idx', 0) + 1, max(len(self.queue) - 1, 0))
            self._save_state()
            self._show_char()

        def save_next(self):
            ch = self.cur_char()
            if not ch:
                return
            if self.smoke:
                self._refresh()
                return
            path, probs = save_traj(self.canvas.buf, ch)
            if probs:
                self.lab_hud.setText('无法保存：' + '；'.join(probs))
                return
            self.state['saved'][ch] = self.state['saved'].get(ch, 0) + 1
            print(f'saved {path}  ({self.canvas.buf.n_points} 点 / '
                  f'{self.canvas.buf.n_strokes} 笔 / {self.canvas.buf.duration_ms} ms)')
            self.canvas.clear()
            self.state['idx'] = min(self.state.get('idx', 0) + 1, max(len(self.queue) - 1, 0))
            self._save_state()
            self._show_char()

        def keyPressEvent(self, e):                     # noqa: N802
            if e.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.save_next()
            elif e.key() == Qt.Key_Z and (e.modifiers() & Qt.ControlModifier):
                self.canvas.undo()
            elif e.key() == Qt.Key_Backspace and (e.modifiers() & Qt.ControlModifier):
                self.canvas.clear()
            elif e.key() == Qt.Key_S:
                self.skip()
            else:
                super().keyPressEvent(e)

    return Win()


# ------------------------------------------------------------------ 冒烟
def smoke():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    before = set(os.listdir(TRAJ_DATA)) if os.path.isdir(TRAJ_DATA) else set()
    app = QApplication.instance() or QApplication(sys.argv)
    win = make_window(smoke=True)
    win.show()
    app.processEvents()

    # 直接在缓冲里注入 3 笔（绕过鼠标事件），验证 stroke_idx / pen 段 / 点数统计
    buf = win.canvas.buf
    import math
    for k in range(3):
        t0 = k * 300
        buf.t0 = time.perf_counter() - (10 - t0 / 1000.0)
        buf.down(40 + k * 100, 40)
        for i in range(1, 12):
            win.canvas.buf.move(40 + k * 100 + i * 8, 40 + int(30 * math.sin(i / 3.0)))
        buf.up(40 + k * 100 + 88, 40)
    app.processEvents()
    win._refresh()
    assert buf.n_strokes == 3, f'笔数应为 3，实得 {buf.n_strokes}'
    assert buf.n_points >= 3 * 12, f'点数异常 {buf.n_points}'
    x, y, t, pen, sid = buf.to_arrays()
    assert np.all(np.diff(t) > 0), 't 非严格递增'
    assert pen[0] == 1 and pen[-1] == 0
    runs = int(((pen[1:] == 1) & (pen[:-1] == 0)).sum() + (pen[0] == 1))
    assert runs == 3, f'pen 段数 {runs} != 3'
    assert int(sid[sid >= 0].max()) == 2, 'stroke_idx 未从 0 递增'
    # 撤销一笔
    buf.undo()
    assert buf.n_strokes == 2, f'撤销后笔数 {buf.n_strokes}'
    probs = buf.problems()
    print(f'  注入后 点数={buf.n_points} 笔数={buf.n_strokes} 用时={buf.duration_ms}ms')
    print(f'  自洽检查: {probs or "通过"}')
    win.close()
    app.processEvents()

    after = set(os.listdir(TRAJ_DATA)) if os.path.isdir(TRAJ_DATA) else set()
    new_files = after - before
    assert not new_files, f'冒烟产生了文件: {new_files}'
    print('  offscreen 冒烟通过：窗口可构建、笔迹缓冲/撤销/stroke_idx 正常、未产生任何笔迹文件')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', choices=list(POOL_CHOICES), default='hard_first')
    ap.add_argument('--need', type=int, default=6, help='每个字需要的轨迹条数')
    ap.add_argument('--n', type=int, default=300, help='候选队列长度')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')

    if args.smoke:
        return smoke()

    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    win = make_window(args.pool, args.seed, args.need, args.n)
    win.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
