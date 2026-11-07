# -*- coding: utf-8 -*-
"""
一键优化训练 · 自包含单文件 (deep-learning training protocol, gen-v3)

一个文件完成: 数据审计/扩类 -> 严格分层 -> 范式对照 -> 训练 -> 校准 ->
              拟合审计 + 泛化评估(自动, 无需人工盲测) -> 门槛 -> 部署 -> 报告。

不依赖项目内任何其它 .py；只依赖 torch / torchvision / opencv / numpy / PIL
以及外部数据文件(charset.json、user_data/*.png、hanzi_data/*.npy、基座 .pth)。

【深度学习训练标准 · 口径明示】
  train : 指纹去重后的用户训练分片 + HWDB 训练分片回放 + 扩类合成样本
  val   : 按字分层严格留出(每字最多 1 张, 跨轮固定) —— 只用于选模/早停/校准约束
  cal   : HWDB 测试集固定 4096 条 —— 只用于 Temperature Scaling
  test  : HWDB 测试集剩余条目 —— 独立测试, 不参与选模/校准/门槛调参
  拟合域: train 中的用户笔迹(教过的样本), 只作为产品信任度质量线, 不参与选模
  泛化域: val(用户同字新写法) / test(通用域) / 合成留出(弱泛化) / TTA 翻转率(鲁棒性)
  范式对照: 经典结构化特征 3NN(无学习) vs 深度 CNN, 只在 val 上比较

用法:
  D:/Anaconda/envs/pytorch/python.exe 一键优化训练.py                 # 完整一键优化并部署
  D:/Anaconda/envs/pytorch/python.exe 一键优化训练.py --report-only   # 只看当前模型的自动泛化报告
  D:/Anaconda/envs/pytorch/python.exe 一键优化训练.py --smoke         # 2 epoch 自检, 不部署
  D:/Anaconda/envs/pytorch/python.exe 一键优化训练.py --no-deploy     # 全流程但不部署
  D:/Anaconda/envs/pytorch/python.exe 一键优化训练.py --epochs 12 --retries 1
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict


def _bootstrap():
    """工作台解释器可能没有 cv2/torch, 自动切换到已装依赖的解释器"""
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return
    except ImportError:
        pass
    alt = r'D:\Anaconda\envs\pytorch\python.exe'
    if (os.path.exists(alt)
            and not os.environ.get('ONEFILE_BOOT')
            and os.path.normcase(os.path.abspath(sys.executable))
            != os.path.normcase(os.path.abspath(alt))):
        env = dict(os.environ, ONEFILE_BOOT='1')
        rc = subprocess.call([alt, os.path.abspath(__file__)] + sys.argv[1:],
                             cwd=os.path.dirname(os.path.abspath(__file__)), env=env)
        sys.exit(rc)


_bootstrap()

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2

# =====================================================================
# 0. 集中配置: 所有超参与路径(可用 CLI 覆盖, 报告会原样记录以保证可复现)
# =====================================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
HANZI = os.path.join(ROOT, 'hanzi')
DATA = os.path.join(ROOT, 'hanzi_data')
USER_DIR = os.path.join(HANZI, 'user_data')
CHARSET = os.path.join(HANZI, 'charset.json')
CHARSET_3755 = os.path.join(HANZI, 'charset_3755.json')
EXT_META = os.path.join(HANZI, 'ext_chars.json')
BASE_CKPT = os.path.join(HANZI, 'hanzi_char_mobilenet_backup_base.pth')
MAIN_CKPT = os.path.join(HANZI, 'hanzi_char_mobilenet.pth')
FT_CKPT = os.path.join(HANZI, 'hanzi_char_mobilenet_ft.pth')
PREV_CKPT = os.path.join(HANZI, 'hanzi_char_mobilenet_prev.pth')
SPLIT_FILE = os.path.join(HANZI, 'holdout_split.json')
HISTORY_FILE = os.path.join(HANZI, 'hard_history.json')
GAP_FILE = os.path.join(HANZI, 'paradigm_gap.json')
REPORT_FILE = os.path.join(HANZI, 'optimization_report.json')
EXT_TRAIN_I = os.path.join(DATA, 'ext_train_imgs.npy')
EXT_TRAIN_L = os.path.join(DATA, 'ext_train_lbls.npy')
EXT_VAL_I = os.path.join(DATA, 'ext_val_imgs.npy')
EXT_VAL_L = os.path.join(DATA, 'ext_val_lbls.npy')
# 签名风格迁移合成域: 由 hanzi/synth_from_signatures.py 产出。
# sign_train 只进训练; sign_test 是完全不训练的独立未见字, 仅用于风格迁移评估。
SIGN_TRAIN_I = os.path.join(DATA, 'sign_train_imgs.npy')
SIGN_TRAIN_L = os.path.join(DATA, 'sign_train_lbls.npy')
SIGN_TEST_I = os.path.join(DATA, 'sign_test_imgs.npy')
SIGN_TEST_L = os.path.join(DATA, 'sign_test_lbls.npy')
SIGN_META = os.path.join(HANZI, 'sign_synth_meta.json')

CFG = {
    'seed': 42,
    'epochs': 10,
    'batch': 128,
    'lr': 2e-5,
    'weight_decay': 2e-4,
    'label_smoothing': 0.03,
    'ema_decay': 0.999,
    'warmup_ratio': 0.05,
    'min_lr_ratio': 0.70,   # 短周期微调: 缓余弦(2e-5 -> 1.4e-5), 衰减过狠会欠拟合
    'grad_clip': 5.0,
    'per_class_draws': 64,
    'max_per_image': 32,
    'ext_boost': 1.5,
    'ext_repeat': 1.5,     # 扩类合成曝光倍数: 2.0 拟合完美但通用域差0.1pp, 1.0 反之
    'sign_repeat': 5,       # 签名风格合成样本的每轮曝光倍数(只进 train)
    'replay_quota': 12,
    # 2026-11-06: 24000 -> 36000。依据: 用户笔迹 964 -> 1434 张(+49%),
    # 按类均衡采样量随之从 16482 涨到约 24200, 但固定回放没变, 导致用户域梯度
    # 占比从 29.6% 升到 36.7%, 通用域被挤压。此处按同比例放大以恢复上一轮配比,
    # 不是为过门槛而调参。数据量再次显著增长时需同步复核本值。
    'random_replay': 36000,
    'raw_replay_ratio': 0.68,  # 更多 HWDB 原域样本不增强, 抵消新增合成域带来的通用域漂移
    'freeze_bn': True,
    'patience': 3,
    'guard': 2.5,
    'hw_cal_n': 4096,
    'tta_flip_n': 512,
    'min_holdout': 0.55,
    'new_cls_bias_offset': -0.15,
    'max_retries': 1,
    'base_n': 3755,
}
EXTRA_CONFUSABLES = (
    '穴空窍究窃窗帘穿突窄窝窘窠'
    '狙疽阻组租祖粗且目月且'
    '鸥鸦鸭鹅鹊鹃鹤鸡鸣鸟九匀勾勺'
    '幼幻兆桃挑跳逃窕姚'
    '朕腾腾雎雀焦集售准谁难'
    '鸠鸩鸡鸭鸦鸥鸽'
    '关淑女君子好逑窈窕雎鸠'
)
WRITE_STATE = True   # --smoke / --report-only 时关闭, 避免污染累积难例记忆与正式报告
FONTDIR = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts')
FONT_NAMES = ('msyh.ttc', 'msyhbd.ttc', 'simhei.ttf', 'simkai.ttf', 'simfang.ttf',
              'simsun.ttc', 'STZHONGS.TTF', 'STKAITI.TTF', 'STFANGSO.TTF',
              'STXIHEI.TTF', 'Deng.ttf', 'Dengb.ttf', 'STLITI.TTF')


# =====================================================================
# 1. 可复现性: 固定种子 + 确定性算子 + 环境版本记录
# =====================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def env_info(device):
    return {
        'python': sys.version.split()[0], 'torch': torch.__version__,
        'torchvision': torchvision.__version__, 'cv2': cv2.__version__,
        'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
        'device': device,
        'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
        'seed': CFG['seed'], 'cudnn_deterministic': True,
        'cudnn_benchmark': False, 'amp': True,
    }


# =====================================================================
# 2. 数据层: 字符表扩类 / 指纹去重 / 严格分层 / HWDB / 扩类合成保底
# =====================================================================
def ensure_charset():
    """扩类元数据必须「追加」而不是覆盖。

    历史 bug: 新增一个表外字时把 ext_chars.json 整体重写成 {新字: 当前长度},
    导致旧的扩类字(窈窕雎鸠)丢失 — 它们不再是 ext_cls, 合成保底样本被
    重新生成成只剩新字, 于是旧扩类字几乎没有训练样本、拟合域集中报错。
    现在: base_classes 恒取纯净基座 3755 类, 所有 >= base_n 的类都是新类。
    """
    with open(CHARSET, encoding='utf-8') as f:
        chars = json.load(f)['chars']
    tags = set()
    for fn in glob.glob(os.path.join(USER_DIR, '*.png')):
        m = re.match(r'(.+)_\d+\.png', os.path.basename(fn))
        if m and not m.group(1).startswith('_'):
            tags.add(m.group(1))
    if os.path.exists(CHARSET_3755):
        with open(CHARSET_3755, encoding='utf-8') as f:
            base_n = len(json.load(f)['chars'])
    else:
        base_n = len(chars)
    old = {}
    if os.path.exists(EXT_META):
        try:
            with open(EXT_META, encoding='utf-8') as f:
                old = json.load(f)
        except Exception:
            old = {}
    ext_map = {str(k): int(v) for k, v in (old.get('chars') or {}).items()}
    ext = [t for t in sorted(tags) if t not in chars]
    if ext:
        if not os.path.exists(CHARSET_3755):
            shutil.copy2(CHARSET, CHARSET_3755)
        start = len(chars)
        chars = chars + ext
        for i, c in enumerate(ext):
            ext_map[c] = start + i
        with open(CHARSET, 'w', encoding='utf-8') as f:
            json.dump({'chars': chars}, f, ensure_ascii=False)
        print(f'  字符表扩类: {start} -> {len(chars)} (新增 {"".join(ext)}, '
              f'累计扩类 {len(chars) - base_n} 字)', flush=True)
    # 自愈: 无论是否有新字, 元数据始终等于「所有超出基座的类」
    want = {chars[i]: i for i in range(base_n, len(chars))}
    if ext_map != want or int(old.get('base_classes', -1)) != base_n:
        ext_map = want
        with open(EXT_META, 'w', encoding='utf-8') as f:
            json.dump({'base_classes': base_n, 'chars': ext_map,
                       'note': '全部超出纯净基座的扩展字; 新增表外字只追加, '
                               'base_classes 恒为纯净基座类数'},
                      f, ensure_ascii=False, indent=2)
        print(f'  扩类元数据自愈: base={base_n}, 扩展字 {"".join(want)}', flush=True)
    ext_cls = set(range(base_n, len(chars)))
    return chars, ext_cls


def load_user(chars):
    """指纹去重 + 同图异标污染剔除(杜绝同图副本跨 train/val 泄漏)"""
    cmap = {c: i for i, c in enumerate(chars)}
    items, meta = [], []
    seen, fp_labels, dup = {}, defaultdict(set), 0
    for fn in sorted(glob.glob(os.path.join(USER_DIR, '*.png'))):
        base = os.path.basename(fn)
        m = re.match(r'(.+)_\d+\.png', base)
        if not m:
            continue
        tag = m.group(1)
        if tag.startswith('_'):
            continue
        if tag not in cmap:
            print(f'  跳过(字符表外): {tag}', flush=True)
            continue
        im = cv2.imdecode(np.fromfile(fn, np.uint8), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        im = np.asarray(im, np.uint8)
        fp = hashlib.md5(im.tobytes()).hexdigest()
        fp_labels[fp].add(cmap[tag])
        if fp in seen:
            dup += 1
            continue
        seen[fp] = len(items)
        items.append((im, cmap[tag]))
        meta.append({'fn': base, 'tag': tag, 'fp': fp})
    bad = sorted({seen[f] for f, labs in fp_labels.items() if len(labs) > 1 and f in seen})
    if bad:
        print('  ! 同图异标污染样本已剔除: ' + ', '.join(meta[i]['fn'] for i in bad[:6]),
              flush=True)
        bs = set(bad)
        items = [v for i, v in enumerate(items) if i not in bs]
        meta = [v for i, v in enumerate(meta) if i not in bs]
    print(f'  数据审计: 去重 {dup} 张 / 污染 {len(bad)} 张 -> '
          f'唯一笔迹 {len(items)} 张 / {len({m["tag"] for m in meta})} 字', flush=True)
    return items, cmap, meta, {'duplicates': dup, 'polluted': len(bad)}


def make_split(items, meta, cmap_inv):
    """按字分层严格留出: 每字最多 1 张, 名单持久化, 跨轮稳定"""
    by = defaultdict(list)
    for i, (_, y) in enumerate(items):
        by[cmap_inv[y]].append(i)
    existing = set()
    if os.path.exists(SPLIT_FILE):
        try:
            with open(SPLIT_FILE, encoding='utf-8') as f:
                existing = set(json.load(f).get('holdout', []))
        except Exception:
            existing = set()
    valid = {m['fn'] for m in meta}
    existing &= valid
    hold = set()
    for ch, idxs in sorted(by.items()):
        cur = sorted(meta[i]['fn'] for i in idxs if meta[i]['fn'] in existing)
        if cur:
            hold.add(cur[0])
        elif len(idxs) >= 2:
            pick = min(idxs, key=lambda i: hashlib.md5(
                ('holdout-v1' + meta[i]['fn']).encode('utf-8')).hexdigest())
            hold.add(meta[pick]['fn'])
    tr = [i for i, m in enumerate(meta) if m['fn'] not in hold]
    va = [i for i, m in enumerate(meta) if m['fn'] in hold]
    info = {'version': 1, 'seed': 'holdout-v1',
            'note': '按字分层严格留出, 每字最多1张, 跨轮稳定, 不参与训练/混淆挖掘/选模之外构造',
            'updated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'holdout': sorted(hold), 'n_train': len(tr), 'n_holdout': len(va),
            'chars_total': len(by), 'chars_with_holdout': len(hold)}
    with open(SPLIT_FILE, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f'  严格分层: train {len(tr)} / val(留出) {len(va)} '
          f'({len(hold)}/{len(by)} 字有留出覆盖)', flush=True)
    return tr, va, info


class HWDB:
    """HWDB npy 分片: 训练分片用于回放, 测试分片拆成 cal / test 两个独立域"""

    def __init__(self):
        self.tr_imgs = [np.load(p, mmap_mode='r') for p in
                        sorted(glob.glob(os.path.join(DATA, 'train_imgs_*.npy')))]
        self.tr_lbls = [np.load(p, mmap_mode='r') for p in
                        sorted(glob.glob(os.path.join(DATA, 'train_lbls_*.npy')))]
        self.tr_sizes = [len(a) for a in self.tr_imgs]
        self.tr_total = sum(self.tr_sizes)
        self.te_imgs = np.load(os.path.join(DATA, 'test_imgs_00.npy'), mmap_mode='r')
        self.te_lbls = np.load(os.path.join(DATA, 'test_lbls_00.npy'))
        if not self.tr_imgs or len(self.te_lbls) == 0:
            raise RuntimeError('HWDB npy 数据缺失, 检查 hanzi_data/')
        perm = np.random.default_rng(CFG['seed']).permutation(len(self.te_lbls))
        self.cal_idx = perm[:min(CFG['hw_cal_n'], len(perm))]
        self.test_idx = perm[len(self.cal_idx):]
        self.full_idx = np.arange(len(self.te_lbls))

    def get_train(self, g):
        for ims, lbs, n in zip(self.tr_imgs, self.tr_lbls, self.tr_sizes):
            if g < n:
                return np.asarray(ims[g], np.uint8), int(lbs[g])
            g -= n
        raise IndexError(g)

    def test_x(self, idxs):
        x = np.stack([np.asarray(self.te_imgs[i], np.uint8) for i in idxs]).astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        return torch.from_numpy(np.repeat(x[:, None], 3, axis=1))


def _normalize64(a):
    ys, xs = np.nonzero(a > 32)
    if len(xs) == 0:
        return None
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = 58.0 / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    r = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((64, 64), np.uint8)
    ox, oy = (64 - nw) // 2, (64 - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = r
    return out


def _is_tofu(a):
    b = (a > 64).astype(np.uint8)
    if b.sum() < 60:
        return True
    edge = b[:5].sum() + b[-5:].sum() + b[:, :5].sum() + b[:, -5:].sum()
    inner = b[12:-12, 12:-12].sum()
    return inner < 0.35 * edge


def _synth_one(ch, font_path, rng):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype(font_path, int(rng.integers(46, 58)))
    except Exception:
        return None
    im = Image.new('L', (64, 64), 0)
    ImageDraw.Draw(im).text((32, 32), ch, font=font, fill=255, anchor='mm')
    a = np.asarray(im, np.uint8)
    if _is_tofu(a):
        return None
    a = _normalize64(a)
    if a is None:
        return None
    if rng.random() < 0.35:
        e = cv2.erode(a, np.ones((2, 2), np.uint8))
        if e.sum() > 150:
            a = e
    ang = rng.uniform(-10, 10)
    sc = rng.uniform(0.9, 1.08)
    M = cv2.getRotationMatrix2D((32, 32), ang, sc)
    M[0, 2] += rng.uniform(-2, 2)
    M[1, 2] += rng.uniform(-2, 2)
    a = cv2.warpAffine(a, M, (64, 64), borderValue=0)
    if rng.random() < 0.3:
        a = cv2.GaussianBlur(a, (3, 3), 0)
    return _normalize64(a)


def synth_ext(chars, ext_cls, per_cls=520, val_per_cls=100):
    """保底合成器: 仅当 ext_*.npy 缺失或未覆盖全部扩展类时调用"""
    rng = np.random.default_rng(CFG['seed'])
    fonts = [os.path.join(FONTDIR, n) for n in FONT_NAMES
             if os.path.exists(os.path.join(FONTDIR, n))]
    if not fonts:
        raise RuntimeError('未找到系统中文字体, 无法合成扩类样本')
    ti, tl, vi, vl = [], [], [], []
    for y in sorted(ext_cls):
        ch = chars[y]
        made, tries = 0, 0
        while made < per_cls + val_per_cls and tries < per_cls * 12:
            tries += 1
            a = _synth_one(ch, fonts[int(rng.integers(0, len(fonts)))], rng)
            if a is None:
                continue
            (vi if made >= per_cls else ti).append(a)
            (vl if made >= per_cls else tl).append(y)
            made += 1
        print(f'  合成 {ch}: {made} 张', flush=True)
    np.save(EXT_TRAIN_I, np.stack(ti).astype(np.uint8))
    np.save(EXT_TRAIN_L, np.asarray(tl, np.int64))
    np.save(EXT_VAL_I, np.stack(vi).astype(np.uint8))
    np.save(EXT_VAL_L, np.asarray(vl, np.int64))


def ensure_ext_data(chars, ext_cls):
    if not ext_cls:
        return [], []
    ok = all(os.path.exists(p) for p in (EXT_TRAIN_I, EXT_TRAIN_L, EXT_VAL_I, EXT_VAL_L))
    if ok:
        lbl = np.load(EXT_TRAIN_L)
        ok = set(ext_cls).issubset(set(np.unique(lbl).tolist()))
    if not ok:
        print('  扩类合成数据缺失/不完整, 启动内置合成器', flush=True)
        synth_ext(chars, ext_cls)
    ti, tl = np.load(EXT_TRAIN_I), np.load(EXT_TRAIN_L)
    vi, vl = np.load(EXT_VAL_I), np.load(EXT_VAL_L)
    print(f'  扩类合成: train {len(tl)} / val {len(vl)}', flush=True)
    return ([(np.asarray(a, np.uint8), int(y)) for a, y in zip(ti, tl)],
            [(np.asarray(a, np.uint8), int(y)) for a, y in zip(vi, vl)])


def load_sign_synth(n_cls):
    """签名风格迁移合成域。

    数据契约(不可破坏):
      * sign_train 只作为额外的训练增强池进 MixedDS, 不写进 user_data,
        不进 load_user / make_split / holdout 分层, 不进拟合域审计。
      * sign_test 是 20 个完全不训练的未见字, 只用于评估「学到的个人书写
        风格能否迁移到没练过的字」, 不参与选模/早停/校准/门槛。
      * 两者都不参与 HWDB cal/test 的划分。
    """
    if not all(os.path.exists(p) for p in
               (SIGN_TRAIN_I, SIGN_TRAIN_L, SIGN_TEST_I, SIGN_TEST_L)):
        print('  签名合成域: 未生成, 跳过 '
              '(运行 hanzi/synth_from_signatures.py 可生成)', flush=True)
        return [], [], {}
    ti, tl = np.load(SIGN_TRAIN_I), np.load(SIGN_TRAIN_L)
    vi, vl = np.load(SIGN_TEST_I), np.load(SIGN_TEST_L)
    tr = [(np.asarray(a, np.uint8), int(y)) for a, y in zip(ti, tl)
          if 0 <= int(y) < n_cls]
    te = [(np.asarray(a, np.uint8), int(y)) for a, y in zip(vi, vl)
          if 0 <= int(y) < n_cls]
    meta = {}
    if os.path.exists(SIGN_META):
        try:
            with open(SIGN_META, encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            pass
    print(f'  签名合成域: train {len(tr)} 张 / {len(set(y for _, y in tr))} 字  '
          f'独立测试 {len(te)} 张 / {len(set(y for _, y in te))} 字 '
          '(独立 npy, 不污染 holdout/拟合审计)', flush=True)
    return tr, te, {'train': len(tr), 'test': len(te),
                    'train_chars': len(set(y for _, y in tr)),
                    'test_chars': len(set(y for _, y in te)),
                    'seed': meta.get('seed'),
                    'style_real': meta.get('style_real'),
                    'style_synth': meta.get('style_synth')}


# =====================================================================
# 3. 模型层: MobileNetV2 + 分类头扩类移植
# =====================================================================
def build_model(n_cls):
    m = mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_cls)
    return m


def from_base(n_cls, ck):
    m = build_model(n_cls)
    src, old = ck['model'], int(ck['num_classes'])
    if old > n_cls:
        raise RuntimeError(f'基座 {old} 类 > 字符表 {n_cls} 类')
    sd = m.state_dict()
    for k, v in src.items():
        if k.startswith('classifier.1.'):
            continue
        if k in sd and tuple(sd[k].shape) == tuple(v.shape):
            sd[k] = v.clone()
    sd['classifier.1.weight'][:old] = src['classifier.1.weight']
    sd['classifier.1.bias'][:old] = src['classifier.1.bias']
    if n_cls > old:
        wm, bm = src['classifier.1.weight'].mean(0), src['classifier.1.bias'].mean()
        for j in range(old, n_cls):
            sd['classifier.1.weight'][j] = wm + 0.02 * torch.randn_like(wm)
            sd['classifier.1.bias'][j] = bm + CFG['new_cls_bias_offset']
        print(f'  分类头扩类: {old} -> {n_cls} (新类 bias 轻微压低)', flush=True)
    m.load_state_dict(sd)
    return m


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


def load_ckpt(path, n_cls, device):
    """已部署模型可能比当前字符表少几类(用户新写了表外字)。
    直接加载会让新类索引越界/永远判错, 这里移植分类头补齐后再当参考模型。"""
    ck = torch.load(path, map_location='cpu')
    old = int(ck['num_classes'])
    if old < n_cls:
        m = from_base(n_cls, ck).to(device).eval()
        print(f'  参考模型扩类: {old} -> {n_cls} '
              '(旧类权重移植, 新类均值初始化, 仅用于范式对照/报告)', flush=True)
    else:
        m = build_model(old)
        m.load_state_dict(ck['model'])
        m = m.to(device).eval()
    return m, ck


# =====================================================================
# 4. 增广层: 鼠标书写域增强 + HWDB 原域/鼠标域混合 + 5 视角 TTA
# =====================================================================
def aug_user(img, rng):
    b = img
    r = rng.random()
    if r < 0.3:
        b = cv2.dilate(b, np.ones((2, 2), np.uint8))
    elif r < 0.5:
        e = cv2.erode(b, np.ones((2, 2), np.uint8))
        if e.sum() > 200:
            b = e
    ang, sc = rng.uniform(-14, 14), rng.uniform(0.85, 1.15)
    tx, ty = rng.uniform(-3, 3, 2)
    sh = rng.uniform(-0.25, 0.25)
    th = np.deg2rad(ang)
    M = np.float32([[np.cos(th) + sh * np.sin(th), np.sin(th), tx],
                    [(-np.sin(th) + sh * np.cos(th)) / sc, np.cos(th) / sc, ty]])
    M[0, 2] += 32 - (M[0, 0] * 32 + M[0, 1] * 32)
    M[1, 2] += 32 - (M[1, 0] * 32 + M[1, 1] * 32)
    b = cv2.warpAffine(b, M, (64, 64), flags=cv2.INTER_LINEAR, borderValue=0)
    if rng.random() < 0.25:
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    if rng.random() < 0.3:
        b = cv2.GaussianBlur(b, (3, 3), 0)
    return b


def aug_hwdb(img, rng):
    if rng.random() < CFG['raw_replay_ratio']:
        return img
    if rng.random() < 0.25:
        b = (img > 96).astype(np.uint8)
        e = cv2.erode(b, np.ones((2, 2), np.uint8))
        if e.sum() > 80:
            img = (e * 255).astype(np.uint8)
        k = int(rng.integers(2, 5))
        s = cv2.resize(img, (64 // k, 64 // k), interpolation=cv2.INTER_AREA)
        img = cv2.resize(s, (64, 64), interpolation=cv2.INTER_CUBIC)
    M = cv2.getRotationMatrix2D((32, 32), rng.uniform(-8, 8), rng.uniform(0.92, 1.06))
    M[0, 2] += rng.uniform(-2, 2)
    M[1, 2] += rng.uniform(-2, 2)
    return cv2.warpAffine(img, M, (64, 64), borderValue=0)


def tta_variants(img):
    out = [img]
    for ang in (-4, 4):
        M = cv2.getRotationMatrix2D((32, 32), ang, 1.0)
        out.append(cv2.warpAffine(img, M, (64, 64), borderValue=0))
    M = np.float32([[1, 0, 1.5], [0, 1, 0]])
    out.append(cv2.warpAffine(img, M, (64, 64), borderValue=0))
    M = np.float32([[1, 0, 0], [0, 1, -1.5]])
    out.append(cv2.warpAffine(img, M, (64, 64), borderValue=0))
    return out


def tta_logits(model, img, device):
    vs = tta_variants(img)
    x = np.stack([np.asarray(v, np.uint8) for v in vs]).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    t = torch.from_numpy(np.repeat(x[:, None], 3, axis=1)).to(device)
    with torch.no_grad():
        return model(t).cpu().numpy()


def softmax_np(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


# =====================================================================
# 5. 训练层: 按类均衡混合数据集 / 混淆挖掘 / EMA / AMP / 余弦调度 / 早停
# =====================================================================
class MixedDS(Dataset):
    def __init__(self, user_items, replay, hwdb, n_random, ext_items,
                 ext_cls, gap_w, sign_items=None):
        self.user, self.replay, self.n_random = user_items, replay, n_random
        self.hwdb, self.ext = hwdb, ext_items
        self.sign = sign_items or []
        self.n_ext = int(len(ext_items) * CFG['ext_repeat'])
        self.n_sign = len(self.sign) * CFG['sign_repeat'] if self.sign else 0
        by = defaultdict(list)
        for it in user_items:
            by[it[1]].append(it)
        self.pool = []
        for y, group in sorted(by.items()):
            gw = gap_w.get(y, 1.0)
            is_ext = y in ext_cls
            boost = CFG['ext_boost'] if is_ext else 1.0
            d = int(round(CFG['per_class_draws'] * boost * gw))
            # 扩类新字真实笔迹极少而合成样本多样, 单独放宽曝光上限(只影响这几个类)
            cap = CFG['max_per_image'] * len(group) * gw * (2.0 if is_ext else 1.0)
            d = min(d, int(cap))
            for k in range(d):
                self.pool.append(group[k % len(group)])
        self.n_user = len(self.pool)
        self.rng = np.random.default_rng(CFG['seed'] + 1234)

    def __len__(self):
        return (self.n_user + self.n_ext + self.n_sign
                + len(self.replay) + self.n_random)

    def __getitem__(self, i):
        r = self.rng
        if i < self.n_user:
            img, y = self.pool[i]
            img = aug_user(img, r)
        elif i < self.n_user + self.n_ext:
            img, y = self.ext[(i - self.n_user) % len(self.ext)]
            img = aug_hwdb(img, r)
        elif i < self.n_user + self.n_ext + self.n_sign:
            # 签名风格合成: 与真实用户笔迹用同样的鼠标书写域增强
            img, y = self.sign[(i - self.n_user - self.n_ext) % len(self.sign)]
            img = aug_user(img, r)
        elif i < self.n_user + self.n_ext + self.n_sign + len(self.replay):
            k = i - self.n_user - self.n_ext - self.n_sign
            img, y = self.hwdb.get_train(self.replay[k])
            img = aug_hwdb(img, r)
        else:
            img, y = self.hwdb.get_train(int(r.integers(0, self.hwdb.tr_total)))
            img = aug_hwdb(img, r)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return torch.from_numpy(np.repeat(img[None], 3, axis=0)), y


class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for p, q in zip(model.parameters(), self.model.parameters()):
            q.lerp_(p, 1.0 - self.decay)
        for b, q in zip(model.buffers(), self.model.buffers()):
            q.copy_(b)


def mine_confusables(model, items, cmap_inv, device):
    conf = defaultdict(set)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(items), 128):
            batch = items[s:s + 128]
            x = np.stack([im for im, _ in batch]).astype(np.float32) / 255.0
            x = (x - 0.5) / 0.5
            t = torch.from_numpy(np.repeat(x[:, None], 3, axis=1)).to(device)
            top = torch.topk(torch.softmax(model(t), 1), 5).indices.cpu().numpy()
            for (_, y), ids in zip(batch, top):
                conf[cmap_inv[y]].update(cmap_inv[int(i)] for i in ids)
    return conf


def build_replay(hwdb, cmap, conf, gap_cls):
    lbls = np.concatenate([np.load(p) for p in
                           sorted(glob.glob(os.path.join(DATA, 'train_lbls_*.npy')))])
    keep = set()
    for ch, cs in conf.items():
        keep.add(cmap[ch])
        keep.update(cmap[c] for c in cs if c in cmap)
    keep |= {cmap[c] for c in EXTRA_CONFUSABLES if c in cmap}
    keep |= (gap_cls or set())
    idx = np.nonzero(np.isin(lbls, list(keep)))[0]
    by = defaultdict(list)
    for i in idx:
        by[int(lbls[i])].append(int(i))
    rng = np.random.default_rng(CFG['seed'])
    fixed = []
    for c, ids in sorted(by.items()):
        if len(ids) > CFG['replay_quota']:
            ids = rng.choice(np.asarray(ids), CFG['replay_quota'], replace=False).tolist()
        fixed.extend(ids)
    random.Random(CFG['seed']).shuffle(fixed)
    print(f'  回放: 固定 {len(fixed)} 条 / {len(by)} 类 (类内配额 {CFG["replay_quota"]}) '
          f'+ 每轮随机重采 {CFG["random_replay"]} 条', flush=True)
    return fixed


def train_model(model, ds, val_items, hwdb, device, base_hw_acc, base_val, cmap_inv):
    """选模与早停只用 val(严格留出) + HWDB cal 子集; 训练集回测只做拟合监控"""
    crit = nn.CrossEntropyLoss(label_smoothing=CFG['label_smoothing'])
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'],
                            weight_decay=CFG['weight_decay'])
    loader = DataLoader(ds, batch_size=CFG['batch'], shuffle=True, num_workers=0,
                        pin_memory=True,
                        generator=torch.Generator().manual_seed(CFG['seed']))
    total = CFG['epochs'] * max(1, len(loader))
    warm = max(1, int(total * CFG['warmup_ratio']))
    minr = CFG['min_lr_ratio']

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        p = min(1.0, (step - warm) / max(1, total - warm))
        return minr + (1.0 - minr) * 0.5 * (1.0 + math.cos(math.pi * p))

    # 手动按 step 设置 LR(避免 AMP 首 iter 跳过 optimizer.step 时 scheduler 顺序警告)
    def lr_at(step):
        return CFG['lr'] * lr_lambda(step)

    scaler = torch.amp.GradScaler(device)
    ema = EMA(model, CFG['ema_decay'])
    monitor = val_items[:min(256, len(val_items))]
    best, hist, no_imp = {'score': -1e9}, [], 0
    for ep in range(CFG['epochs']):
        model.train()
        if CFG['freeze_bn']:
            freeze_bn(model)
        t0, run, nb = time.time(), 0.0, 0
        for bi, (x, y) in enumerate(loader):
            lr_now = lr_at(ep * len(loader) + bi)
            for g in opt.param_groups:
                g['lr'] = lr_now
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            run += loss.item()
            nb += 1
            if bi % 120 == 0:
                print(f'    ep{ep+1} it{bi}/{len(loader)} loss={loss.item():.3f} '
                      f'lr={lr_now:.2e}', flush=True)
        cands = {'raw': model, 'ema': ema.model}
        ep_best = None
        for name, m in cands.items():
            v = eval_user(m, val_items, device)
            h = eval_hwdb(m, hwdb, hwdb.cal_idx, device)
            floor = base_val['hwdb_cal'] * 100
            # 校准(ECE / 过置信)可由训练后 Temperature Scaling 修复, 选模阶段
            # 不做硬罚。历史教训: 50 分硬罚 + cal 子集下限会把 ep7~ep10 这些
            # 充分训练的 epoch 打到个位数, 反而选中欠拟合的早期 epoch。
            # 最终仍有 check_gates 的「置信校准不崩坏」和「通用域 guard」兜底。
            overconf = h['conf'] - h['micro']
            raw = (v['top1'] * 100 + v['top3'] * 15 + v['cover50'] * 10
                   + (h['micro'] * 100 - base_val['hwdb_cal'] * 100) * 3.0
                   - v['ece'] * 10 - h['ece'] * 10)
            # floor 是基座在 cal 子集上的精度, 与门槛用的独立 test 集不同口径,
            # 容差放宽到 2*guard; 只有真正的通用域崩坏才触发硬罚。
            pen = 100.0 if h['micro'] * 100 < floor - CFG['guard'] * 2 else 0.0
            pen += min(20.0, max(0.0, (overconf - 0.05) * 200.0))
            sc = raw - pen
            if ep_best is None or sc > ep_best['score']:
                ep_best = {'name': name, 'score': sc, 'val': v, 'hw': h}
        fit = eval_user(model, monitor, device)
        print(f'    ep{ep+1}: loss={run/max(1,nb):.3f} '
              f'[拟合监控]train子集 Top-1={fit["top1"]*100:.1f}% '
              f'| [val] Top-1={ep_best["val"]["top1"]*100:.2f}% '
              f'Top-3={ep_best["val"]["top3"]*100:.2f}% '
              f'ECE={ep_best["val"]["ece"]*100:.2f}% '
              f'| [HWDB cal]={ep_best["hw"]["micro"]*100:.2f}% '
              f'ECE={ep_best["hw"]["ece"]*100:.2f}% '
              f'| score={ep_best["score"]:.2f} ({ep_best["name"]}) '
              f'({time.time()-t0:.0f}s)', flush=True)
        hist.append({'epoch': ep + 1, 'loss': run / max(1, nb),
                     'fit_train_subset_top1': fit['top1'],
                     'val_top1': ep_best['val']['top1'],
                     'hwdb_cal_top1': ep_best['hw']['micro'],
                     'score': ep_best['score'], 'source': ep_best['name']})
        if ep_best['score'] > best['score'] + 0.05:
            src = cands[ep_best['name']]
            best = dict(ep_best, epoch=ep + 1,
                        model_state={k: v.detach().cpu().clone()
                                     for k, v in src.state_dict().items()})
            no_imp = 0
        else:
            no_imp += 1
        if ep_best['hw']['micro'] * 100 < floor - CFG['guard'] * 4 and ep >= 2:
            print('    通用域硬退化, 提前停止', flush=True)
            break
        if no_imp >= CFG['patience'] and ep >= 2:
            print('    val 泛化分连续未提升, 早停', flush=True)
            break
    return best, hist


# =====================================================================
# 6. 评估层: ECE / 分域指标 / TTA 鲁棒性 / 范式对照 / 温度校准
# =====================================================================
def ece_of(conf, ok, bins=10):
    conf, ok = np.asarray(conf, np.float64), np.asarray(ok, np.float64)
    if len(conf) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            e += m.sum() / len(conf) * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def eval_user(model, items, device):
    """任意用户分片: TTA 5 视角平均概率, 指标口径与画板一致"""
    if not items:
        return {'n': 0, 'top1': 0.0, 'top3': 0.0, 'cover50': 0.0,
                'macro': 0.0, 'ece': 0.0, 'conf': 0.0, 'wrong': [], 'low': []}
    confs, oks, per = [], [], defaultdict(lambda: [0, 0])
    h1 = h3 = cov = 0
    wrong, low = [], []
    model.eval()
    with torch.no_grad():
        for im, y in items:
            p = softmax_np(tta_logits(model, im, device)).mean(0)
            top3 = np.argsort(-p)[:3].tolist()
            hit = int(top3[0] == y)
            c = float(p.max())
            confs.append(c)
            oks.append(hit)
            h1 += hit
            h3 += int(y in top3)
            cov += int(hit and c >= 0.5)
            per[y][0] += hit
            per[y][1] += 1
            if not hit:
                wrong.append(int(y))
            elif c < 0.5:
                low.append(int(y))
    n = len(items)
    return {'n': n, 'top1': h1 / n, 'top3': h3 / n, 'cover50': cov / n,
            'macro': float(np.mean([v[0] / v[1] for v in per.values()])),
            'ece': ece_of(confs, oks), 'conf': float(np.mean(confs)),
            'wrong': wrong, 'low': low}


def eval_hwdb(model, hwdb, idxs, device, batch=256):
    per = defaultdict(lambda: [0, 0])
    confs, oks = [], []
    top3 = hijack = 0
    model.eval()
    with torch.no_grad():
        for s in range(0, len(idxs), batch):
            bi = idxs[s:s + batch]
            t = hwdb.test_x(bi).to(device)
            p = torch.softmax(model(t), 1)
            c, pred = p.max(1)
            c, pred = c.cpu().numpy(), pred.cpu().numpy()
            y = hwdb.te_lbls[bi]
            ok = (pred == y)
            for yy, o in zip(y, ok):
                per[int(yy)][0] += int(o)
                per[int(yy)][1] += 1
            confs.append(c)
            oks.append(ok.astype(np.float32))
            hijack += int((pred >= CFG['base_n']).sum())
            top3 += int((torch.topk(p, 3).indices.cpu().numpy() == y[:, None]).any(1).sum())
    confs, oks = np.concatenate(confs), np.concatenate(oks)
    n = len(idxs)
    worst = sorted(((k, v[1], v[0] / v[1]) for k, v in per.items()), key=lambda t: t[2])[:8]
    return {'n': n, 'micro': float(oks.mean()), 'top3': top3 / n,
            'macro': float(np.mean([v[0] / v[1] for v in per.values()])),
            'ece': ece_of(confs, oks), 'conf': float(confs.mean()),
            'cover50': float(((confs >= 0.5) & (oks == 1)).mean()),
            'hijack': hijack / n, 'worst': worst}


def eval_ext_val(model, ext_val, device):
    if not ext_val:
        return None
    confs, oks = [], []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(ext_val), 256):
            b = ext_val[s:s + 256]
            x = np.stack([im for im, _ in b]).astype(np.float32) / 255.0
            x = (x - 0.5) / 0.5
            t = torch.from_numpy(np.repeat(x[:, None], 3, axis=1)).to(device)
            p = torch.softmax(model(t), 1)
            c, pred = p.max(1)
            confs.append(c.cpu().numpy())
            oks.append((pred.cpu().numpy() == np.array([y for _, y in b])).astype(np.float32))
    confs, oks = np.concatenate(confs), np.concatenate(oks)
    return {'n': len(ext_val), 'top1': float(oks.mean()), 'ece': ece_of(confs, oks)}


def tta_flip(model, hwdb, device, n):
    rng = np.random.default_rng(CFG['seed'])
    pick = rng.choice(len(hwdb.te_lbls), min(n, len(hwdb.te_lbls)), replace=False)
    same = tot = 0
    model.eval()
    with torch.no_grad():
        for i in pick:
            p = [torch.softmax(model(
                torch.from_numpy(np.repeat(
                    ((np.asarray(v, np.uint8).astype(np.float32) / 255.0 - 0.5) / 0.5)
                    [None, None], 3, axis=1)).to(device)), 1)[0] for v in
                tta_variants(np.asarray(hwdb.te_imgs[i], np.uint8))]
            avg = int(torch.stack(p).mean(0).argmax())
            preds = [int(q.argmax()) for q in p]
            same += sum(int(x == avg) for x in preds)
            tot += len(preds)
    return 1.0 - same / max(1, tot)


def classical_feat(img):
    """无学习结构化特征: 8x8 网格 + 行/列投影 + 4x4x8 方向梯度直方图"""
    def norm(v):
        return v / (float(np.linalg.norm(v)) + 1e-6)
    a = np.asarray(img, np.float32) / 255.0
    if a.shape != (64, 64):
        a = cv2.resize(a, (64, 64))
    grid = cv2.resize(a, (8, 8), interpolation=cv2.INTER_AREA).ravel()
    rp = cv2.resize(a.sum(1, keepdims=True), (1, 16), interpolation=cv2.INTER_AREA).ravel()
    cp = cv2.resize(a.sum(0, keepdims=True), (16, 1), interpolation=cv2.INTER_AREA).ravel()
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hog = np.zeros((4, 4, 8), np.float32)
    for r in range(4):
        for c in range(4):
            m = mag[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16].ravel()
            bb = np.nan_to_num(ang[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16] / 45.0)
            b = np.clip(bb.astype(np.int64), 0, 7).ravel()
            np.add.at(hog[r, c], b, m)
    return norm(np.concatenate([norm(grid) * 1.5, norm(rp) * 0.7,
                                norm(cp) * 0.7, norm(hog.ravel()) * 1.2]))


def classical_ranks(train_items, queries, k=3):
    F = np.stack([classical_feat(im) for im, _ in train_items])
    Y = np.array([y for _, y in train_items], np.int64)
    out = []
    for im, _ in queries:
        sim = F @ classical_feat(im)
        sc = defaultdict(float)
        for i in np.argsort(-sim)[:k]:
            sc[int(Y[i])] += max(0.0, float(sim[i]))
        out.append([c for c, _ in sorted(sc.items(), key=lambda t: -t[1])])
    return out


def paradigm_gap(ref_model, chars, cmap, train_items, val_items, ext_cls, device):
    """经典特征 vs CNN 对照 -> 深度盲区; 合并累积难例记忆 -> 按类采样权重"""
    closed = sorted({y for _, y in train_items} | {y for _, y in val_items})
    n_cls = int(ref_model.classifier[1].out_features)
    ci = [c for c in closed if c < n_cls]
    cl = classical_ranks(train_items, val_items)
    deep, tr_conf = [], []
    ref_model.eval()
    with torch.no_grad():
        for im, _ in val_items:
            p = softmax_np(tta_logits(ref_model, im, device)).mean(0)
            deep.append([ci[j] for j in np.argsort(-p[ci])[:3]])
        for im, _ in train_items:
            p = softmax_np(tta_logits(ref_model, im, device)).mean(0)
            tr_conf.append(float(p.max()))
    cl_ok = [y in r[:1] for r, (_, y) in zip(cl, val_items)]
    dp_ok = [y in r[:1] for r, (_, y) in zip(deep, val_items)]
    blind = sorted({chars[y] for (_, y), c, d in zip(val_items, cl_ok, dp_ok) if c and not d})
    hard = sorted({chars[y] for (_, y), c, d in zip(val_items, cl_ok, dp_ok) if not c and not d})
    low = sorted({chars[y] for (_, y), c in zip(train_items, tr_conf) if c < 0.5})
    ext_tags = sorted({chars[y] for y in ext_cls if y < len(chars)})
    hist = {}
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, encoding='utf-8') as f:
                hist = json.load(f).get('tags', {})
        except Exception:
            hist = {}
    w = {}
    for t in blind:
        w[t] = 2.0
    for t in hard:
        w[t] = max(w.get(t, 1.0), 1.6)
    for t in low:
        w[t] = max(w.get(t, 1.0), 1.5)
    for t in ext_tags:
        w[t] = max(w.get(t, 1.0), 2.0)
    for t, v in hist.items():
        if t in cmap:
            w[t] = max(w.get(t, 1.0), min(float(v), 2.0))
    summary = {
        'reference': os.path.basename(MAIN_CKPT), 'val_n': len(val_items),
        'classical_top1': float(np.mean(cl_ok)),
        'classical_top3': float(np.mean([y in r[:3] for r, (_, y) in zip(cl, val_items)])),
        'deep_closed_top1': float(np.mean(dp_ok)),
        'deep_closed_top3': float(np.mean([y in r[:3] for r, (_, y) in zip(deep, val_items)])),
        'complement': {
            'both_right': int(sum(c and d for c, d in zip(cl_ok, dp_ok))),
            'deep_only': int(sum((not c) and d for c, d in zip(cl_ok, dp_ok))),
            'deep_blind': int(sum(c and (not d) for c, d in zip(cl_ok, dp_ok))),
            'both_wrong': int(sum((not c) and (not d) for c, d in zip(cl_ok, dp_ok))),
        },
        'blind_tags': blind, 'hard_tags': hard, 'low_conf_tags': low,
        'ext_tags': ext_tags, 'history_tags': sorted(hist),
        'weighted_chars': len(w),
        'note': '旧协议模型训练时见过这批样本, 其高准确率是记忆证据; '
                '本文件 val 由 holdout_split.json 严格隔离',
    }
    if WRITE_STATE:
        with open(GAP_FILE, 'w', encoding='utf-8') as f:
            json.dump({'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                       'weights': w, **summary}, f, ensure_ascii=False, indent=2)
    gap_w = {cmap[t]: v for t, v in w.items() if t in cmap}
    gap_cls = set(gap_w)
    print(f'  范式对照: 经典 {summary["classical_top1"]*100:.2f}% / '
          f'深度(闭集) {summary["deep_closed_top1"]*100:.2f}% | '
          f'深度盲区 {len(blind)} 字 / 共同难例 {len(hard)} 字 / '
          f'历史记忆 {len(hist)} 字 -> 加权 {len(gap_w)} 类', flush=True)
    return gap_w, gap_cls, summary


def fit_temperature(model, hwdb, train_items, val_items, device):
    """标准 Temperature Scaling: 校准集=HWDB cal(独立) + val(选模集), 不用 train 拟合指标选 T
       硬约束: 拟合域零 <50% 低置信(产品信任度线); 在满足约束的 T 中选 cal ECE 最小者。
       1/T 吸收进 classifier 权重, 推理端零改动。"""
    cal_lg = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(hwdb.cal_idx), 256):
            bi = hwdb.cal_idx[s:s + 256]
            cal_lg.append(model(hwdb.test_x(bi).to(device)).cpu().numpy())
    cal_lg = np.concatenate(cal_lg)
    cal_y = hwdb.te_lbls[hwdb.cal_idx]
    tr_lg = np.stack([tta_logits(model, im, device) for im, _ in train_items])
    va_lg = np.stack([tta_logits(model, im, device) for im, _ in val_items])
    tr_y = np.array([y for _, y in train_items])
    va_y = np.array([y for _, y in val_items])
    rows = []
    for T in np.arange(0.40, 1.001, 0.05):
        tp = softmax_np(tr_lg / T).mean(1)
        vp = softmax_np(va_lg / T).mean(1)
        hp = softmax_np(cal_lg / T)
        low = int(((tp.argmax(1) == tr_y) &
                   (tp[np.arange(len(tr_y)), tr_y] < 0.5)).sum())
        cov = float(((vp.argmax(1) == va_y) & (vp.max(1) >= 0.5)).mean())
        hc = hp.max(1)
        hok = (hp.argmax(1) == cal_y).astype(np.float64)
        rows.append({'T': round(float(T), 2), 'train_lowconf': low,
                     'cal_ece': ece_of(hc, hok), 'val_cover50': cov,
                     'cal_conf': float(hc.mean()), 'cal_acc': float(hok.mean())})
    elig = [r for r in rows if r['train_lowconf'] == 0]
    chosen = min(elig or rows, key=lambda r: (r['cal_ece'], -r['val_cover50']))
    return chosen, rows


def apply_temperature(state, T):
    state['model']['classifier.1.weight'] = state['model']['classifier.1.weight'] / T
    state['model']['classifier.1.bias'] = state['model']['classifier.1.bias'] / T
    state['temperature'] = float(T)
    state['calibration'] = {'method': 'temperature scaling absorbed into classifier',
                            'T': float(T),
                            'note': 'softmax(logits/T) == (W/T)h + (b/T), 推理端无需改动'}
    return state


def update_history(bad_tags):
    if not WRITE_STATE:
        print(f'  (自检模式, 不写难例记忆; 本轮边界字 {len(bad_tags)} 个)', flush=True)
        return 0
    if not bad_tags:
        return 0
    hist = {}
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, encoding='utf-8') as f:
                hist = json.load(f).get('tags', {})
        except Exception:
            hist = {}
    for t in bad_tags:
        hist[t] = max(float(hist.get(t, 1.0)), 2.0)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump({'note': '累积式难例记忆: 拟合域错误/低置信字永久保留, 下轮自动加权',
                   'updated': time.strftime('%Y-%m-%d'), 'tags': hist},
                  f, ensure_ascii=False, indent=2)
    print(f'  难例记忆: 新增 {len(bad_tags)} 字 -> 累积 {len(hist)} 字', flush=True)
    return len(hist)


# =====================================================================
# 7. 门槛 / 部署 / 报告
# =====================================================================
def check_gates(fit, val, hw, base_val, base_test_acc):
    """通用域基准必须是基座在同一个独立 test 集上的精度, 不能混用 cal/ckpt 记录"""
    g = [
        ('通用域 guard', hw['micro'] * 100 >= base_test_acc - CFG['guard'],
         f'{hw["micro"]*100:.2f}% >= {base_test_acc-CFG["guard"]:.2f}% '
         f'(基准=基座同 test 集 {base_test_acc:.2f}%)'),
        ('通用域不退化', hw['micro'] * 100 - base_test_acc >= -CFG['guard'],
         f'{hw["micro"]*100-base_test_acc:+.2f}pp'),
        ('置信校准不崩坏', hw['ece'] <= base_val['hwdb_ece'] + 0.03,
         f'ECE {hw["ece"]*100:.2f}% vs 基座 {base_val["hwdb_ece"]*100:.2f}%'),
        ('用户域不劣于基座', val['top1'] >= base_val['top1'] - 0.01,
         f'{val["top1"]*100:.2f}% vs {base_val["top1"]*100:.2f}%'),
        ('用户域绝对下限', val['top1'] >= CFG['min_holdout'],
         f'{val["top1"]*100:.2f}% >= {CFG["min_holdout"]*100:.0f}%'),
        ('拟合域零错误', len(fit['wrong']) == 0, f'{len(fit["wrong"])} 条'),
        ('拟合域零低置信', len(fit['low']) == 0, f'{len(fit["low"])} 条 <50%'),
    ]
    print('\n' + '=' * 74 + '\n泛化与质量门槛 (全部通过才部署)\n' + '=' * 74, flush=True)
    for name, ok, detail in g:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}: {detail}', flush=True)
    return g, all(ok for _, ok, _ in g)


def deploy(state):
    os.makedirs(HANZI, exist_ok=True)
    torch.save(state, FT_CKPT)
    if os.path.exists(MAIN_CKPT):
        shutil.copy2(MAIN_CKPT, PREV_CKPT)
    shutil.copy2(FT_CKPT, MAIN_CKPT)
    print(f'\n已部署 {os.path.basename(FT_CKPT)} -> {os.path.basename(MAIN_CKPT)} '
          f'(旧版备份 {os.path.basename(PREV_CKPT)})', flush=True)


def auto_report(model, chars, hwdb, ext_val, train_items, val_items, device,
                sign_test=None):
    """内建自动分层评估: 拟合域 / 用户域泛化 / 通用域独立测试 / 合成域 / 鲁棒性"""
    fit = eval_user(model, train_items, device)
    val = eval_user(model, val_items, device)
    hw = eval_hwdb(model, hwdb, hwdb.test_idx, device)
    full = eval_hwdb(model, hwdb, hwdb.full_idx, device)
    ext = eval_ext_val(model, ext_val, device)
    flip = tta_flip(model, hwdb, device, CFG['tta_flip_n'])
    # 独立未见字合成测试: 这些字从未进过训练, 衡量风格迁移而非记忆
    sg = eval_user(model, sign_test, device) if sign_test else None
    return {'fit': fit, 'val': val, 'hw': hw, 'hw_full': full,
            'ext': ext, 'flip': flip, 'sign_test': sg}


def print_report(rep, chars, split, paradigm, calibration):
    f, v, h, hf = rep['fit'], rep['val'], rep['hw'], rep['hw_full']
    print('\n' + '=' * 74 + '\n自动分层泛化报告 (无需人工盲测)\n' + '=' * 74, flush=True)
    print(f'[拟合域 · 教过的样本]  n={f["n"]}  Top-1={f["top1"]*100:.2f}%  '
          f'错误={len(f["wrong"])}  低置信(<50%)={len(f["low"])}', flush=True)
    print(f'[泛化域 · 用户严格留出 val]  n={v["n"]}  Top-1={v["top1"]*100:.2f}%  '
          f'Top-3={v["top3"]*100:.2f}%  macro={v["macro"]*100:.2f}%  '
          f'≥50%覆盖={v["cover50"]*100:.2f}%  ECE={v["ece"]*100:.2f}%', flush=True)
    print(f'[泛化域 · HWDB 独立测试 test]  n={h["n"]}  Top-1={h["micro"]*100:.2f}%  '
          f'Top-3={h["top3"]*100:.2f}%  macro={h["macro"]*100:.2f}%', flush=True)
    print(f'    ECE={h["ece"]*100:.2f}%  平均置信={h["conf"]*100:.1f}%  '
          f'≥50%覆盖={h["cover50"]*100:.2f}%  新类抢答率={h["hijack"]*100:.3f}%', flush=True)
    print(f'    全量历史可比口径 n={hf["n"]} Top-1={hf["micro"]*100:.2f}% '
          f'(含温度校准子集, 与历史基线可比)', flush=True)
    print('    最差类(单类样本少, 仅作定位线索): ' +
          ' '.join(f'{chars[k]}:{a*100:.0f}%({n})' for k, n, a in h['worst']),
          flush=True)
    if rep['ext']:
        print(f'[弱泛化 · 扩类合成留出]  n={rep["ext"]["n"]}  '
              f'Top-1={rep["ext"]["top1"]*100:.2f}%  ECE={rep["ext"]["ece"]*100:.2f}%',
              flush=True)
    if rep.get('sign_test'):
        s = rep['sign_test']
        print(f'[风格迁移 · 未见字合成测试(不训练)]  n={s["n"]}  '
              f'Top-1={s["top1"]*100:.2f}%  Top-3={s["top3"]*100:.2f}%  '
              f'≥50%覆盖={s["cover50"]*100:.2f}%  ECE={s["ece"]*100:.2f}%',
              flush=True)
    print(f'[鲁棒性 · TTA 5视角]  翻转率={rep["flip"]*100:.2f}% '
          f'(一致率 {(1-rep["flip"])*100:.2f}%)', flush=True)
    print(f'[范式对照 · 同一 val]  经典结构特征 3NN={paradigm["classical_top1"]*100:.2f}%  '
          f'深度(闭集)={paradigm["deep_closed_top1"]*100:.2f}%  '
          f'深度盲区={paradigm["complement"]["deep_blind"]} 条', flush=True)
    print(f'[校准]  T={calibration["T"]:.2f}  1/T 已吸收进分类头', flush=True)
    print(f'[分层]  train={split["n_train"]}  val={split["n_holdout"]}  '
          f'cal={CFG["hw_cal_n"]}  test={h["n"]}', flush=True)
    print('=' * 74, flush=True)


# =====================================================================
# 8. 主流程: 一键完成 (含难例记忆驱动的自动重试)
# =====================================================================
def main():
    global WRITE_STATE
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='2 epoch 自检, 不部署')
    ap.add_argument('--no-deploy', action='store_true')
    ap.add_argument('--report-only', action='store_true', help='只评估当前已部署模型')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--retries', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()
    if args.seed is not None:
        CFG['seed'] = args.seed
    if args.epochs is not None:
        CFG['epochs'] = args.epochs
    if args.lr is not None:
        CFG['lr'] = args.lr
    if args.retries is not None:
        CFG['max_retries'] = args.retries
    if args.smoke:
        CFG.update(epochs=2, hw_cal_n=1024, tta_flip_n=64,
                   max_retries=0, random_replay=6000)
        WRITE_STATE = False
    if args.report_only:
        WRITE_STATE = False

    rep_path = REPORT_FILE if WRITE_STATE else REPORT_FILE.replace('.json', '_dryrun.json')
    set_seed(CFG['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 74, flush=True)
    print('一键优化训练 · gen-v3 (自包含单文件)', flush=True)
    print(f'device={device} | torch={torch.__version__} | seed={CFG["seed"]} | '
          f'epochs={CFG["epochs"]} | smoke={args.smoke}', flush=True)
    print('=' * 74, flush=True)

    chars, ext_cls = ensure_charset()
    n_cls = len(chars)
    cmap = {c: i for i, c in enumerate(chars)}
    cmap_inv = {i: c for c, i in cmap.items()}
    items, _, meta, audit = load_user(chars)
    tr_idx, va_idx, split = make_split(items, meta, cmap_inv)
    train_items = [items[i] for i in tr_idx]
    val_items = [items[i] for i in va_idx]
    hwdb = HWDB()
    if args.smoke:
        hwdb.test_idx = hwdb.test_idx[:4096]
        hwdb.full_idx = hwdb.test_idx
    ext_train, ext_val = ensure_ext_data(chars, ext_cls)
    sign_train, sign_test, sign_info = load_sign_synth(n_cls)
    print(f'  独立分层: cal={len(hwdb.cal_idx)} test={len(hwdb.test_idx)} '
          f'(HWDB 测试集按 seed 固定切分, test 不参与选模/校准)', flush=True)

    # ---- report-only: 只看当前模型的自动泛化报告 ----
    if args.report_only:
        if not os.path.exists(MAIN_CKPT):
            raise RuntimeError('主模型不存在')
        model, ck = load_ckpt(MAIN_CKPT, n_cls, device)
        rep = auto_report(model, chars, hwdb, ext_val, train_items,
                          val_items, device, sign_test)
        para = {'classical_top1': 0.0, 'deep_closed_top1': 0.0,
                'complement': {'deep_blind': 0}, 'reference': 'report-only'}
        print('=' * 74 + '\n当前已部署模型的自动分层报告\n' + '=' * 74, flush=True)
        print(f'[拟合域] n={rep["fit"]["n"]} Top-1={rep["fit"]["top1"]*100:.2f}% '
              f'错误={len(rep["fit"]["wrong"])} 低置信={len(rep["fit"]["low"])}', flush=True)
        print(f'[用户域 val] n={rep["val"]["n"]} Top-1={rep["val"]["top1"]*100:.2f}% '
              f'Top-3={rep["val"]["top3"]*100:.2f}% '
              f'≥50%覆盖={rep["val"]["cover50"]*100:.2f}%', flush=True)
        print(f'[通用域 test] n={rep["hw"]["n"]} Top-1={rep["hw"]["micro"]*100:.2f}% '
              f'Top-3={rep["hw"]["top3"]*100:.2f}% ECE={rep["hw"]["ece"]*100:.2f}% '
              f'新类抢答={rep["hw"]["hijack"]*100:.3f}%', flush=True)
        print(f'[鲁棒性] TTA 翻转率={rep["flip"]*100:.2f}%', flush=True)
        if rep.get('sign_test'):
            s = rep['sign_test']
            print(f'[风格迁移 · 未见字合成测试] n={s["n"]} '
                  f'Top-1={s["top1"]*100:.2f}% Top-3={s["top3"]*100:.2f}% '
                  f'≥50%覆盖={s["cover50"]*100:.2f}%', flush=True)
        print('注: 拟合/泛化按当前 holdout_split.json 划分; 对历史部署模型, '
              'train 分片可能包含其训练结束后新增的样本。', flush=True)
        with open(rep_path, 'w', encoding='utf-8') as fp:
            json.dump({'mode': 'report-only', 'env': env_info(device),
                       'n_cls': n_cls, 'split': split, 'domains': rep},
                      fp, ensure_ascii=False, indent=2, default=float)
        return 0

    ck = torch.load(BASE_CKPT, map_location='cpu')
    base_acc = float(ck.get('acc', 0) * 100)
    base = from_base(n_cls, ck).to(device).eval()
    base_val = eval_user(base, val_items, device)
    base_hw = eval_hwdb(base, hwdb, hwdb.cal_idx, device)
    base_test = eval_hwdb(base, hwdb, hwdb.test_idx, device)
    base_test_acc = base_test['micro'] * 100
    base_val_info = {'top1': base_val['top1'],
                     'hwdb_cal': base_hw['micro'], 'hwdb_ece': base_hw['ece']}
    print(f'  基座基线: val Top-1={base_val["top1"]*100:.2f}% | '
          f'HWDB cal={base_hw["micro"]*100:.2f}% ECE={base_hw["ece"]*100:.2f}% | '
          f'HWDB test(门槛基准)={base_test_acc:.2f}% | ckpt记录(仅参考)={base_acc:.2f}%',
          flush=True)
    del base

    decision, report = 'not_deployed', {}
    for attempt in range(CFG['max_retries'] + 1):
        print(f'\n--- 训练轮次 {attempt+1}/{CFG["max_retries"]+1} ---', flush=True)
        set_seed(CFG['seed'] + attempt)
        model = from_base(n_cls, ck).to(device)
        if CFG['freeze_bn']:
            freeze_bn(model)

        # 范式对照(用上一代已部署模型) -> 难例权重
        if os.path.exists(MAIN_CKPT):
            ref, _ = load_ckpt(MAIN_CKPT, n_cls, device)
        else:
            ref = from_base(n_cls, ck).to(device).eval()
        gap_w, gap_cls, para = paradigm_gap(
            ref, chars, cmap, train_items, val_items, ext_cls, device)
        del ref

        conf = mine_confusables(model, train_items, cmap_inv, device)
        replay = build_replay(hwdb, cmap, conf, gap_cls)
        ds = MixedDS(train_items, replay, hwdb, CFG['random_replay'],
                     ext_train, ext_cls, gap_w, sign_train)
        print(f'  混合训练集: {len(ds)} (用户按类均衡 {ds.n_user} / 扩类合成 {ds.n_ext} '
              f'/ 签名风格合成 {ds.n_sign} / 固定回放 {len(replay)} '
              f'/ 随机重采 {CFG["random_replay"]})', flush=True)

        best, hist = train_model(model, ds, val_items, hwdb, device,
                                 base_acc, base_val_info, cmap_inv)
        state = {'model': best['model_state'], 'num_classes': n_cls,
                 'epoch': f'gen3_ep{best["epoch"]}_{best["name"]}',
                 'protocol': 'gen-v3-onefile', 'acc': best['hw']['micro'],
                 'holdout_top1': best['val']['top1'], 'seed': CFG['seed'],
                 'ext_chars': sorted(ext_cls)}
        cal_model = build_model(n_cls).to(device)
        cal_model.load_state_dict(state['model'])
        cal_model.eval()
        chosen, grid = fit_temperature(cal_model, hwdb, train_items, val_items, device)
        print(f'  温度校准: T={chosen["T"]:.2f} (硬约束 拟合域零<50%低置信) '
              f'| cal ECE={chosen["cal_ece"]*100:.2f}% '
              f'| val ≥50%覆盖={chosen["val_cover50"]*100:.2f}%', flush=True)
        state = apply_temperature(state, chosen['T'])
        cal_model.load_state_dict(state['model'])

        rep = auto_report(cal_model, chars, hwdb, ext_val,
                          train_items, val_items, device, sign_test)
        gates, all_ok = check_gates(rep['fit'], rep['val'], rep['hw'],
                                    base_val_info, base_test_acc)
        print_report(rep, chars, split, para, {'T': chosen['T']})

        bad = sorted({cmap_inv[y] for y in rep['fit']['wrong'] + rep['fit']['low']})
        if all_ok:
            decision = 'deployed'
            if not (args.smoke or args.no_deploy):
                deploy(state)
            else:
                print('--smoke/--no-deploy: 跳过部署', flush=True)
        else:
            update_history(bad)
            print('门槛未通过: ' + ('难例已写入累积记忆'
                                  if WRITE_STATE else '自检模式, 未写入难例记忆'),
                  flush=True)
        report = {
            'protocol': 'gen-v3-onefile',
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'decision': decision,
            'attempt': attempt + 1,
            'env': env_info(device), 'config': CFG,
            'data': {'n_cls': n_cls, 'ext_chars': sorted(ext_cls),
                     'unique_user_samples': len(items), 'audit': audit,
                     'train_files': split['n_train'], 'val_files': split['n_holdout'],
                     'hwdb_train_total': hwdb.tr_total,
                     'hwdb_cal': len(hwdb.cal_idx), 'hwdb_test': len(hwdb.test_idx),
                     'ext_train': len(ext_train), 'ext_val': len(ext_val),
                     'sign_synth': sign_info},
            'split_note': 'train/val/cal/test 严格分离; val 选模与早停, cal 校准, test 独立报告',
            'base': {'ckpt_acc_reference_only': base_acc, 'val_top1': base_val['top1'],
                     'hwdb_cal_top1': base_hw['micro'], 'hwdb_cal_ece': base_hw['ece'],
                     'hwdb_test_top1': base_test['micro'],
                     'note': '门槛基准用基座在独立 test 集上的精度, 与最终评估同口径'},
            'best_epoch': best['epoch'], 'best_source': best['name'],
            'best_score': best['score'], 'epochs': hist,
            'calibration': {'chosen': chosen, 'grid': grid},
            'paradigm': para,
            'domains': rep,
            'gates': [{'name': n, 'pass': bool(o), 'detail': d} for n, o, d in gates],
            'all_gates_pass': all_ok,
            'bad_tags': bad,
        }
        with open(rep_path, 'w', encoding='utf-8') as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2, default=float)
        print(f'报告 -> {os.path.basename(rep_path)}', flush=True)
        if all_ok:
            break
        if attempt < CFG['max_retries']:
            print('检测到边界难例, 用累积记忆重训一次(不放宽门槛)', flush=True)
        del model, cal_model, ds
        if device == 'cuda':
            torch.cuda.empty_cache()

    print(f'\n最终决策: {decision.upper()} | '
          f'val Top-1={report["domains"]["val"]["top1"]*100:.2f}% | '
          f'HWDB test Top-1={report["domains"]["hw"]["micro"]*100:.2f}% | '
          f'ECE={report["domains"]["hw"]["ece"]*100:.2f}%', flush=True)
    return 0 if decision == 'deployed' else 1


if __name__ == '__main__':
    sys.exit(main())
