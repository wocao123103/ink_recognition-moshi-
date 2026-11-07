# -*- coding: utf-8 -*-
"""
手写数字识别画板 (PySide6 GUI)
================================
用法: D:/Anaconda/envs/pytorch/python.exe 手写数字画板.py

左侧画布用鼠标写一个数字, 点"识别"输出结果与置信度;
右侧小图显示模型实际看到的 28x28 输入(便于排查笔画太细/太偏等问题)。
"""
import sys
import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import (QAction, QColor, QImage, QPainter, QPen, QPixmap,
                           QFont)
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QProgressBar, QVBoxLayout, QWidget)

MODEL_PATH = "mnist_cnn.pth"
MEAN, STD = 0.1307, 0.3081
CANVAS_SIZE = 400
PEN_WIDTH = 18


# ---------------- 模型(与 notebook MnistCNN 完全一致) ----------------
class MnistCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def preprocess(canvas_img):
    """画布图(白底黑字灰度) -> MNIST 风格 28x28 张量:
    裁剪笔画外接框 -> 保持比例缩到 20x20 内 -> 按质心贴到 28x28 中心"""
    a = 255 - canvas_img                        # 反色: 笔画变亮(黑底白字)
    ys, xs = np.where(a > 40)
    if len(xs) == 0:
        return None, None
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    s = 20.0 / max(h, w)                        # 长边缩到 20 (MNIST 惯例)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
    # 质心居中
    m = a.astype(np.float32)
    my, mx = np.mgrid[0:nh, 0:nw]
    tot = m.sum() + 1e-6
    cy, cx = (m * my).sum() / tot, (m * mx).sum() / tot
    canvas28 = np.zeros((28, 28), np.uint8)
    ox = int(round(14 - cx))
    oy = int(round(14 - cy))
    x0, y0 = max(0, ox), max(0, oy)
    x1 = min(28, ox + nw)
    y1 = min(28, oy + nh)
    canvas28[y0:y1, x0:x1] = a[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    x = canvas28.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0), canvas28


class DrawCanvas(QWidget):
    """手写画布: QPixmap 上用圆头粗笔绘制, 可导出灰度 ndarray"""
    def __init__(self, size=CANVAS_SIZE, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._clear()
        self._last = None

    def _clear(self):
        self._pixmap = QPixmap(self.size())
        self._pixmap.fill(QColor("white"))

    def clear(self):
        self._clear()
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._last = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if self._last is None:
            return
        p = e.position().toPoint()
        painter = QPainter(self._pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("black"), PEN_WIDTH)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(self._last, p)
        painter.end()
        self._last = p
        self.update()

    def mouseReleaseEvent(self, e):
        self._last = None

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._pixmap)

    def to_gray(self):
        """导出白底黑字灰度 ndarray"""
        img = self._pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
        w, h = img.width(), img.height()
        ptr = img.constBits()
        arr = np.frombuffer(ptr, dtype=np.uint8, count=w * h).reshape(h, w).copy()
        return arr


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手写识别画板 (PySide6)")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 数字模型
        self.model_digit = MnistCNN()
        self.model_digit.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model_digit.eval()
        # 汉字模型(训完自动启用)
        self.model_hanzi = None
        self.hanzi_chars = []
        self._try_load_hanzi()

        self.canvas = DrawCanvas()
        self.result_label = QLabel("请写一个数字")
        self.result_label.setFont(QFont("Arial", 22, QFont.Bold))
        self.result_label.setStyleSheet("color:#c62828;")
        self.preview = QLabel()
        self.preview.setFixedSize(112, 112)
        self.preview.setStyleSheet("border:1px solid #999;")
        self.preview.setAlignment(Qt.AlignCenter)

        btn_rec = QPushButton("识 别")
        btn_clr = QPushButton("清 空")
        for b in (btn_rec, btn_clr):
            b.setFixedWidth(120)
            b.setFont(QFont("Arial", 12, QFont.Bold))
        btn_rec.clicked.connect(self.recognize)
        btn_clr.clicked.connect(self.on_clear)

        # 模式切换
        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItems(["数字 (0-9)", "汉字 (常用字)"])
        self.mode_box.setFixedWidth(120)
        self.mode_box.setFont(QFont("Arial", 11))
        self.mode_box.currentIndexChanged.connect(self.on_mode_change)

        # 调试模式: 开启后显示 Top-3 勾选 + 真值输入 + 存为样本(采集笔迹入库)
        self.debug_box = QtWidgets.QCheckBox("调试模式 (勾选正确字留样本)")
        self.debug_box.setFont(QFont("Arial", 10))
        self.debug_box.setChecked(True)
        self.debug_box.toggled.connect(self._update_ui_mode)

        right = QVBoxLayout()
        right.addWidget(self.mode_box)
        right.addWidget(self.debug_box)
        right.addWidget(btn_rec)
        right.addWidget(btn_clr)
        right.addSpacing(8)
        right.addWidget(self.result_label)
        right.addSpacing(6)
        # Top-3 候选: 每行 "名次 + 字 + 勾选 + 概率条 + 百分比"
        self.top_labels = []
        self.top_checks = []
        for rank in range(3):
            row = QHBoxLayout()
            rank_lab = QLabel(f"{rank+1}.")
            rank_lab.setFont(QFont("Arial", 12))
            lab = QLabel("–")
            lab.setFont(QFont("Arial", 20, QFont.Bold))
            lab.setFixedWidth(40)
            lab.setAlignment(Qt.AlignCenter)
            chk = QtWidgets.QRadioButton()
            chk.setFont(QFont("Arial", 14))
            chk.setToolTip("勾选识别正确的字, 点\"存为样本\"即可入库优化")
            chk.setAutoExclusive(False)        # Qt互斥组会阻断程序化取消勾选, 改手动互斥
            chk.setVisible(False)              # 默认数字模式隐藏
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setFixedHeight(16)
            bar.setTextVisible(False)
            pct = QLabel("")
            pct.setFont(QFont("Arial", 11))
            pct.setFixedWidth(64)
            pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(rank_lab)
            row.addWidget(lab)
            row.addWidget(chk)
            row.addWidget(bar, 1)
            row.addWidget(pct)
            right.addLayout(row)
            self.top_labels.append((lab, bar, pct))
            self.top_checks.append(chk)
            chk.toggled.connect(lambda on, k=rank: self._top_check_toggled(k, on))
        right.addSpacing(8)
        self.cap = QLabel("模型看到的 28×28:")
        right.addWidget(self.cap)
        right.addWidget(self.preview)
        # 采集区: 汉字写对/写错都可入库, 供微调
        self.user_tag = QtWidgets.QLineEdit()
        self.user_tag.setPlaceholderText("输入正确字(如:草)")
        self.user_tag.setFixedWidth(120)
        self.user_tag.setFont(QFont("Arial", 11))
        self.collect_btn = QPushButton("存为样本")
        self.collect_btn.setFixedWidth(120)
        self.collect_btn.clicked.connect(self._on_collect)
        right.addSpacing(6)
        right.addWidget(self.user_tag)
        right.addWidget(self.collect_btn)
        self._last_small = None
        right.addStretch(1)

        layout = QHBoxLayout()
        layout.addWidget(self.canvas)
        layout.addLayout(right)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self._update_ui_mode()                 # 初始化采集区可见性

    # ---------- 模式与模型 ----------
    HANZI_MODEL = os.path.join("hanzi", "hanzi_char_mobilenet.pth")
    HANZI_CHARSET = os.path.join("hanzi", "charset.json")

    def _try_load_hanzi(self):
        """汉字模型存在则加载, 不存在则等训练完成(识别时再试)"""
        try:
            from torchvision.models import mobilenet_v2
            ckpt = torch.load(self.HANZI_MODEL, map_location="cpu")
            n_cls = ckpt["num_classes"]
            m = mobilenet_v2(weights=None)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_cls)
            m.load_state_dict(ckpt["model"])
            m.eval()
            with open(self.HANZI_CHARSET, encoding="utf-8") as f:
                self.hanzi_chars = json.load(f)["chars"][:n_cls]
            self.model_hanzi = m
        except Exception:
            self.model_hanzi = None

    def _update_ui_mode(self):
        """调试模式开关 + 数字/汉字模式 共同决定采集区可见性"""
        debug = self.debug_box.isChecked()
        hanzi = self.mode_box.currentIndex() == 1
        for chk in self.top_checks:
            chk.setVisible(debug and hanzi)   # 勾选入库仅 调试+汉字
        self.user_tag.setVisible(debug and hanzi)
        self.collect_btn.setVisible(debug and hanzi)

    def on_mode_change(self, _):
        self._update_ui_mode()
        if self.mode_box.currentIndex() == 0:
            self.cap.setText("模型看到的 28×28:")
            self.result_label.setText("请写一个数字")
        else:
            self.cap.setText("模型看到的 64×64:")
            if self.model_hanzi is None:
                self._try_load_hanzi()
            self.result_label.setText("请写一个汉字" if self.model_hanzi else "汉字模型未就绪")

    def _preprocess64(self, canvas_img):
        """汉字: 外接框裁剪 -> 最长边 62 -> 64x64 居中 (对齐 HWDB 训练分布: 字充满画布)"""
        a = 255 - canvas_img
        ys, xs = np.where(a > 40)
        if len(xs) == 0:
            return None, None
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = a.shape
        s = 62.0 / max(h, w)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((64, 64), np.uint8)
        ox, oy = (64 - nw) // 2, (64 - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = a
        x = canvas.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        return torch.from_numpy(x).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1), canvas

    def on_clear(self):
        self.canvas.clear()
        self.result_label.setText("请写一个数字" if self.mode_box.currentIndex() == 0 else "请写一个汉字")
        for lab, bar, pct in self.top_labels:
            lab.setText("–")
            bar.setValue(0)
            pct.setText("")
        for chk in self.top_checks:
            chk.setText("")
            chk.setChecked(False)
        self.preview.clear()

    def _show_top3(self, prob, names):
        top3 = torch.topk(prob, 3)
        for rank, (p, i) in enumerate(zip(top3.values, top3.indices)):
            lab, bar, pct = self.top_labels[rank]
            name = names[int(i)]
            lab.setText(name)
            self.top_checks[rank].setText(name)
            self.top_checks[rank].setChecked(False)
            v = p.item()
            bar.setValue(int(v * 1000))
            pct.setText(f"{v*100:.2f}%")
            if rank == 0:
                lab.setStyleSheet("color:#c62828;")
                bar.setStyleSheet("QProgressBar::chunk{background:#c62828;}")
            else:
                lab.setStyleSheet("color:#333;")
                bar.setStyleSheet("QProgressBar::chunk{background:#888;}")

    def _show_preview(self, small):
        big = cv2.resize(small, (112, 112), interpolation=cv2.INTER_NEAREST)
        h, w = big.shape
        qimg = QImage(big.data, w, h, w, QImage.Format_Grayscale8)
        self.preview.setPixmap(QPixmap.fromImage(qimg).scaled(
            112, 112, Qt.KeepAspectRatio, Qt.FastTransformation))

    @staticmethod
    def _tta_variants(small):
        """测试时增强: 原图 + 4 个微变换(±4°旋转/±1px平移), 预测取平均, 抗抖动"""
        out = [small]
        for ang in (-4, 4):
            M = cv2.getRotationMatrix2D((32, 32), ang, 1.0)
            out.append(cv2.warpAffine(small, M, (64, 64), borderValue=0))
        M = np.float32([[1, 0, 1.5], [0, 1, 0]])
        out.append(cv2.warpAffine(small, M, (64, 64), borderValue=0))
        M = np.float32([[1, 0, 0], [0, 1, -1.5]])
        out.append(cv2.warpAffine(small, M, (64, 64), borderValue=0))
        return out

    def _infer_tta(self, model, small, in_ch):
        """5 视角 TTA 平均"""
        probs = None
        for v in self._tta_variants(small):
            x = v.astype(np.float32) / 255.0
            x = (x - 0.5) / 0.5
            t = torch.from_numpy(x)[None, None]
            if in_ch == 3:
                t = t.repeat(1, 3, 1, 1)
            with torch.no_grad():
                p = torch.softmax(model(t), dim=1)[0]
            probs = p if probs is None else probs + p
        return probs / 5.0

    def _top_check_toggled(self, idx, on):
        """手动互斥: 同一时刻最多勾一个候选字"""
        if on:
            for j, c in enumerate(self.top_checks):
                if j != idx:
                    c.setChecked(False)

    def _on_collect(self):
        if self._last_small is None:
            self.collect_btn.setText("先点识别")
            return
        if self.mode_box.currentIndex() != 1:
            self.collect_btn.setText("仅汉字模式")
            return
        # 勾选的候选字优先, 其次输入框手打的正确字
        tag = ""
        for chk in self.top_checks:
            if chk.isChecked() and chk.text():
                tag = chk.text()
                break
        if not tag:
            tag = self.user_tag.text().strip()
        if not tag:
            self.collect_btn.setText("勾选或输入正确字")
            return
        import time as _time
        d = os.path.join("hanzi", "user_data")
        os.makedirs(d, exist_ok=True)
        fn = os.path.join(d, f"{tag}_{int(_time.time()*1000)}.png")
        ok, buf = cv2.imencode('.png', self._last_small)
        buf.tofile(fn)
        self.user_tag.clear()
        for chk in self.top_checks:
            chk.setChecked(False)
        self.collect_btn.setText(f"已存({tag})")

    def recognize(self):
        gray = self.canvas.to_gray()
        if self.mode_box.currentIndex() == 1:          # 汉字模式
            if self.model_hanzi is None:
                self._try_load_hanzi()
            if self.model_hanzi is None:
                self.result_label.setText("汉字模型未就绪")
                return
            x, small = self._preprocess64(gray)
            if x is None:
                self.result_label.setText("画布是空的")
                return
            prob = self._infer_tta(self.model_hanzi, small, 3)
            conf, idx = prob.max(0)
            ch = self.hanzi_chars[int(idx)] if int(idx) < len(self.hanzi_chars) else "?"
            self.result_label.setText(f"识别结果: {ch}  ({conf.item()*100:.1f}%)")
            self._show_top3(prob, self.hanzi_chars + ["?"] * 10)
            self._show_preview(small)
            self._last_small = small
            return
        # 数字模式
        x, small = preprocess(gray)
        if x is None:
            self.result_label.setText("画布是空的")
            return
        with torch.no_grad():
            prob = torch.softmax(self.model_digit(x), dim=1)[0]
        conf, idx = prob.max(0)
        self.result_label.setText(f"识别结果: {int(idx)}  ({conf.item()*100:.1f}%)")
        self._show_top3(prob, [str(d) for d in range(10)])
        self._show_preview(small)
        self._last_small = None


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
