# -*- coding: utf-8 -*-
"""墨识桌面外壳 · PySide6 + QWebEngineView。

显示层统一使用 PySide6：原生白色窗口承载本地 http://127.0.0.1:18620/
的 Web 界面，保留浏览器访问、网页调试与网页内训练能力。
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import (QColor, QFont, QIcon, QKeySequence, QPainter, QPalette,
                           QPixmap, QShortcut)
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
                               QMainWindow, QPushButton, QVBoxLayout, QWidget)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
INFO = RUNTIME / "server.json"
SETTINGS = QSettings("Moshi", "MoshiHandwriting")


def find_python() -> str:
    p = RUNTIME / "python.txt"
    if p.exists():
        v = p.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        if v and Path(v[0]).exists():
            return v[0]
    candidates = [sys.executable]
    for c in (r"D:\Anaconda\envs\pytorch\pythonw.exe",
              r"D:\Anaconda\envs\pytorch\python.exe"):
        if Path(c).exists():
            candidates.append(c)
    return candidates[0]


def light_title_bar(widget):
    if os.name != "nt":
        return
    try:
        import ctypes
        hwnd = ctypes.c_void_p(int(widget.winId()))
        for attr, value in ((20, 0), (35, 0x00FBF9F7), (36, 0x003D2116), (34, 0x00F2EAE5)):
            v = ctypes.c_int(value)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(v), ctypes.sizeof(v))
    except Exception:
        pass


def health(url: str, timeout=0.8):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/health", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


class Brand(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(54, 54)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#0f9f95"))
        p.drawRoundedRect(0, 0, 54, 54, 14, 14)
        p.setBrush(QColor("#ffffff"))
        p.drawRoundedRect(13, 13, 28, 28, 8, 8)
        p.setBrush(QColor("#0f9f95"))
        p.drawRoundedRect(19, 19, 6, 6, 2, 2)
        p.drawRoundedRect(29, 19, 6, 6, 2, 2)
        p.drawRoundedRect(19, 29, 6, 6, 2, 2)
        p.drawRoundedRect(29, 29, 6, 6, 2, 2)


class LaunchScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:#ffffff;")
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(14)
        row = QHBoxLayout()
        row.setAlignment(Qt.AlignCenter)
        row.addWidget(Brand())
        title = QLabel("墨识")
        title.setFont(QFont("Microsoft YaHei UI", 30, QFont.Bold))
        title.setStyleSheet("color:#14213d;")
        row.addWidget(title)
        lay.addLayout(row)
        self.msg = QLabel("正在启动本地识别引擎…")
        self.msg.setFont(QFont("Microsoft YaHei UI", 11))
        self.msg.setStyleSheet("color:#7b879c;")
        self.msg.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.msg)
        self.detail = QLabel("仅监听 127.0.0.1 · 数据不出本机")
        self.detail.setFont(QFont("Microsoft YaHei UI", 9))
        self.detail.setStyleSheet("color:#a5afc0;")
        self.detail.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.detail)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("墨识 · 手写识别")
        ico = ROOT / "assets" / "moshi.ico"
        if ico.exists():
            self.setWindowIcon(QIcon(str(ico)))
        self.resize(1320, 880)
        self.setMinimumSize(1080, 720)
        self.server_proc = None
        self.url = ""
        self.tries = 0
        self.view = None
        self.launch = LaunchScreen()
        self.setCentralWidget(self.launch)

        QShortcut(QKeySequence("F5"), self, activated=self.reload_page)
        QShortcut(QKeySequence("F12"), self, activated=self.devtools)
        QShortcut(QKeySequence("Ctrl+Q"), self, activated=self.close)

        QTimer.singleShot(120, self.start_or_connect)

    def showEvent(self, e):
        super().showEvent(e)
        QTimer.singleShot(20, lambda: light_title_bar(self))

    def start_or_connect(self):
        info = {}
        try:
            info = json.loads(INFO.read_text(encoding="utf-8"))
        except Exception:
            pass
        candidates = []
        if info.get("url"):
            candidates.append(info["url"])
        candidates.append("http://127.0.0.1:18620")
        for u in candidates:
            h = health(u)
            if h:
                self.url = u
                self.show_web()
                return
        self.spawn_server()

    def spawn_server(self):
        py = find_python()
        pyw = Path(py)
        if pyw.name.lower() == "python.exe":
            cand = pyw.with_name("pythonw.exe")
            if cand.exists():
                pyw = cand
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env["MOSHI_ROOT"] = str(ROOT)
        out = open(LOGS / "desktop.log", "ab", buffering=0)
        creation = 0x08000000 if os.name == "nt" else 0
        try:
            self.server_proc = subprocess.Popen(
                [str(pyw), str(ROOT / "app" / "server.py"), "--no-browser"],
                cwd=str(ROOT), env=env, stdout=out, stderr=out, creationflags=creation,
            )
        except Exception as exc:
            self.launch.msg.setText("无法启动本地服务：" + str(exc))
            self.launch.detail.setText("请查看 logs/desktop.log")
            return
        self.launch.msg.setText("正在载入 3760 类汉字模型…")
        QTimer.singleShot(600, self.poll_server)

    def poll_server(self):
        self.tries += 1
        info = {}
        try:
            info = json.loads(INFO.read_text(encoding="utf-8"))
        except Exception:
            pass
        if info.get("url") and health(info["url"]):
            self.url = info["url"]
            self.show_web()
            return
        if self.tries > 90:
            self.launch.msg.setText("本地服务启动超时")
            self.launch.detail.setText("请查看 logs/desktop.log，或重新打开程序")
            return
        if self.tries in (8, 20, 40):
            self.launch.msg.setText("正在载入模型…首次启动会更慢一点")
        QTimer.singleShot(500, self.poll_server)

    def show_web(self):
        self.view = QWebEngineView()
        s = self.view.settings()
        s.setAttribute(QWebEngineSettings.LocalStorageEnabled, True)
        s.setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        s.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, False)
        s.setAttribute(QWebEngineSettings.ErrorPageEnabled, True)
        self.view.setStyleSheet("background:#f6f8fb;")
        self.view.load(QUrl(self.url))
        self.view.titleChanged.connect(lambda t: self.setWindowTitle(t or "墨识 · 手写识别"))
        self.view.renderProcessTerminated.connect(self.on_crash)
        self.setCentralWidget(self.view)
        geo = SETTINGS.value("geometry")
        if geo:
            self.restoreGeometry(geo)

    def on_crash(self, status, code):
        self.launch.msg.setText("显示组件异常退出")
        self.launch.detail.setText("请重新打开程序；服务日志在 logs/server.log")
        self.setCentralWidget(self.launch)

    def reload_page(self):
        if self.view:
            self.view.reload()

    def devtools(self):
        if not self.view:
            return
        try:
            w = QWebEngineView()
            w.setWindowTitle("墨识 · Web 调试器")
            w.resize(1100, 760)
            self._dev = w
            self.view.page().setDevToolsPage(w.page())
            w.show()
        except Exception as exc:
            self.launch.msg.setText("调试器打开失败：" + str(exc))

    def closeEvent(self, e):
        SETTINGS.setValue("geometry", self.saveGeometry())
        if self.server_proc and self.server_proc.poll() is None:
            try:
                self.server_proc.terminate()
                self.server_proc.wait(timeout=3)
            except Exception:
                try:
                    self.server_proc.kill()
                except Exception:
                    pass
        e.accept()


def main():
    os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-features=Translate")
    app = QApplication(sys.argv)
    app.setApplicationName("墨识")
    app.setOrganizationName("Moshi")
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    pal = app.palette()
    pal.setColor(QPalette.Highlight, QColor("#0fa99f"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)
    ico = ROOT / "assets" / "moshi.ico"
    if ico.exists():
        app.setWindowIcon(QIcon(str(ico)))
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
