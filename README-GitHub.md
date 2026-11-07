# 墨识 · GitHub Pages 与下载说明

这是墨识的 GitHub 发布说明，包含在线介绍页、网页体验版和完整版安装包入口。

---

## 1. GitHub Pages

**仓库**

https://github.com/wocao123103/ink_recognition-moshi

**GitHub Pages 首页**

https://wocao123103.github.io/ink_recognition-moshi/

GitHub Pages 指向仓库根目录的 **`index.html`**。

### 开启方式

1. 打开仓库 **Settings → Pages**；
2. Build and deployment 选择 **Deploy from a branch**；
3. Branch 选择 `main`，目录选择 **`/ (root)`**；
4. 保存后等待 GitHub Pages 构建完成；
5. 访问：

```text
https://wocao123103.github.io/ink_recognition-moshi/
```

> 如果 Pages 域名暂时打不开，可以直接打开仓库，再点根目录的 `index.html` 预览。

---

## 2. 网页体验版

**入口**

```text
https://wocao123103.github.io/ink_recognition-moshi/tiyanban.html
```

![网页体验版](docs/screenshots/tiyanban.png)

`tiyanban.html` 是一个**只读识别体验页**：

- 只加载打包时的本地 **5200 类汉字模型** / **10 类数字模型**；
- 模型以 ONNX 形式在浏览器里运行，图片不上传服务器；
- 不包含训练入口；
- 不调用任何 AI API（DeepSeek、通义千问、OpenAI 等）；
- 不能新增、删除或修改样本库；
- 刷新页面后不保留任何内容。

体验版适合快速试写，完整功能请下载安装助手。

### 体验版依赖的静态文件

```text
tiyanban.html
tiyanban_assets/
  ├─ moshi_hanzi_5k.onnx      # 约 34 MB，5200 类模型
  ├─ moshi_digit.onnx         # 数字模型
  └─ charset.json             # 5200 字字符表
```

请确保这些文件一起上传到 GitHub 仓库根目录，并且不要被 `.gitignore` 忽略。

> `moshi_hanzi_5k.onnx` 约 **35 MB**。GitHub 网页端单文件上传有 25 MB 限制，请用 `git push`、GitHub Desktop 或 GitHub CLI 上传；不要用网页拖拽上传这个大文件。
>
> 浏览器推理使用 ONNX Runtime Web（CDN 加载）。第一次打开需要下载约 35 MB 模型；之后由浏览器缓存。

---

## 3. 完整版安装包

**百度网盘**

https://pan.baidu.com/s/1Y4puuVb8J-xsr4F-bLqMDQ?pwd=8912

**提取码**

```text
8912
```

文件名：`墨识安装助手.zip`

解压后运行 `墨识安装助手.exe`，按向导安装即可。

完整版包含：

- 5200 类汉字 + 10 类数字；
- 数字串 / 动态识别；
- 网页内优化训练；
- 20 张样本新增任意类别；
- AI 交叉复核（DeepSeek、通义千问 VL、智谱 GLM-4V、OpenAI、Moonshot、Ollama、自定义端点）；
- 样本库管理与导出；
- 调试台、本机硬件记录；
- PySide6 桌面窗口；
- 安装助手：管理员权限、自动建目录、Conda 优先、覆盖升级保留样本。

> 安装包不放进 GitHub 仓库，统一走百度网盘；GitHub 只放介绍页、体验版和文档。

---

## 4. 本地完整版启动

默认地址：

```text
http://127.0.0.1:18620/
```

服务只绑定 `127.0.0.1`，不监听局域网。

安装助手会自动：

- 检测本机 Python / Conda 环境；
- 有 Conda 时优先复用；
- 创建安装目录；
- 创建桌面快捷方式；
- 复制安装助手到桌面，方便以后覆盖升级。

---

## 5. 页面与文档

| 文件 | 用途 |
| --- | --- |
| `index.html` | 项目介绍页（GitHub Pages 首页） |
| `tiyanban.html` | 网页体验版（仅识别） |
| `README.md` | 项目总说明 |
| `机制说明.md` | 模型、训练、分割、安装机制 |
| `AI接入说明.md` | AI 提供商与接口说明 |
| `运行环境说明.txt` | Python / Conda / 依赖版本 |

---

## 6. 体验版和完整版对比

| 能力 | 网页体验版 `tiyanban.html` | 完整版安装助手 |
| --- | --- | --- |
| 本地模型识别 | ✅ | ✅ |
| 数字识别 | ✅ 单个数字 | ✅ 数字串 / 动态 |
| 上传图片 | ✅ 单字 | ✅ 单行自动分割 |
| 新增类别 | ❌ | ✅ 20 张样本 |
| 优化训练 | ❌ | ✅ |
| AI 复核 | ❌ | ✅ |
| 修改样本库 | ❌ | ✅ |
| 调试台 | ❌ | ✅ |
| 桌面 PySide6 | ❌ | ✅ |

---

## 7. 说明

- 体验版的模型是**打包时的快照**，不会随着本地训练更新；
- 如果浏览器无法加载模型，请确认是通过 GitHub Pages / HTTP 打开，而不是直接双击 `file://`；
- 百度网盘链接和提取码如失效，请在 GitHub Issues 反馈；
- 本页面不包含任何“闪退 / 关网页 / 碎裂”等恶作剧功能。
