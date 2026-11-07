# 墨识 AI 接入说明

版本：1.0.0  
默认本地服务：`http://127.0.0.1:18620/`  
AI 配置目录：`runtime/ai/`

---

## 1. 支持哪些模型

只要接口兼容 **OpenAI Chat Completions 多模态格式**，就能接入：

| 预置名称 | Base URL 示例 | 模型名示例 |
| --- | --- | --- |
| 通义千问 VL | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-vl-max` |
| 智谱 GLM-4V | `https://open.bigmodel.cn/api/paas/v4` | `glm-4v-flash` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Moonshot | `https://api.moonshot.cn/v1` | `moonshot-v1-8k-vision-preview` |
| DeepSeek（文本复核） | `https://api.deepseek.com/v1` | `deepseek-chat` |
| Ollama 本地 | `http://127.0.0.1:11434/v1` | `qwen2.5vl:7b` |
| 自定义 | 你自己的兼容端点 | 你自己的模型名 |

> 模型名以服务商当前控制台为准，可直接在页面里修改。  
> **DeepSeek 预设为文本复核模式**：它不接收图片，只根据本地识别文本与 Top-5 候选概率判断最可能的字；适合纠正常见形近字。
> 如果你的 DeepSeek 端点支持视觉，在编辑接入时勾选「支持图像输入」并填写视觉模型名，即可切换为直接读图。

Ollama 示例：

```powershell
ollama pull qwen2.5vl:7b
ollama serve
```

Base URL 填 `http://127.0.0.1:11434/v1`，API Key 留空，模型名填 `qwen2.5vl:7b`。

---

## 2. 请求格式

墨识会向 `Base URL + /chat/completions` 发送：

```json
{
  "model": "qwen-vl-max",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "请识别图片中的汉字或数字，只输出 JSON：{\"text\":\"识别结果\"}"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }
  ],
  "temperature": 0,
  "max_tokens": 64,
  "stream": false
}
```

响应解析顺序：

```text
choices[0].message.content
→ 去掉 ``` 代码围栏
→ 解析 {"text":"..."} 
→ 仍失败则取第一段汉字/数字/字母
```

---

## 3. 本地 API

### 3.1 查看配置

`GET /api/ai/config`

返回：

```json
{
  "ok": true,
  "active": "qwen",
  "providers": [
    {
      "id": "qwen",
      "name": "通义千问 VL",
      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "model": "qwen-vl-max",
      "api_key": "••••••••",
      "has_key": true,
      "enabled": true,
      "configured": true
    }
  ],
  "weights": {"local": 1.3, "providers": {"qwen": 0.8}},
  "stats": {"total": 12, "local_acc": 0.83, "ai_acc": 0.75}
}
```

API Key 永远不回传明文。

### 3.2 新增 / 修改接入

`POST /api/ai/provider`

```json
{
  "id": "custom_xxx",          // 新增时省略；修改时传已有 id
  "name": "我的模型",
  "base_url": "https://example.com/v1",
  "model": "my-vl-model",
  "api_key": "sk-...",         // 省略或留空 = 不修改
  "enabled": true,
  "timeout": 60,
  "vision": true,              // false = 文本复核，不发送图片
  "prompt": "请识别图片中的文字，只输出 JSON：{\"text\":\"...\"}"
}
```

- 新接入建议不传 `id`，服务会生成 `custom_xxxxxx`。
- `enabled=true` 会自动设为默认复核模型。
- 内置预置不可删除；可以停用或改配置。

### 3.2.1 设为默认 / 删除

- `POST /api/ai/active`：`{"id":"qwen"}`，把某个已配置模型设为默认。
- `DELETE /api/ai/provider/{id}`：删除自定义接入（内置预设不可删除）。

### 3.3 测试连接

`POST /api/ai/test`

```json
{"id": "qwen"}
```

服务会用 `runtime/data/samples` 中的第一张样本做一次真实识别，返回：

```json
{"ok": true, "text": "三", "latency_ms": 812.4, "model": "qwen-vl-max"}
```

### 3.4 AI 复核

`POST /api/ai/recognize`

```json
{
  "provider_id": "qwen",
  "image": "data:image/png;base64,...",
  "mode": "hanzi",
  "local": {
    "text": "三",
    "items": [
      {"char": "三", "confidence": 0.93,
       "topk": [{"char": "三", "prob": 0.93}, {"char": "二", "prob": 0.04}]}
    ]
  }
}
```

返回：

```json
{
  "ok": true,
  "ai": {
    "text": "三",
    "latency_ms": 812.4,
    "provider": "通义千问 VL",
    "provider_id": "qwen",
    "model": "qwen-vl-max"
  },
  "compare": {
    "local_text": "三",
    "ai_text": "三",
    "same": true,
    "similarity": 1.0,
    "positions": [{"local": "三", "ai": "三", "same": true}],
    "fused_text": "三",
    "fused_sources": ["local"],
    "weights": {"local": 1.3, "ai": 0.8}
  },
  "weights": {"local": 1.3, "providers": {"qwen": 0.8}}
}
```

融合规则：

```text
每个字：本地 Top-5 概率 × 本地权重
      + AI 字 × AI 权重 × 0.92
取最高分作为融合结果
长度不同则保留本地结果，并标注“无法逐字融合”
```

### 3.5 记录反馈

`POST /api/ai/feedback`

```json
{
  "local_text": "三",
  "ai_text": "二",
  "correct_text": "三",
  "provider_id": "qwen",
  "accepted": "local",        // local | ai | custom
  "mode": "hanzi"
}
```

写入 `runtime/ai/feedback.jsonl`。每条记录同时保存 `local_ok`、`ai_ok`。

### 3.6 智能优化权重

`POST /api/ai/optimize`

至少需要 3 条反馈。算法：

```text
local_acc = 本地正确数 / 总数
ai_acc    = AI 正确数 / 总数
w         = accuracy^1.6，加 Laplace 平滑
归一化到 wl + wa ≈ 2
写入 runtime/ai/weights.json
```

返回优化后的权重与统计。

### 3.7 直接修改权重

`POST /api/ai/weights`

```json
{
  "local": 1.2,
  "providers": {"qwen": 0.9, "custom_xxx": 1.1}
}
```

范围 `0–2.5`，立即写盘，下一次复核生效。

---

## 4. 反馈如何参与训练

在识别页的「AI 交叉复核」里：

1. 点击「本地正确」或「AI 正确」；
2. 勾选「加入样本库」；
3. 单字情况下会保存一张 64×64 样本，AI 标签默认 **weight=0.65**；
4. 训练器读取 `runtime/data/samples/*.json` 的 `weight`，在交叉熵中按权重相乘；
5. 下一次「优化训练」会同时使用：
   - 本地手工样本（weight 1.0）
   - AI 确认样本（weight 0.65）
   - 字体合成样本
   - HWDB 通用域回放

也就是说：AI 不是只做“显示对比”，它确认的标签会以较低影响进入微调；错得离谱的 AI 标签影响也会被权重削弱。

多字反馈目前只记录到反馈表，样本需要逐字在结果卡片里添加；避免把多字整图错误地当成一个类别。

---

## 5. 安全与隐私

- API Key 在 Windows 上用 **DPAPI** 加密后写入 `runtime/ai/providers.json`，仅当前 Windows 用户可解密；
- API Key 不会通过 `/api/ai/config` 回传明文；
- 调用 AI 时，图片会从本机直接发往你配置的 Base URL；这是使用云端模型的必要步骤；
- 本地识别、样本、模型、报告永远不会上传；
- 服务只监听 `127.0.0.1`，不监听局域网。

---

## 6. 自定义模型 checklist

- [ ] 支持 `POST /chat/completions`
- [ ] 支持 `messages[].content` 数组
- [ ] 支持 `image_url` 类型，值可以是 `data:image/png;base64,...`
- [ ] 返回 `choices[0].message.content` 字符串
- [ ] 最好只输出 `{"text":"..."}`

如果自定义服务字段不同，可以在墨识前面加一层很薄的兼容代理（把请求转成上面的格式），无需改墨识代码。

---

## 7. 常见问题

**Q：没有 API Key 能用吗？**  
可以接 Ollama 本地模型；云端模型必须填 Key。

**Q：AI 返回很多解释文字怎么办？**  
墨识会自动找 JSON、`"text"` 字段或第一段汉字/数字；也可以在提示词里强调“只输出 JSON”。

**Q：AI 和本地结果长度不同？**  
会逐字对齐显示差异，但融合结果默认保留本地；建议对多字图片先确认本地分割是否正确。

**Q：权重调多少合适？**  
默认本地 1.0、AI 1.0。AI 在少量样本上更准时可调到 1.2–1.5；AI 经常错时降到 0.5 以下。也可以点「智能优化」让反馈自动算。
