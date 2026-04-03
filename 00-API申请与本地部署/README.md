# 00-API 申请与本地部署

大家好！在开始学习 AI Agent 之前，我们需要先搞定两件事：
1. **申请一个免费的 API**（这样就能调用大模型了）
2. **或者在本地部署模型**（完全免费，隐私安全）

## 一、快速开始：推荐方案

### 方案 A：使用免费 API（推荐新手）

**NVIDIA NIM API** - 完全免费，国内可访问

```python
import openai

# 配置 API
API_KEY = "你的 API Key"
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "meta/llama-3.1-70b-instruct"

# 创建客户端
client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 调用模型
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "你好"}]
)

print(response.choices[0].message.content)
```

**申请步骤：**
1. 访问 https://build.nvidia.com/
2. 注册 NVIDIA 账号
3. 点击 "Get API Key"
4. 复制 API Key 到代码中

### 方案 B：本地部署（推荐有显卡的同学）

**Ollama** - 一键安装，超简单

```bash
# 1. 下载安装 Ollama
# 访问 https://ollama.com 下载安装包

# 2. 安装后运行模型
ollama run qwen2.5:7b
```

```python
import openai

# 本地模型也兼容 OpenAI 格式！
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # 随便填
)

response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "你好"}]
)

print(response.choices[0].message.content)
```

---

## 二、免费 API 大全

### 1. NVIDIA NIM（强烈推荐⭐）

**特点：**
- ✅ 部分模型完全免费
- ✅ 国内访问稳定
- ✅ 模型种类多

**可用模型：**
```python
models = [
    "meta/llama-3.1-70b-instruct",    # Llama 3.1 70B
    "meta/llama-3.1-8b-instruct",     # Llama 3.1 8B
    "qwen/qwen2.5-72b-instruct",      # 通义千问
    "deepseek-ai/deepseek-r1",        # DeepSeek R1
]
```

**完整示例代码：**
```python
import openai

client = openai.OpenAI(
    api_key="你的 API Key",
    base_url="https://integrate.api.nvidia.com/v1"
)

response = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手"},
        {"role": "user", "content": "你好，请介绍一下自己"}
    ],
    temperature=0.7,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

### 2. Google Gemini（免费额度大）

**特点：**
- ✅ 免费层级额度充足
- ✅ 支持多模态（图片、视频）
- ✅ 新用户送 $300 额度

**申请：** https://aistudio.google.com/apikeys

```python
import google.generativeai as genai

genai.configure(api_key="你的 API Key")

model = genai.GenerativeModel('gemini-2.0-flash')
response = model.generate_content("你好")
print(response.text)
```

### 3. DeepSeek（国产性价比之王）

**特点：**
- ✅ 新用户送 10 元额度
- ✅ 中文能力强
- ✅ 价格超低

**申请：** https://platform.deepseek.com/

```python
import openai

client = openai.OpenAI(
    api_key="你的 API Key",
    base_url="https://api.deepseek.com/v1"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好"}]
)

print(response.choices[0].message.content)
```

### 4. 其他推荐

| 平台 | 特点 | 地址 |
|------|------|------|
| 硅基流动 | 国内稳定，多种模型 | https://siliconflow.cn/ |
| 阿里云百炼 | 通义千问官方 | https://bailian.console.aliyun.com/ |
| 智谱 AI | ChatGLM 官方 | https://open.bigmodel.cn/ |
| 月之暗面 | Kimi 长文本 | https://platform.moonshot.cn/ |

---

## 三、本地部署详解

### Ollama（最简单）

**安装步骤：**

1. Windows 用户：下载 `.exe` 安装包，双击安装
2. macOS 用户：下载 `.dmg` 安装包
3. Linux 用户：`curl -fsSL https://ollama.com/install.sh | sh`

**常用命令：**
```bash
# 查看可用模型
ollama list

# 下载模型
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull deepseek-r1:7b

# 运行模型
ollama run qwen2.5:7b

# 删除模型
ollama rm qwen2.5:7b
```

**配置要求：**
```
7B 模型：最低 8GB 显存，推荐 12GB
14B 模型：最低 16GB 显存，推荐 24GB
32B 模型：最低 24GB 显存，推荐 48GB
```

**修改模型存储位置：**
```bash
# Windows
setx OLLAMA_MODELS "D:\ollama_models"

# macOS/Linux
export OLLAMA_MODELS="/path/to/models"
```

### LM Studio（图形界面）

适合不喜欢命令行的同学：

1. 下载：https://lmstudio.ai/
2. 安装后搜索模型
3. 点击下载
4. 在 Chat 界面直接使用

---

## 四、实战：用 API 完成第一个任务

### 例子 1：让 AI 自我介绍

```python
import openai

client = openai.OpenAI(
    api_key="你的 API Key",
    base_url="https://integrate.api.nvidia.com/v1"
)

response = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[
        {"role": "user", "content": "请用一句话介绍你自己"}
    ]
)

print(response.choices[0].message.content)
```

### 例子 2：翻译句子

```python
import openai

client = openai.OpenAI(
    api_key="你的 API Key",
    base_url="https://integrate.api.nvidia.com/v1"
)

response = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[
        {"role": "system", "content": "你是一个翻译助手"},
        {"role": "user", "content": "把这句话翻译成英文：你好，很高兴认识你"}
    ]
)

print(response.choices[0].message.content)
```

### 例子 3：批改作文（JSON 格式输出）

```python
import openai
import json
import json_repair

client = openai.OpenAI(
    api_key="你的 API Key",
    base_url="https://integrate.api.nvidia.com/v1"
)

essay = """
My Self-Introduction
Hello, everyone! My name is Li Ming.
I'm ten years old. I study at Green Tree Primary School.
I like reading books and playing football.
I have a happy family. I want to make friends with all of you.
Thank you!
"""

prompt = """
用小学四年级的标准给我的英文作文打分
满分 100 分，请用这样的格式输出
{"分数":"xxx","评语":"xxxx"}
直接输出不要有任何多余的输出，只要 JSON
"""

response = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": essay}
    ]
)

result = response.choices[0].message.content
print("原始结果:", result)

# 解析 JSON
rjson = json.loads(json_repair.repair_json(result))
score = rjson.get("分数", "未知")
comment = rjson.get("评语", "无评语")

print(f"\n批改结果：")
print(f"分数：{score}")
print(f"评语：{comment}")
```

---

## 五、常见问题

### Q1: API Key 安全吗？

**重要：** 不要把 API Key 上传到 GitHub！

正确做法：
```python
import os
from dotenv import load_dotenv

# 创建.env 文件存储 API Key
load_dotenv()
API_KEY = os.getenv("API_KEY")

client = openai.OpenAI(api_key=API_KEY)
```

`.env` 文件内容：
```
API_KEY=你的 API Key
```

`.gitignore` 文件：
```
.env
```

### Q2: 本地模型太慢怎么办？

- 使用更小的模型（7B 而不是 70B）
- 降低精度（使用量化版本）
- 使用 GPU 而不是 CPU

### Q3: 免费额度够用吗？

对于学习和测试：
- NVIDIA NIM：完全够用
- Gemini：免费层级很慷慨
- DeepSeek：10 元能用很久

---

## 下一步

搞定了 API，接下来我们正式学习：
→ [01-什么是大模型](../01-什么是大模型/)
