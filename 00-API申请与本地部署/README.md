# API 申请与本地部署指南

本章详细介绍如何申请各种大模型 API，以及如何在本地部署大模型。

## 一、主流 API 申请教程

### 1. OpenAI API（GPT 系列）

#### 申请步骤

1. **访问官网**：https://platform.openai.com/
2. **注册账号**：需要邮箱注册（国内用户需要科学上网）
3. **绑定支付方式**：需要海外信用卡（如 Depay 虚拟卡）
4. **获取 API Key**：
   - 登录后进入 API Keys 页面
   - 点击 "Create new secret key"
   - 保存好 API Key（只显示一次）

#### 价格（2025年最新）

| 模型 | 输入价格 | 输出价格 |
|------|----------|----------|
| GPT-4o | $2.50/1M tokens | $10.00/1M tokens |
| GPT-4o-mini | $0.15/1M tokens | $0.60/1M tokens |
| GPT-4 Turbo | $10.00/1M tokens | $30.00/1M tokens |
| o1-preview | $15.00/1M tokens | $60.00/1M tokens |
| o1-mini | $1.75/1M tokens | $7.00/1M tokens |

#### 调用示例

```python
from openai import OpenAI

client = OpenAI(api_key="sk-xxx")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "你好"}]
)

print(response.choices[0].message.content)
```

---

### 2. Anthropic Claude API

#### 申请步骤

1. **访问官网**：https://console.anthropic.com/
2. **注册账号**：支持 Google 登录
3. **获取 API Key**：
   - 进入 Dashboard
   - 点击 "Create API Key"
   - 新用户有 $5 免费额度

#### 价格（2025年最新）

| 模型 | 输入价格 | 输出价格 |
|------|----------|----------|
| Claude Opus 4 | $15.00/1M tokens | $75.00/1M tokens |
| Claude Sonnet 4 | $3.00/1M tokens | $15.00/1M tokens |
| Claude Haiku 3.5 | $0.80/1M tokens | $4.00/1M tokens |

#### 调用示例

```python
import anthropic

client = anthropic.Anthropic(api_key="sk-ant-xxx")

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "你好"}]
)

print(message.content[0].text)
```

---

### 3. Google Gemini API（推荐！免费额度大）

#### 申请步骤

1. **访问 Google AI Studio**：https://aistudio.google.com/apikeys
2. **登录 Google 账号**
3. **创建 API Key**：点击 "Create API Key"
4. **绑定信用卡**（可选）：可获得 $300/90天 免费额度

#### 免费额度（2025年最新）

| 模型 | RPM（每分钟请求） | RPD（每日请求） | TPM（每分钟Token） |
|------|-------------------|-----------------|-------------------|
| Gemini 2.5 Pro | 5 | 100 | 250,000 |
| Gemini 2.5 Flash | 10 | 250 | 250,000 |
| Gemini 2.5 Flash-Lite | 15 | 1,000 | 250,000 |

#### 调用示例

```python
import google.generativeai as genai

genai.configure(api_key="AIza-xxx")

model = genai.GenerativeModel('gemini-2.5-flash')

response = model.generate_content("你好")
print(response.text)
```

---

### 4. DeepSeek API（国产推荐！性价比高）

#### 申请步骤

1. **访问官网**：https://platform.deepseek.com/
2. **注册账号**：支持手机号、微信登录
3. **获取 API Key**：
   - 进入 API Keys 页面
   - 点击 "创建 API Key"
   - 新用户赠送 10 元额度

#### 价格（2025年最新）

| 模型 | 输入价格 | 输出价格 | 备注 |
|------|----------|----------|------|
| DeepSeek-V3 | ¥2/百万 tokens | ¥8/百万 tokens | 错峰时段 5 折 |
| DeepSeek-R1 | ¥4/百万 tokens | ¥16/百万 tokens | 错峰时段 2.5 折 |

**错峰时段**：北京时间 00:30 - 08:30

#### 调用示例

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-xxx",
    base_url="https://api.deepseek.com/v1"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好"}]
)

print(response.choices[0].message.content)
```

---

### 5. NVIDIA NIM API（免费！强烈推荐）

#### 简介

NVIDIA NIM 提供多种开源模型的 API 调用，**部分模型完全免费**，国内访问稳定。

#### 申请步骤

1. **访问官网**：https://build.nvidia.com/
2. **注册/登录 NVIDIA 账号**
3. **选择模型**：浏览可用模型列表
4. **获取 API Key**：点击 "Get API Key"

#### 免费可用模型

| 模型 | 说明 |
|------|------|
| meta/llama-3.1-405b-instruct | Llama 3.1 405B |
| meta/llama-3.1-70b-instruct | Llama 3.1 70B |
| meta/llama-3.1-8b-instruct | Llama 3.1 8B |
| mistralai/mistral-7b-instruct-v0.3 | Mistral 7B |
| deepseek-ai/deepseek-r1 | DeepSeek R1 |
| moonshotai/kimi-k2-2505 | Kimi K2.5 |

#### 调用示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1/",
    api_key="nvapi-xxx"
)

response = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[{"role": "user", "content": "你好"}],
    temperature=0.7,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

---

### 6. 其他免费/低成本 API

#### 硅基流动

- 官网：https://siliconflow.cn/
- 特点：国内访问稳定，多种开源模型
- 免费额度：新用户送额度

#### 阿里云百炼（通义千问）

- 官网：https://bailian.console.aliyun.com/
- 特点：国产模型，中文能力强
- 免费额度：新用户有免费额度

#### 智谱 AI（ChatGLM）

- 官网：https://open.bigmodel.cn/
- 特点：国产模型，开源版本可本地部署
- 免费额度：新用户送 tokens

#### 月之暗面（Kimi）

- 官网：https://platform.moonshot.cn/
- 特点：长文本处理能力强
- 免费额度：新用户送额度

---

## 二、本地模型部署

### 1. Ollama（推荐！最简单）

#### 简介

Ollama 是最简单的本地大模型部署工具，支持 Windows、macOS、Linux。

#### 安装步骤

1. **下载安装**：https://ollama.com/
   - Windows: 下载 .exe 安装包
   - macOS: 下载 .dmg 安装包
   - Linux: `curl -fsSL https://ollama.com/install.sh | sh`

2. **验证安装**：
   ```bash
   ollama --version
   ```

3. **下载模型**：
   ```bash
   # DeepSeek R1（推荐）
   ollama run deepseek-r1:7b
   
   # Qwen 系列
   ollama run qwen2.5:7b
   
   # Llama 系列
   ollama run llama3.1:8b
   
   # Mistral
   ollama run mistral:7b
   ```

#### 常用命令

```bash
# 查看已安装模型
ollama list

# 运行模型
ollama run deepseek-r1:7b

# 删除模型
ollama rm deepseek-r1:7b

# 查看模型信息
ollama show deepseek-r1:7b
```

#### API 调用

Ollama 默认在 11434 端口提供 API 服务：

```python
import requests

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "deepseek-r1:7b",
        "messages": [{"role": "user", "content": "你好"}],
        "stream": False
    }
)

print(response.json()["message"]["content"])
```

#### 兼容 OpenAI API

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # 任意值
)

response = client.chat.completions.create(
    model="deepseek-r1:7b",
    messages=[{"role": "user", "content": "你好"}]
)

print(response.choices[0].message.content)
```

#### 修改模型存储路径

```bash
# Windows: 设置环境变量
setx OLLAMA_MODELS "D:\ollama_models"

# macOS/Linux
export OLLAMA_MODELS="/path/to/models"
```

---

### 2. LM Studio

#### 简介

LM Studio 是一个图形化的本地模型运行工具，适合非技术用户。

#### 安装步骤

1. **下载**：https://lmstudio.ai/
2. **安装**：双击安装包
3. **搜索模型**：在搜索框输入模型名称
4. **下载模型**：点击下载按钮
5. **运行对话**：在 Chat 界面使用

#### 支持的模型

- Llama 系列
- Mistral 系列
- Qwen 系列
- DeepSeek 系列
- Phi 系列

---

### 3. vLLM（高性能推理）

#### 简介

vLLM 是高性能的大模型推理引擎，适合生产环境。

#### 安装

```bash
pip install vllm
```

#### 使用示例

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

outputs = llm.generate(["你好"], sampling_params)

print(outputs[0].outputs[0].text)
```

---

### 4. Hugging Face Transformers

#### 安装

```bash
pip install transformers torch
```

#### 使用示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

messages = [{"role": "user", "content": "你好"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 三、模型选择建议

### 按场景选择

| 场景 | 推荐方案 |
|------|----------|
| 学习测试 | Gemini API（免费）或 NVIDIA NIM（免费） |
| 日常开发 | DeepSeek API（便宜）或 Ollama 本地 |
| 生产环境 | OpenAI GPT-4o 或 Claude |
| 长文本处理 | Kimi 或 Claude |
| 代码生成 | GPT-4o 或 DeepSeek-R1 |
| 中文任务 | DeepSeek 或 Qwen |

### 按预算选择

| 预算 | 推荐方案 |
|------|----------|
| 零成本 | NVIDIA NIM + Gemini API + Ollama 本地 |
| 低成本 | DeepSeek API（错峰使用） |
| 中等预算 | GPT-4o-mini + DeepSeek |
| 高预算 | GPT-4o + Claude Opus |

---

## 四、常见问题

### Q1: 国内如何访问 OpenAI API？

使用代理或 API 中转服务（如能用 AI、OpenRouter 等）。

### Q2: 本地部署需要什么配置？

| 模型大小 | 最低显存 | 推荐显存 |
|----------|----------|----------|
| 7B | 8GB | 12GB |
| 14B | 16GB | 24GB |
| 32B | 24GB | 48GB |
| 70B | 48GB | 80GB |

### Q3: 如何选择本地模型？

- **日常对话**：Qwen2.5-7B、Llama3.1-8B
- **代码生成**：DeepSeek-Coder、CodeLlama
- **推理任务**：DeepSeek-R1-Distill-Qwen
- **低配置电脑**：Phi-3-mini、Qwen2.5-3B

---

## 下一步

掌握了 API 申请和本地部署后，开始学习 Prompt 工程：
→ [05-Prompt工程](../05-Prompt工程/)
