# 多模态 Agent

多模态 Agent 是能够处理**文本、图像、音频、视频**等多种模态信息的智能代理，广泛应用于视觉分析、图像理解、视频处理等场景。

## 什么是多模态 Agent？

传统 Agent 主要处理文本，而多模态 Agent 可以：

```
┌─────────────────────────────────────────────────────────────┐
│                    多模态 Agent 架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   输入层                                                     │
│   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                   │
│   │ 文本 │  │ 图像 │  │ 音频 │  │ 视频 │                   │
│   └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘                   │
│      │         │         │         │                        │
│      ▼         ▼         ▼         ▼                        │
│   ┌─────────────────────────────────────┐                   │
│   │        多模态编码器 (Encoder)        │                   │
│   │   将不同模态转换为统一的向量表示      │                   │
│   └─────────────────────────────────────┘                   │
│                      │                                      │
│                      ▼                                      │
│   ┌─────────────────────────────────────┐                   │
│   │        多模态大模型 (VLM/LLM)        │                   │
│   │   理解、推理、决策                   │                   │
│   └─────────────────────────────────────┘                   │
│                      │                                      │
│                      ▼                                      │
│   ┌─────────────────────────────────────┐                   │
│   │           工具调用层                 │                   │
│   │   图像处理 │ 搜索 │ 代码执行 │ ...   │                   │
│   └─────────────────────────────────────┘                   │
│                      │                                      │
│                      ▼                                      │
│              输出：文本/图像/操作                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 主流多模态模型

### Flash 模型

Flash 模型通常指**快速、高效**的多模态模型，主打性价比。

#### Google Gemini Flash

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel('gemini-2.0-flash')

response = model.generate_content([
    "这张图片里有什么？",
    {
        "mime_type": "image/jpeg",
        "data": open("image.jpg", "rb").read()
    }
])

print(response.text)
```

**特点：**
- 响应速度快，成本低
- 支持多模态输入（文本、图像、视频、音频）
- 适合大规模部署
- 免费额度充足

#### Gemini Flash vs Pro 对比

| 特性 | Flash | Pro |
|------|-------|-----|
| 速度 | ⚡⚡⚡ | ⚡⚡ |
| 成本 | 低 | 中 |
| 推理能力 | 良好 | 优秀 |
| 适用场景 | 实时应用、批量处理 | 复杂推理、高质量输出 |

### VL 模型（Vision-Language）

VL 模型专注于**视觉-语言理解**，是图像理解和视觉分析的核心。

#### 主流 VL 模型

| 模型 | 提供商 | 特点 | 免费额度 |
|------|--------|------|----------|
| **GPT-4o** | OpenAI | 最强多模态能力 | 有限 |
| **GPT-4o-mini** | OpenAI | 快速、便宜 | 有 |
| **Gemini Pro Vision** | Google | 视频理解强 | 充足 |
| **Claude 3.5 Sonnet** | Anthropic | 图像理解优秀 | 有限 |
| **Qwen-VL** | 阿里 | 开源、中文友好 | 有 |
| **DeepSeek-VL** | DeepSeek | 开源、免费API | 有 |
| **LLaVA** | 开源 | 可本地部署 | 免费 |

#### GPT-4o 视觉示例

```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "分析这张图表的数据趋势"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/chart.png"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

#### Claude 视觉示例

```python
import anthropic

client = anthropic.Anthropic(api_key="YOUR_API_KEY")

with open("image.jpg", "rb") as f:
    image_data = f.read()

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(image_data).decode()
                    }
                },
                {
                    "type": "text",
                    "text": "请详细描述这张图片的内容"
                }
            ]
        }
    ]
)

print(message.content[0].text)
```

## 多模态 Agent 应用场景

### 1. 视觉分析助手

```python
class VisualAnalysisAgent:
    def __init__(self, model="gpt-4o"):
        self.client = OpenAI()
        self.model = model
    
    def analyze_image(self, image_url: str, task: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的图像分析助手，能够准确分析图像内容。"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
        )
        return response.choices[0].message.content
    
    def extract_text(self, image_url: str):
        return self.analyze_image(image_url, "请提取图片中的所有文字")
    
    def describe_scene(self, image_url: str):
        return self.analyze_image(image_url, "请详细描述图片中的场景")
    
    def analyze_chart(self, image_url: str):
        return self.analyze_image(image_url, "分析这张图表的数据和趋势")


agent = VisualAnalysisAgent()
result = agent.analyze_chart("https://example.com/sales_chart.png")
print(result)
```

### 2. 文档理解 Agent

```python
class DocumentAgent:
    def __init__(self):
        self.client = OpenAI()
    
    def analyze_document(self, image_path: str, questions: list):
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
        
        prompt = "请根据文档内容回答以下问题：\n"
        for i, q in enumerate(questions, 1):
            prompt += f"{i}. {q}\n"
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content


doc_agent = DocumentAgent()
answers = doc_agent.analyze_document(
    "contract.png",
    ["合同金额是多少？", "签约日期是什么时候？", "双方是谁？"]
)
```

### 3. 视频内容分析

```python
import google.generativeai as genai

class VideoAnalysisAgent:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def analyze_video(self, video_path: str, task: str):
        video_file = genai.upload_file(path=video_path)
        
        response = self.model.generate_content([
            video_file,
            task
        ])
        return response.text
    
    def summarize_video(self, video_path: str):
        return self.analyze_video(video_path, "请总结这个视频的主要内容")
    
    def extract_keyframes_description(self, video_path: str):
        return self.analyze_video(video_path, "描述视频中的关键场景和变化")


video_agent = VideoAnalysisAgent()
summary = video_agent.summarize_video("meeting.mp4")
```

### 4. 多模态搜索 Agent

```python
class MultimodalSearchAgent:
    def __init__(self):
        self.client = OpenAI()
        self.search_tool = self._init_search()
    
    def search_by_image(self, image_url: str, query: str):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个搜索助手，根据图片内容帮助用户找到相关信息。"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            tools=[self.search_tool]
        )
        
        if response.choices[0].message.tool_calls:
            return self._execute_search(response.choices[0].message.tool_calls[0])
        return response.choices[0].message.content
    
    def identify_and_search(self, image_url: str):
        query = "识别图片中的物品，并提供购买链接或相关信息"
        return self.search_by_image(image_url, query)
```

## 本地部署多模态模型

### 使用 Ollama 部署 LLaVA

```bash
# 安装 Ollama
# 访问 https://ollama.ai 下载

# 拉取 LLaVA 模型
ollama pull llava

# 运行
ollama run llava
```

```python
import requests

def local_vision_query(image_path: str, question: str):
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llava",
            "prompt": question,
            "images": [image_base64]
        }
    )
    return response.json()["response"]


result = local_vision_query("photo.jpg", "这张图片里有什么？")
```

### 使用 vLLM 部署

```bash
# 安装 vLLM
pip install vllm

# 启动服务
vllm serve llava-hf/llava-1.5-7b-hf
```

## 多模态 Agent 框架

### LangChain 多模态

```python
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")

message = HumanMessage(
    content=[
        {"type": "text", "text": "描述这张图片"},
        {"type": "image_url", "image_url": {"url": "image_url"}}
    ]
)

response = model.invoke([message])
```

### LlamaIndex 多模态

```python
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageDocument

mm_llm = OpenAIMultiModal(model="gpt-4o", max_new_tokens=1000)

image_doc = ImageDocument(image_path="image.jpg")
response = mm_llm.complete("描述这张图片", image_documents=[image_doc])
```

## 最佳实践

### 1. 图像预处理

```python
from PIL import Image
import io
import base64

def optimize_image(image_path: str, max_size: int = 1024):
    img = Image.open(image_path)
    
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    
    return base64.b64encode(buffer.getvalue()).decode()
```

### 2. 批量处理

```python
import asyncio
from openai import AsyncOpenAI

async def process_images_async(image_urls: list[str], prompt: str):
    client = AsyncOpenAI()
    
    async def process_single(url: str):
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": url}}
                    ]
                }
            ]
        )
        return response.choices[0].message.content
    
    tasks = [process_single(url) for url in image_urls]
    return await asyncio.gather(*tasks)


results = asyncio.run(process_images_async(
    ["img1.jpg", "img2.jpg", "img3.jpg"],
    "分类这张图片的类型"
))
```

### 3. 错误处理

```python
class RobustVisionAgent:
    def __init__(self, models: list[str] = None):
        self.models = models or ["gpt-4o", "gemini-2.0-flash", "claude-sonnet-4-20250514"]
        self.clients = self._init_clients()
    
    def analyze_with_fallback(self, image_url: str, prompt: str):
        for model in self.models:
            try:
                return self._try_model(model, image_url, prompt)
            except Exception as e:
                print(f"{model} 失败: {e}")
                continue
        raise Exception("所有模型都失败了")
    
    def _try_model(self, model: str, image_url: str, prompt: str):
        if model.startswith("gpt"):
            return self._call_openai(model, image_url, prompt)
        elif model.startswith("gemini"):
            return self._call_gemini(model, image_url, prompt)
        elif model.startswith("claude"):
            return self._call_claude(model, image_url, prompt)
```

## 成本优化

| 策略 | 说明 |
|------|------|
| 使用 Flash/Mini 模型 | GPT-4o-mini 比 GPT-4o 便宜 10x+ |
| 图像压缩 | 减小图像尺寸可降低 token 消耗 |
| 批量处理 | 合并多个请求减少 API 调用 |
| 本地模型 | 完全免费，适合隐私敏感场景 |
| 缓存结果 | 相同图像避免重复处理 |

## 常见问题

### Q: Flash 模型和 Pro 模型怎么选？

**选 Flash：** 实时应用、批量处理、成本敏感场景
**选 Pro：** 复杂推理、高质量输出、准确性要求高

### Q: 多模态 Agent 的 token 怎么算？

图像会被转换为 token，通常：
- 低分辨率图：约 85 tokens
- 高分辨率图：按 512x512 分块，每块约 170 tokens

### Q: 如何处理超长视频？

1. 使用 Gemini（支持长视频）
2. 预处理提取关键帧
3. 分段处理后汇总

## 相关资源

- [Gemini API 文档](https://ai.google.dev/docs)
- [OpenAI Vision API](https://platform.openai.com/docs/guides/vision)
- [LLaVA 项目](https://github.com/haotian-liu/LLaVA)
- [Ollama](https://ollama.ai)
