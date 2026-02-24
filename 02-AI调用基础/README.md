# AI 调用基础

## 什么是 AI API 调用？

AI API 调用就是通过编程方式向大模型发送请求，获取模型的响应。类似于调用一个函数，输入问题，输出答案。

## 基本流程

```
你的程序 → 发送请求 → AI 服务端 → 处理请求 → 返回结果 → 你的程序
```

## 主流 AI API

### OpenAI API

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": "你好"}
    ]
)

print(response.choices[0].message.content)
```

### Claude API

```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "你好"}
    ]
)

print(message.content[0].text)
```

### 国内模型 API

```python
from dashscope import Generation

response = Generation.call(
    model='qwen-turbo',
    messages=[
        {'role': 'user', 'content': '你好'}
    ]
)

print(response.output.choices[0].message.content)
```

## 核心概念

### 1. Messages（消息）

消息是对话的基本单位，包含角色和内容：

```python
messages = [
    {"role": "system", "content": "你是一个助手"},  # 系统提示
    {"role": "user", "content": "你好"},           # 用户消息
    {"role": "assistant", "content": "你好！"},    # AI 回复
    {"role": "user", "content": "今天天气怎么样"}  # 继续对话
]
```

| 角色 | 说明 |
|------|------|
| system | 设定 AI 的行为和角色 |
| user | 用户的输入 |
| assistant | AI 的回复 |

### 2. Temperature（温度）

控制输出的随机性，范围 0-2：

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    temperature=0.7  # 默认值
)
```

| Temperature 值 | 效果 |
|----------------|------|
| 0 | 最确定性，重复性高 |
| 0.7 | 平衡，适合对话 |
| 1.0+ | 更有创造性，多样性 |

### 3. Max Tokens（最大令牌数）

限制输出的长度：

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    max_tokens=500  # 限制输出最多 500 个 token
)
```

### 4. Stream（流式输出）

逐字输出，提升用户体验：

```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    stream=True  # 启用流式输出
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## 完整示例

### 基础对话

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

def chat(user_input: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "你是一个友好的助手"},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message.content

answer = chat("什么是 Python？")
print(answer)
```

### 多轮对话

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

class ChatSession:
    def __init__(self):
        self.messages = [
            {"role": "system", "content": "你是一个友好的助手"}
        ]
    
    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=self.messages
        )
        
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message

session = ChatSession()
print(session.chat("你好"))
print(session.chat("我刚才说了什么？"))  # AI 会记住上下文
```

## 费用计算

### Token 计费

API 调用通常按 Token 计费：

```
费用 = (输入 Token 数 × 输入单价) + (输出 Token 数 × 输出单价)
```

### 示例价格（仅供参考）

| 模型 | 输入价格 | 输出价格 |
|------|----------|----------|
| GPT-4 | $0.03/1K tokens | $0.06/1K tokens |
| GPT-3.5-turbo | $0.001/1K tokens | $0.002/1K tokens |
| Claude-3 Opus | $0.015/1K tokens | $0.075/1K tokens |

## 错误处理

```python
from openai import OpenAI, APIError, RateLimitError

client = OpenAI(api_key="your-api-key")

try:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "你好"}]
    )
except RateLimitError:
    print("请求太频繁，请稍后再试")
except APIError as e:
    print(f"API 错误: {e}")
except Exception as e:
    print(f"其他错误: {e}")
```

## 最佳实践

1. **保护 API Key**：不要硬编码，使用环境变量
   ```python
   import os
   client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
   ```

2. **控制成本**：设置 max_tokens 限制输出长度

3. **处理超时**：设置合理的超时时间
   ```python
   client = OpenAI(timeout=30.0)
   ```

4. **重试机制**：网络问题自动重试
   ```python
   from tenacity import retry, stop_after_attempt
   
   @retry(stop=stop_after_attempt(3))
   def call_api():
       ...
   ```

## 下一步

学会了 AI API 调用后，接下来学习什么是 AI Agent：
→ [03-什么是AI代理](../03-什么是AI代理/)
