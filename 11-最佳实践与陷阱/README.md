# 最佳实践与陷阱

## 设计原则

### 1. 单一职责原则

每个 Agent 应该有明确的、单一的职责。

```python
# 好的设计
researcher = Agent("研究员", "收集和分析信息")
writer = Agent("作者", "撰写内容")
editor = Agent("编辑", "审核和优化")

# 不好的设计
super_agent = Agent("超级助手", "做所有事情")
```

### 2. 明确的接口定义

Agent 之间的通信应该有清晰的格式。

```python
class AgentMessage:
    def __init__(
        self,
        sender: str,
        receiver: str,
        content: str,
        message_type: str,
        metadata: dict = None
    ):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.message_type = message_type
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> dict:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "type": self.message_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
```

### 3. 错误隔离

一个 Agent 的失败不应该导致整个系统崩溃。

```python
class ResilientAgent:
    def __init__(self, name: str, max_retries: int = 3):
        self.name = name
        self.max_retries = max_retries
    
    def execute(self, task: str) -> dict:
        for attempt in range(self.max_retries):
            try:
                result = self._do_work(task)
                return {"success": True, "result": result}
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {
                        "success": False,
                        "error": str(e),
                        "agent": self.name
                    }
                continue
```

### 4. 可观测性

记录 Agent 的行为，便于调试和优化。

```python
import logging

class ObservableAgent:
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"agent.{name}")
        self.history = []
    
    def log_action(self, action: str, details: dict):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details
        }
        self.history.append(entry)
        self.logger.info(f"[{self.name}] {action}: {details}")
    
    def think(self, prompt: str) -> str:
        self.log_action("thinking", {"prompt": prompt[:100]})
        result = self._call_llm(prompt)
        self.log_action("thought", {"result": result[:100]})
        return result
```

## 常见陷阱

### 陷阱 1：无限循环

Agent 可能陷入无限调用工具的循环。

```python
# 问题代码
def run_agent(self, task):
    while True:
        response = self.llm.call(task)
        if response.needs_tool:
            result = self.use_tool(response.tool)
            task = result  # 可能导致无限循环

# 解决方案
def run_agent(self, task, max_iterations=10):
    for _ in range(max_iterations):
        response = self.llm.call(task)
        if not response.needs_tool:
            return response.content
        result = self.use_tool(response.tool)
        task = result
    return "达到最大迭代次数，请简化任务"
```

### 陷阱 2：上下文溢出

对话历史过长导致超出 token 限制。

```python
# 问题代码
def chat(self, message):
    self.history.append(message)
    return self.llm.call(self.history)  # 可能超出限制

# 解决方案
def chat(self, message, max_tokens=4000):
    self.history.append(message)
    
    while self._count_tokens(self.history) > max_tokens:
        self._summarize_old_messages()
    
    return self.llm.call(self.history)

def _summarize_old_messages(self):
    old_messages = self.history[:-5]
    summary = self.llm.call(f"总结：{old_messages}")
    self.history = [{"role": "system", "content": f"历史摘要：{summary}"}] + self.history[-5:]
```

### 陷阱 3：工具参数错误

LLM 生成的工具参数可能不符合预期。

```python
# 问题代码
def use_tool(self, tool_name, params):
    return self.tools[tool_name](**params)  # 可能参数错误

# 解决方案
def use_tool(self, tool_name, params):
    try:
        tool = self.tools[tool_name]
        validated_params = self._validate_params(tool, params)
        return tool(**validated_params)
    except KeyError:
        return {"error": f"未知工具：{tool_name}"}
    except Exception as e:
        return {"error": f"工具执行失败：{str(e)}"}

def _validate_params(self, tool, params):
    sig = inspect.signature(tool)
    valid_params = {}
    for name, param in sig.parameters.items():
        if name in params:
            valid_params[name] = params[name]
        elif param.default is param.empty:
            raise ValueError(f"缺少必需参数：{name}")
    return valid_params
```

### 陷阱 4：幻觉问题

Agent 可能生成不真实的信息。

```python
# 问题代码
def answer(self, question):
    return self.llm.call(question)  # 可能产生幻觉

# 解决方案
def answer(self, question):
    prompt = f"""
    回答以下问题。如果你不确定答案，请说"我不知道"。
    不要编造信息。
    
    问题：{question}
    """
    return self.llm.call(prompt)

# 更好的方案：使用 RAG
def answer_with_rag(self, question):
    context = self.retriever.search(question)
    prompt = f"""
    根据以下信息回答问题。如果信息中没有答案，请说"我不知道"。
    
    信息：{context}
    问题：{question}
    """
    return self.llm.call(prompt)
```

### 陷阱 5：敏感信息泄露

Agent 可能在输出中泄露敏感信息。

```python
# 问题代码
def process(self, data):
    return self.llm.call(f"处理数据：{data}")  # 可能泄露敏感信息

# 解决方案
class SecureAgent:
    SENSITIVE_PATTERNS = [
        r'\b\d{16}\b',  # 信用卡号
        r'\b\d{17}[\dXx]\b',  # 身份证号
        r'\b[\w\.-]+@[\w\.-]+\.\w+\b',  # 邮箱
    ]
    
    def sanitize(self, text: str) -> str:
        import re
        for pattern in self.SENSITIVE_PATTERNS:
            text = re.sub(pattern, '[已脱敏]', text)
        return text
    
    def process(self, data):
        sanitized_data = self.sanitize(data)
        return self.llm.call(f"处理数据：{sanitized_data}")
```

### 陷阱 6：成本失控

频繁调用 API 导致成本过高。

```python
# 问题代码
def process_batch(self, items):
    return [self.llm.call(item) for item in items]  # 成本高

# 解决方案
class CostControlledAgent:
    def __init__(self, daily_budget: float = 10.0):
        self.daily_budget = daily_budget
        self.current_spend = 0.0
    
    def call_with_budget(self, prompt: str, estimated_cost: float = 0.01):
        if self.current_spend + estimated_cost > self.daily_budget:
            raise Exception("超出每日预算")
        
        result = self.llm.call(prompt)
        self.current_spend += estimated_cost
        return result
    
    def batch_process(self, items, batch_size: int = 10):
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            batch_prompt = f"批量处理以下内容：\n" + "\n".join(batch)
            result = self.call_with_budget(batch_prompt, estimated_cost=0.05)
            results.append(result)
        return results
```

## 性能优化

### 1. 缓存策略

```python
from functools import lru_cache
import hashlib

class CachedAgent:
    def __init__(self):
        self.cache = {}
    
    def _get_cache_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def call(self, prompt: str) -> str:
        cache_key = self._get_cache_key(prompt)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = self.llm.call(prompt)
        self.cache[cache_key] = result
        return result
```

### 2. 并行处理

```python
import concurrent.futures

class ParallelAgent:
    def parallel_process(self, tasks: list) -> list:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.process, task) for task in tasks]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        return results
```

### 3. 流式输出

```python
def stream_response(self, prompt: str):
    stream = self.llm.stream(prompt)
    for chunk in stream:
        yield chunk.content
```

## 测试策略

### 单元测试

```python
import unittest
from unittest.mock import Mock, patch

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agent = Agent("test")
    
    @patch('openai.ChatCompletion.create')
    def test_think(self, mock_create):
        mock_create.return_value = Mock(
            choices=[Mock(message=Mock(content="测试回复"))]
        )
        
        result = self.agent.think("测试问题")
        
        self.assertEqual(result, "测试回复")
    
    def test_tool_execution(self):
        self.agent.tools = {"test_tool": lambda x: f"result: {x}"}
        result = self.agent.use_tool("test_tool", "input")
        self.assertEqual(result, "result: input")
```

### 集成测试

```python
class TestAgentIntegration(unittest.TestCase):
    def test_full_workflow(self):
        agent = ResearchAgent(api_key="test-key")
        
        result = agent.research("AI发展趋势")
        
        self.assertIn("summary", result)
        self.assertIn("sources", result)
        self.assertTrue(len(result["sources"]) > 0)
```

## 部署建议

### 1. 环境变量管理

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
```

### 2. 日志配置

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('agent')
```

### 3. 健康检查

```python
from fastapi import FastAPI
import time

app = FastAPI()
agent = Agent()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "agent_status": agent.status
    }

@app.post("/chat")
async def chat(message: str):
    return {"response": agent.chat(message)}
```

## 检查清单

| 类别 | 检查项 |
|------|--------|
| 安全 | API Key 不硬编码 |
| 安全 | 敏感信息脱敏 |
| 安全 | 输入验证 |
| 性能 | 设置超时 |
| 性能 | 限制迭代次数 |
| 性能 | 使用缓存 |
| 可靠性 | 错误处理 |
| 可靠性 | 重试机制 |
| 可靠性 | 降级策略 |
| 可观测性 | 日志记录 |
| 可观测性 | 指标监控 |
| 可观测性 | 链路追踪 |

## 下一步

了解了最佳实践后，看看 AI Agent 的未来发展趋势：
→ [12-未来展望](../12-未来展望/)
