# 什么是 AI Agent（AI 代理）

## 简单定义

AI Agent（人工智能代理）是一个能够**自主感知环境、做出决策、执行行动**的 AI 系统。它不仅仅是回答问题，还能主动使用工具、规划任务、完成复杂目标。

## 核心区别

### 普通 AI vs AI Agent

```
普通 AI：
用户提问 → AI 回答 → 结束

AI Agent：
用户目标 → AI 理解 → 规划步骤 → 使用工具 → 执行任务 → 检查结果 → 完成目标
```

### 形象比喻

| 类型 | 比喻 | 能力 |
|------|------|------|
| 普通 AI | 百科全书 | 只能回答问题 |
| AI Agent | 智能助手 | 能帮你做事 |

**例子**：
- 普通 AI：「今天北京天气怎么样？」→ AI 回答天气信息
- AI Agent：「帮我规划一个北京三日游」→ AI 查天气、订酒店、安排行程、生成攻略

## Agent 的核心组成

```
┌─────────────────────────────────────────┐
│              AI Agent                   │
│  ┌─────────────────────────────────┐    │
│  │         大模型 (LLM)            │    │
│  │      (大脑/决策中心)            │    │
│  └─────────────────────────────────┘    │
│              ↓                          │
│  ┌─────────────────────────────────┐    │
│  │          规划器                  │    │
│  │    (分解任务、制定计划)          │    │
│  └─────────────────────────────────┘    │
│              ↓                          │
│  ┌─────────────────────────────────┐    │
│  │          记忆系统                │    │
│  │    (存储对话、知识、历史)        │    │
│  └─────────────────────────────────┘    │
│              ↓                          │
│  ┌─────────────────────────────────┐    │
│  │          工具集                  │    │
│  │  (搜索、代码执行、API调用等)     │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

### 1. 大模型（LLM）- 大脑

负责理解、推理和决策：

```python
def think(self, task):
    reasoning = self.llm.generate(f"如何完成这个任务：{task}")
    return reasoning
```

### 2. 规划器 - 计划

将复杂任务分解为小步骤：

```python
def plan(self, goal):
    steps = [
        "1. 理解用户需求",
        "2. 收集必要信息",
        "3. 执行核心任务",
        "4. 验证结果",
        "5. 返回答案"
    ]
    return steps
```

### 3. 记忆系统 - 记忆

存储历史信息和上下文：

```python
class Memory:
    def __init__(self):
        self.short_term = []  # 短期记忆（当前对话）
        self.long_term = []   # 长期记忆（历史知识）
    
    def add(self, message):
        self.short_term.append(message)
    
    def recall(self, query):
        return self.search_long_term(query)
```

### 4. 工具集 - 能力

让 Agent 能够执行实际操作：

```python
tools = {
    "search": search_web,        # 搜索
    "calculator": calculate,      # 计算
    "code_executor": run_code,    # 执行代码
    "weather_api": get_weather,   # 获取天气
    "send_email": send_email      # 发送邮件
}
```

## Agent 工作流程

### ReAct 模式（推理+行动）

```
用户：北京今天天气怎么样？适合户外运动吗？

Agent 思考过程：
1. [Thought] 需要获取北京今天的天气信息
2. [Action] 调用天气 API
3. [Observation] 北京今天晴天，气温 25°C，空气质量良好
4. [Thought] 根据天气信息判断是否适合户外运动
5. [Action] 给出建议
6. [Answer] 北京今天天气很好，适合户外运动...
```

### 代码实现示例

```python
class SimpleAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
    
    def run(self, user_input):
        while True:
            prompt = f"""
            用户输入: {user_input}
            
            你可以使用以下工具:
            {list(self.tools.keys())}
            
            请按以下格式思考和行动:
            Thought: [你的思考]
            Action: [工具名称]
            Action Input: [工具输入]
            """
            
            response = self.llm.generate(prompt)
            
            if "Final Answer:" in response:
                return self.extract_answer(response)
            
            action, action_input = self.parse_action(response)
            observation = self.tools[action](action_input)
            
            user_input += f"\n观察结果: {observation}"

agent = SimpleAgent(
    llm=OpenAI(),
    tools={"search": search_web, "weather": get_weather}
)

result = agent.run("北京今天天气怎么样？")
```

## Agent 的类型

### 1. 对话型 Agent

专注于自然对话，如 ChatGPT：

```
特点：多轮对话、上下文理解
应用：客服、聊天机器人
```

### 2. 任务型 Agent

专注于完成特定任务：

```
特点：目标导向、使用工具
应用：预订系统、数据查询
```

### 3. 自主型 Agent

能够独立完成复杂任务：

```
特点：自我规划、自我反思
应用：AutoGPT、BabyAGI
```

### 4. 多 Agent 系统

多个 Agent 协作完成任务：

```
特点：角色分工、协作完成
应用：软件开发团队模拟、复杂问题解决
```

## 主流 Agent 框架

### LangChain

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import OpenAI

llm = OpenAI()

tools = [
    Tool(name="Search", func=search, description="搜索网络")
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
agent.run("今天新闻有什么？")
```

### AutoGPT

```python
# AutoGPT 是一个自主 Agent 框架
# 能够自动分解目标、执行任务、自我反思
# 无需人工干预即可完成复杂目标
```

### CrewAI

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="研究员",
    goal="收集信息",
    backstory="你是一个专业研究员"
)

writer = Agent(
    role="作者",
    goal="撰写文章",
    backstory="你是一个专业作者"
)

crew = Crew(agents=[researcher, writer], tasks=[...])
crew.kickoff()
```

## Agent 的能力边界

### 能做什么

- ✅ 信息检索和整理
- ✅ 代码生成和执行
- ✅ 多步骤任务规划
- ✅ 工具调用和 API 集成
- ✅ 数据分析和报告

### 不能做什么

- ❌ 访问未授权的系统
- ❌ 执行物理操作（除非有机器人）
- ❌ 100% 准确（可能出错）
- ❌ 完全替代人类判断

## 为什么 Agent 是未来？

1. **从被动到主动**：不只是回答问题，而是解决问题
2. **从单一到多元**：整合多种工具和能力
3. **从简单到复杂**：处理多步骤、多领域任务
4. **从人工到自动**：减少人工干预，提高效率

## 下一步

了解了 AI Agent 后，来看看实际的案例：
→ [04-Agent案例](../04-Agent案例/)
