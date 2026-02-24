# Agent 案例

这里提供几个从简单到复杂的 AI Agent 案例，帮助你理解 Agent 的实际应用。

## 案例 1：简单问答 Agent

最基础的 Agent，只使用大模型进行对话。

```python
from openai import OpenAI

class SimpleChatAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.messages = []
    
    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=self.messages
        )
        
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message

agent = SimpleChatAgent(api_key="your-api-key")
print(agent.chat("你好，介绍一下你自己"))
print(agent.chat("我刚才问了什么？"))
```

## 案例 2：工具调用 Agent

能够使用工具的 Agent。

```python
from openai import OpenAI
import json
import requests

class ToolAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.tools = {
            "get_weather": self.get_weather,
            "calculate": self.calculate
        }
    
    def get_weather(self, city: str) -> str:
        weather_data = {
            "北京": "晴天，25°C",
            "上海": "多云，28°C",
            "广州": "小雨，30°C"
        }
        return weather_data.get(city, "未知城市")
    
    def calculate(self, expression: str) -> str:
        try:
            result = eval(expression)
            return str(result)
        except:
            return "计算错误"
    
    def run(self, user_input: str) -> str:
        messages = [
            {"role": "system", "content": f"""
            你是一个助手，可以使用以下工具：
            - get_weather(city): 获取城市天气
            - calculate(expression): 计算数学表达式
            
            如果需要使用工具，请按以下格式回复：
            TOOL: 工具名
            INPUT: 输入参数
            """},
            {"role": "user", "content": user_input}
        ]
        
        while True:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            
            content = response.choices[0].message.content
            messages.append({"role": "assistant", "content": content})
            
            if "TOOL:" in content:
                lines = content.split("\n")
                tool_name = lines[0].split(":")[1].strip()
                tool_input = lines[1].split(":")[1].strip()
                
                result = self.tools[tool_name](tool_input)
                messages.append({"role": "user", "content": f"工具结果: {result}"})
            else:
                return content

agent = ToolAgent(api_key="your-api-key")
print(agent.run("北京今天天气怎么样？"))
print(agent.run("帮我计算 123 * 456"))
```

## 案例 3：ReAct Agent

使用 ReAct（Reasoning + Acting）模式的 Agent。

```python
from openai import OpenAI
import re

class ReActAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.tools = {
            "search": self.search,
            "calculate": self.calculate
        }
    
    def search(self, query: str) -> str:
        return f"搜索结果: 关于'{query}'的相关信息..."
    
    def calculate(self, expression: str) -> str:
        try:
            return str(eval(expression))
        except:
            return "计算错误"
    
    def run(self, question: str) -> str:
        prompt = f"""
        回答以下问题，使用 ReAct 格式：
        
        Question: {question}
        
        按以下格式思考：
        Thought: 你的思考过程
        Action: 工具名称[工具参数]
        Observation: 工具返回结果
        ... (重复 Thought/Action/Observation 直到得出答案)
        Thought: 我现在知道答案了
        Answer: 最终答案
        
        可用工具: search, calculate
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content

agent = ReActAgent(api_key="your-api-key")
print(agent.run("2024年世界杯在哪里举办？"))
```

## 案例 4：任务规划 Agent

能够分解任务并逐步执行的 Agent。

```python
from openai import OpenAI
import json

class PlanningAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def plan(self, goal: str) -> list:
        prompt = f"""
        目标: {goal}
        
        请将这个目标分解为具体的步骤，以 JSON 数组格式返回：
        ["步骤1", "步骤2", "步骤3", ...]
        
        只返回 JSON 数组，不要其他内容。
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return []
    
    def execute_step(self, step: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"执行这个步骤: {step}"}]
        )
        return response.choices[0].message.content
    
    def run(self, goal: str):
        print(f"目标: {goal}\n")
        
        steps = self.plan(goal)
        print("计划步骤:")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
        print()
        
        results = []
        for i, step in enumerate(steps, 1):
            print(f"执行步骤 {i}: {step}")
            result = self.execute_step(step)
            results.append(result)
            print(f"结果: {result[:100]}...\n")
        
        return results

agent = PlanningAgent(api_key="your-api-key")
agent.run("写一篇关于 AI Agent 的文章")
```

## 案例 5：多 Agent 协作

多个 Agent 协作完成任务。

```python
from openai import OpenAI

class Agent:
    def __init__(self, name: str, role: str, api_key: str):
        self.name = name
        self.role = role
        self.client = OpenAI(api_key=api_key)
    
    def work(self, task: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"你是{self.role}"},
                {"role": "user", "content": task}
            ]
        )
        return response.choices[0].message.content

class MultiAgentSystem:
    def __init__(self, api_key: str):
        self.agents = {
            "researcher": Agent("研究员", "一个专业的研究员，擅长收集和分析信息", api_key),
            "writer": Agent("作者", "一个专业的作者，擅长撰写文章", api_key),
            "editor": Agent("编辑", "一个专业的编辑，擅长审核和优化内容", api_key)
        }
    
    def run(self, topic: str):
        print(f"主题: {topic}\n")
        
        print("=== 研究员收集信息 ===")
        research = self.agents["researcher"].work(f"收集关于'{topic}'的关键信息")
        print(f"{research[:200]}...\n")
        
        print("=== 作者撰写文章 ===")
        article = self.agents["writer"].work(f"基于以下信息撰写文章:\n{research}")
        print(f"{article[:200]}...\n")
        
        print("=== 编辑审核优化 ===")
        final = self.agents["editor"].work(f"审核并优化以下文章:\n{article}")
        print(f"{final[:200]}...\n")
        
        return final

system = MultiAgentSystem(api_key="your-api-key")
system.run("人工智能的未来发展")
```

## 案例 6：LangChain Agent

使用 LangChain 框架构建 Agent。

```python
from langchain_openai import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import os

os.environ["OPENAI_API_KEY"] = "your-api-key"

def get_word_length(word: str) -> str:
    return str(len(word))

def multiply(a: str) -> str:
    a, b = map(int, a.split(","))
    return str(a * b)

tools = [
    Tool(
        name="word_length",
        func=get_word_length,
        description="计算单词长度，输入一个单词"
    ),
    Tool(
        name="multiply",
        func=multiply,
        description="两数相乘，输入格式: '数字1,数字2'"
    )
]

llm = OpenAI(temperature=0)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("单词 'artificial' 有多少个字母？")
print(result)
```

## 案例 7：自主 Agent

能够自我反思和改进的 Agent。

```python
from openai import OpenAI

class SelfReflectingAgent:
    def __init__(self, api_key: str, max_iterations: int = 3):
        self.client = OpenAI(api_key=api_key)
        self.max_iterations = max_iterations
    
    def generate(self, task: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": task}]
        )
        return response.choices[0].message.content
    
    def reflect(self, task: str, result: str) -> str:
        prompt = f"""
        任务: {task}
        
        当前结果: {result}
        
        请评估这个结果：
        1. 是否完全解决了任务？
        2. 有什么可以改进的地方？
        3. 给出改进建议或说"结果已经很好"
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def improve(self, task: str, result: str, feedback: str) -> str:
        prompt = f"""
        任务: {task}
        
        当前结果: {result}
        
        改进建议: {feedback}
        
        请根据建议改进结果：
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def run(self, task: str) -> str:
        result = self.generate(task)
        
        for i in range(self.max_iterations):
            feedback = self.reflect(task, result)
            
            if "结果已经很好" in feedback:
                print(f"第 {i+1} 次迭代: 结果满意")
                break
            
            print(f"第 {i+1} 次迭代: 发现改进空间")
            result = self.improve(task, result, feedback)
        
        return result

agent = SelfReflectingAgent(api_key="your-api-key")
result = agent.run("写一首关于 AI 的短诗")
print(f"\n最终结果:\n{result}")
```

## 总结

| 案例 | 复杂度 | 特点 |
|------|--------|------|
| 简单问答 | ⭐ | 基础对话 |
| 工具调用 | ⭐⭐ | 使用工具 |
| ReAct | ⭐⭐⭐ | 推理+行动 |
| 任务规划 | ⭐⭐⭐ | 分解任务 |
| 多Agent协作 | ⭐⭐⭐⭐ | 角色分工 |
| LangChain | ⭐⭐⭐ | 框架支持 |
| 自主Agent | ⭐⭐⭐⭐⭐ | 自我反思 |

选择合适的 Agent 类型取决于你的具体需求！
