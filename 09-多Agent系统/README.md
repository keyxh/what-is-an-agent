# 多 Agent 系统

## 什么是多 Agent 系统？

多 Agent 系统（Multi-Agent System，MAS）是由多个 Agent 协作完成复杂任务的系统。每个 Agent 扮演不同角色，分工合作。

```
单 Agent：一个人完成所有任务
多 Agent：团队协作，各司其职
```

## 为什么需要多 Agent？

### 单 Agent 的局限

```
1. 能力有限：一个 Agent 难以精通所有领域
2. 负载过重：所有任务集中处理
3. 缺乏验证：没有其他 Agent 审核结果
4. 效率瓶颈：串行处理，无法并行
```

### 多 Agent 的优势

```
1. 专业分工：每个 Agent 专注自己的领域
2. 并行处理：多个 Agent 同时工作
3. 相互验证：Agent 之间可以互相审核
4. 灵活扩展：可以随时添加新 Agent
```

## 多 Agent 架构模式

### 1. 层级架构

```
        ┌─────────────┐
        │  主控 Agent │
        └──────┬──────┘
               │
       ┌───────┼───────┐
       ↓       ↓       ↓
   ┌───────┐ ┌───────┐ ┌───────┐
   │Agent A│ │Agent B│ │Agent C│
   └───────┘ └───────┘ └───────┘
```

```python
class HierarchicalSystem:
    def __init__(self, llm):
        self.llm = llm
        self.master = Agent("主管", "协调和分配任务")
        self.workers = [
            Agent("研究员", "收集和分析信息"),
            Agent("作者", "撰写内容"),
            Agent("编辑", "审核和优化")
        ]
    
    def run(self, task: str):
        subtasks = self.master.plan(task)
        
        results = []
        for subtask in subtasks:
            for worker in self.workers:
                if worker.can_handle(subtask):
                    result = worker.execute(subtask)
                    results.append(result)
                    break
        
        return self.master.integrate(results)
```

### 2. 对等架构

```
┌───────┐     ┌───────┐
│Agent A│ ←→ │Agent B│
└───┬───┘     └───┬───┘
    │             │
    ↓             ↓
┌───────┐     ┌───────┐
│Agent C│ ←→ │Agent D│
└───────┘     └───────┘
```

```python
class PeerToPeerSystem:
    def __init__(self):
        self.agents = {}
        self.message_queue = []
    
    def add_agent(self, name: str, agent: Agent):
        self.agents[name] = agent
    
    def send_message(self, from_agent: str, to_agent: str, content: str):
        self.message_queue.append({
            "from": from_agent,
            "to": to_agent,
            "content": content
        })
    
    def process_messages(self):
        while self.message_queue:
            msg = self.message_queue.pop(0)
            recipient = self.agents[msg["to"]]
            response = recipient.receive(msg)
            
            if response.get("forward"):
                self.send_message(
                    msg["to"],
                    response["forward_to"],
                    response["content"]
                )
```

### 3. 群组架构

```
        ┌─────────────────┐
        │   共享消息池    │
        └────────┬────────┘
                 │
    ┌────────────┼────────────┐
    ↓            ↓            ↓
┌───────┐   ┌───────┐   ┌───────┐
│Agent A│   │Agent B│   │Agent C│
└───────┘   └───────┘   └───────┘
```

```python
class GroupChatSystem:
    def __init__(self):
        self.agents = []
        self.shared_memory = []
    
    def add_agent(self, agent: Agent):
        self.agents.append(agent)
    
    def broadcast(self, message: str, sender: str):
        self.shared_memory.append({
            "sender": sender,
            "content": message
        })
        
        for agent in self.agents:
            if agent.name != sender:
                agent.receive_broadcast(message, sender)
    
    def run_discussion(self, topic: str, max_rounds: int = 10):
        self.broadcast(f"讨论主题：{topic}", "system")
        
        for round_num in range(max_rounds):
            for agent in self.agents:
                response = agent.think(self.shared_memory)
                if response:
                    self.broadcast(response, agent.name)
```

## Agent 角色设计

### 角色定义模板

```python
class AgentRole:
    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        backstory: str,
        skills: list,
        tools: list = None
    ):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.skills = skills
        self.tools = tools or []
    
    def get_system_prompt(self) -> str:
        return f"""
# 角色信息
- 名称：{self.name}
- 角色：{self.role}
- 目标：{self.goal}
- 背景：{self.backstory}

# 技能
{chr(10).join(f'- {s}' for s in self.skills)}

# 可用工具
{chr(10).join(f'- {t}' for t in self.tools) if self.tools else '无'}
"""
```

### 常见角色类型

```python
ROLES = {
    "researcher": AgentRole(
        name="研究员",
        role="信息收集与分析专家",
        goal="收集、分析、整理信息",
        backstory="你是一位经验丰富的研究员，擅长从各种来源收集和分析数据",
        skills=["信息检索", "数据分析", "报告撰写"],
        tools=["search", "database_query"]
    ),
    
    "writer": AgentRole(
        name="作者",
        role="内容创作专家",
        goal="创作高质量的内容",
        backstory="你是一位专业作家，擅长将复杂信息转化为易懂的内容",
        skills=["写作", "编辑", "创意思维"],
        tools=["text_editor", "grammar_check"]
    ),
    
    "coder": AgentRole(
        name="程序员",
        role="软件开发专家",
        goal="编写高质量代码",
        backstory="你是一位资深程序员，精通多种编程语言和最佳实践",
        skills=["编程", "调试", "代码审查"],
        tools=["code_editor", "terminal", "git"]
    ),
    
    "reviewer": AgentRole(
        name="审核员",
        role="质量保证专家",
        goal="确保输出质量",
        backstory="你是一位严谨的审核员，擅长发现问题并提出改进建议",
        skills=["审核", "测试", "反馈"],
        tools=["checklist", "test_runner"]
    ),
    
    "coordinator": AgentRole(
        name="协调员",
        role="任务协调专家",
        goal="协调团队高效完成任务",
        backstory="你是一位经验丰富的项目经理，擅长协调团队和分配任务",
        skills=["任务分配", "进度跟踪", "冲突解决"],
        tools=["task_manager", "calendar"]
    )
}
```

## 协作模式

### 1. 顺序协作

```python
class SequentialCollaboration:
    def __init__(self, agents: list):
        self.agents = agents
    
    def run(self, task: str):
        result = task
        
        for agent in self.agents:
            result = agent.process(result)
        
        return result

workflow = SequentialCollaboration([
    researcher,
    writer,
    reviewer
])

final_result = workflow.run("写一篇关于 AI 的文章")
```

### 2. 并行协作

```python
import concurrent.futures

class ParallelCollaboration:
    def __init__(self, agents: list):
        self.agents = agents
    
    def run(self, task: str):
        results = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(agent.process, task)
                for agent in self.agents
            ]
            
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        return results

parallel = ParallelCollaboration([
    researcher_a,
    researcher_b,
    researcher_c
])

all_research = parallel.run("研究 AI 发展趋势")
```

### 3. 辩论协作

```python
class DebateCollaboration:
    def __init__(self, agents: list, judge: Agent):
        self.agents = agents
        self.judge = judge
    
    def run(self, topic: str, rounds: int = 3):
        arguments = []
        
        for round_num in range(rounds):
            for agent in self.agents:
                argument = agent.argue(topic, arguments)
                arguments.append({
                    "agent": agent.name,
                    "round": round_num,
                    "argument": argument
                })
        
        return self.judge.decide(topic, arguments)

debate = DebateCollaboration(
    agents=[proponent, opponent],
    judge=judge
)

verdict = debate.run("AI 是否会取代人类工作？")
```

### 4. 投票协作

```python
class VotingCollaboration:
    def __init__(self, agents: list):
        self.agents = agents
    
    def run(self, question: str) -> str:
        votes = {}
        
        for agent in self.agents:
            answer = agent.answer(question)
            votes[answer] = votes.get(answer, 0) + 1
        
        return max(votes, key=votes.get)

voting = VotingCollaboration([
    agent_a, agent_b, agent_c, agent_d, agent_e
])

consensus = voting.run("这个方案是否可行？")
```

## 完整示例：软件开发团队

```python
from openai import OpenAI
from typing import List, Dict
import json

class Agent:
    def __init__(self, name: str, role: str, api_key: str):
        self.name = name
        self.role = role
        self.client = OpenAI(api_key=api_key)
        self.memory = []
    
    def think(self, context: str) -> str:
        messages = [
            {"role": "system", "content": f"你是{self.role}"},
            {"role": "user", "content": context}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        return response.choices[0].message.content
    
    def receive(self, message: Dict):
        self.memory.append(message)

class SoftwareTeam:
    def __init__(self, api_key: str):
        self.product_manager = Agent("产品经理", "产品经理，负责需求分析和产品规划", api_key)
        self.architect = Agent("架构师", "软件架构师，负责系统设计", api_key)
        self.developer = Agent("开发工程师", "开发工程师，负责编码实现", api_key)
        self.tester = Agent("测试工程师", "测试工程师，负责质量保证", api_key)
        
        self.team = [
            self.product_manager,
            self.architect,
            self.developer,
            self.tester
        ]
        
        self.shared_context = []
    
    def develop(self, requirement: str) -> Dict:
        print(f"=== 项目开始：{requirement} ===\n")
        
        print("【产品经理】分析需求...")
        requirements_doc = self.product_manager.think(
            f"分析以下需求，输出需求文档：\n{requirement}"
        )
        self.shared_context.append({"role": "产品经理", "content": requirements_doc})
        print(f"需求文档：{requirements_doc[:100]}...\n")
        
        print("【架构师】设计系统...")
        design_doc = self.architect.think(
            f"根据以下需求设计系统架构：\n{requirements_doc}"
        )
        self.shared_context.append({"role": "架构师", "content": design_doc})
        print(f"设计文档：{design_doc[:100]}...\n")
        
        print("【开发工程师】编写代码...")
        code = self.developer.think(
            f"根据以下设计编写代码：\n{design_doc}"
        )
        self.shared_context.append({"role": "开发工程师", "content": code})
        print(f"代码：{code[:100]}...\n")
        
        print("【测试工程师】测试代码...")
        test_report = self.tester.think(
            f"测试以下代码并输出测试报告：\n{code}"
        )
        self.shared_context.append({"role": "测试工程师", "content": test_report})
        print(f"测试报告：{test_report[:100]}...\n")
        
        return {
            "requirements": requirements_doc,
            "design": design_doc,
            "code": code,
            "test_report": test_report
        }

team = SoftwareTeam(api_key="your-api-key")
result = team.develop("开发一个简单的待办事项应用")
```

## 通信机制

### 1. 直接消息

```python
class DirectMessaging:
    def __init__(self):
        self.agents = {}
    
    def register(self, agent: Agent):
        self.agents[agent.name] = agent
    
    def send(self, from_name: str, to_name: str, content: str):
        if to_name in self.agents:
            self.agents[to_name].receive({
                "from": from_name,
                "content": content
            })
```

### 2. 广播消息

```python
class BroadcastMessaging:
    def __init__(self):
        self.agents = []
    
    def register(self, agent: Agent):
        self.agents.append(agent)
    
    def broadcast(self, from_name: str, content: str):
        for agent in self.agents:
            if agent.name != from_name:
                agent.receive({
                    "from": from_name,
                    "content": content,
                    "type": "broadcast"
                })
```

### 3. 发布订阅

```python
class PubSubMessaging:
    def __init__(self):
        self.topics = {}
    
    def subscribe(self, topic: str, agent: Agent):
        if topic not in self.topics:
            self.topics[topic] = []
        self.topics[topic].append(agent)
    
    def publish(self, topic: str, content: str):
        if topic in self.topics:
            for agent in self.topics[topic]:
                agent.receive({
                    "topic": topic,
                    "content": content
                })
```

## 最佳实践

| 实践 | 说明 |
|------|------|
| 明确分工 | 每个 Agent 有清晰的职责边界 |
| 定义接口 | Agent 之间的通信格式标准化 |
| 设置超时 | 防止单个 Agent 阻塞整个系统 |
| 错误隔离 | 一个 Agent 失败不影响其他 |
| 结果验证 | 重要结果需要多个 Agent 验证 |
| 日志记录 | 记录 Agent 间的通信和决策 |

## 下一步

学会了多 Agent 系统后，看看实际的 Agent 项目案例：
→ [10-Agent实战项目](../10-Agent实战项目/)
