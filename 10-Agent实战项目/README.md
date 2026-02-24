# Agent 实战项目

这里提供几个完整的 Agent 实战项目，从简单到复杂，帮助你掌握 Agent 开发。

## 项目 1：智能客服 Agent

### 项目描述

一个能够回答常见问题、查询订单、处理退款的智能客服系统。

### 项目结构

```
customer-service-agent/
├── agent.py          # Agent 主程序
├── tools.py          # 工具定义
├── knowledge.py      # 知识库
└── main.py           # 入口文件
```

### 代码实现

```python
from openai import OpenAI
import json
from typing import Dict, List
from datetime import datetime

class CustomerServiceAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.conversation_history = []
        self.order_database = self._init_orders()
        self.faq_database = self._init_faq()
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "query_order",
                    "description": "查询订单状态",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {
                                "type": "string",
                                "description": "订单号"
                            }
                        },
                        "required": ["order_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "process_refund",
                    "description": "处理退款申请",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {
                                "type": "string",
                                "description": "订单号"
                            },
                            "reason": {
                                "type": "string",
                                "description": "退款原因"
                            }
                        },
                        "required": ["order_id", "reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_faq",
                    "description": "搜索常见问题",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "问题关键词"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    def _init_orders(self) -> Dict:
        return {
            "ORD001": {"status": "已发货", "product": "iPhone 15", "amount": 7999},
            "ORD002": {"status": "配送中", "product": "MacBook Pro", "amount": 14999},
            "ORD003": {"status": "已签收", "product": "AirPods Pro", "amount": 1999}
        }
    
    def _init_faq(self) -> List[Dict]:
        return [
            {"question": "如何退货", "answer": "请在订单详情页申请退货，或联系客服处理"},
            {"question": "配送时间", "answer": "一般1-3个工作日送达，偏远地区3-7天"},
            {"question": "支付方式", "answer": "支持支付宝、微信、银行卡等多种支付方式"}
        ]
    
    def query_order(self, order_id: str) -> Dict:
        order = self.order_database.get(order_id)
        if order:
            return {"success": True, "order": order}
        return {"success": False, "message": "订单不存在"}
    
    def process_refund(self, order_id: str, reason: str) -> Dict:
        if order_id in self.order_database:
            return {
                "success": True,
                "message": f"退款申请已提交，预计3-5个工作日到账",
                "refund_id": f"REF{datetime.now().strftime('%Y%m%d%H%M%S')}"
            }
        return {"success": False, "message": "订单不存在"}
    
    def search_faq(self, query: str) -> List[Dict]:
        results = []
        for faq in self.faq_database:
            if query in faq["question"] or query in faq["answer"]:
                results.append(faq)
        return results if results else [{"question": "未找到相关问题", "answer": "请转人工客服"}]
    
    def chat(self, user_input: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_input})
        
        messages = [
            {
                "role": "system",
                "content": """你是一个专业的客服助手。
                - 友好、耐心地回答用户问题
                - 需要查询订单时使用 query_order 工具
                - 需要处理退款时使用 process_refund 工具
                - 查找常见问题时使用 search_faq 工具
                - 无法解决的问题，建议用户转人工客服"""
            }
        ] + self.conversation_history
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                if func_name == "query_order":
                    result = self.query_order(func_args["order_id"])
                elif func_name == "process_refund":
                    result = self.process_refund(
                        func_args["order_id"],
                        func_args["reason"]
                    )
                elif func_name == "search_faq":
                    result = self.search_faq(func_args["query"])
                
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": json.dumps(result, ensure_ascii=False)
                })
            
            final_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是专业的客服助手"},
                    *self.conversation_history
                ]
            )
            
            return final_response.choices[0].message.content
        
        self.conversation_history.append({"role": "assistant", "content": message.content})
        return message.content

agent = CustomerServiceAgent(api_key="your-api-key")
print(agent.chat("你好"))
print(agent.chat("查询订单 ORD001"))
print(agent.chat("我想退款，订单号是 ORD003，原因是商品有瑕疵"))
```

## 项目 2：代码助手 Agent

### 项目描述

一个能够生成代码、解释代码、调试代码的智能编程助手。

### 代码实现

```python
from openai import OpenAI
import json
import subprocess
import tempfile
import os

class CodeAssistantAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.conversation_history = []
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "description": "执行代码并返回结果",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "要执行的代码"},
                            "language": {"type": "string", "description": "编程语言"}
                        },
                        "required": ["code", "language"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_error",
                    "description": "分析错误信息并提供解决方案",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "error_message": {"type": "string", "description": "错误信息"}
                        },
                        "required": ["error_message"]
                    }
                }
            }
        ]
    
    def execute_code(self, code: str, language: str) -> Dict:
        try:
            if language.lower() == "python":
                with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.py', delete=False
                ) as f:
                    f.write(code)
                    temp_path = f.name
                
                result = subprocess.run(
                    ['python', temp_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                os.unlink(temp_path)
                
                if result.returncode == 0:
                    return {"success": True, "output": result.stdout}
                else:
                    return {"success": False, "error": result.stderr}
            
            return {"success": False, "error": f"不支持的语言: {language}"}
        
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "执行超时"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def analyze_error(self, error_message: str) -> Dict:
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个代码调试专家，分析错误并提供解决方案"
                },
                {
                    "role": "user",
                    "content": f"分析以下错误并提供解决方案：\n{error_message}"
                }
            ]
        )
        
        return {
            "success": True,
            "analysis": response.choices[0].message.content
        }
    
    def chat(self, user_input: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_input})
        
        messages = [
            {
                "role": "system",
                "content": """你是一个专业的编程助手。
                - 帮助用户编写、解释、调试代码
                - 需要执行代码时使用 execute_code 工具
                - 遇到错误时使用 analyze_error 工具分析
                - 提供清晰、详细的解释和建议"""
            }
        ] + self.conversation_history
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                if func_name == "execute_code":
                    result = self.execute_code(
                        func_args["code"],
                        func_args["language"]
                    )
                elif func_name == "analyze_error":
                    result = self.analyze_error(func_args["error_message"])
                
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": json.dumps(result, ensure_ascii=False)
                })
            
            final_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是专业的编程助手"},
                    *self.conversation_history
                ]
            )
            
            return final_response.choices[0].message.content
        
        self.conversation_history.append({"role": "assistant", "content": message.content})
        return message.content

agent = CodeAssistantAgent(api_key="your-api-key")
print(agent.chat("写一个计算斐波那契数列的 Python 函数"))
print(agent.chat("执行这个函数，计算第10个数"))
```

## 项目 3：研究助手 Agent

### 项目描述

一个能够搜索信息、整理资料、生成报告的研究助手。

### 代码实现

```python
from openai import OpenAI
import json
from typing import List, Dict
from datetime import datetime

class ResearchAssistantAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.research_notes = []
        self.sources = []
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "搜索网络信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索关键词"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "take_note",
                    "description": "记录研究笔记",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "笔记内容"},
                            "category": {"type": "string", "description": "分类"}
                        },
                        "required": ["content", "category"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_report",
                    "description": "生成研究报告",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "报告标题"},
                            "format": {"type": "string", "description": "报告格式"}
                        },
                        "required": ["title"]
                    }
                }
            }
        ]
    
    def search_web(self, query: str) -> Dict:
        mock_results = [
            {
                "title": f"关于{query}的研究报告",
                "snippet": f"这是关于{query}的详细信息...",
                "url": f"https://example.com/{query}"
            },
            {
                "title": f"{query}的发展趋势",
                "snippet": f"{query}领域最新发展...",
                "url": f"https://example.com/{query}/trends"
            }
        ]
        
        self.sources.extend(mock_results)
        
        return {
            "success": True,
            "results": mock_results,
            "count": len(mock_results)
        }
    
    def take_note(self, content: str, category: str) -> Dict:
        note = {
            "id": len(self.research_notes) + 1,
            "content": content,
            "category": category,
            "timestamp": datetime.now().isoformat()
        }
        self.research_notes.append(note)
        
        return {"success": True, "note_id": note["id"]}
    
    def generate_report(self, title: str, format: str = "markdown") -> Dict:
        report_lines = [
            f"# {title}",
            "",
            f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## 研究笔记",
            ""
        ]
        
        categories = {}
        for note in self.research_notes:
            if note["category"] not in categories:
                categories[note["category"]] = []
            categories[note["category"]].append(note)
        
        for category, notes in categories.items():
            report_lines.append(f"### {category}")
            for note in notes:
                report_lines.append(f"- {note['content']}")
            report_lines.append("")
        
        report_lines.extend([
            "## 参考来源",
            ""
        ])
        
        for source in self.sources:
            report_lines.append(f"- [{source['title']}]({source['url']})")
        
        return {
            "success": True,
            "report": "\n".join(report_lines)
        }
    
    def chat(self, user_input: str) -> str:
        messages = [
            {
                "role": "system",
                "content": """你是一个专业的研究助手。
                - 帮助用户收集和整理信息
                - 使用 search_web 搜索信息
                - 使用 take_note 记录重要内容
                - 使用 generate_report 生成研究报告
                - 保持客观、准确的研究态度"""
            },
            {"role": "user", "content": user_input}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        if message.tool_calls:
            tool_results = []
            
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                if func_name == "search_web":
                    result = self.search_web(func_args["query"])
                elif func_name == "take_note":
                    result = self.take_note(
                        func_args["content"],
                        func_args["category"]
                    )
                elif func_name == "generate_report":
                    result = self.generate_report(
                        func_args["title"],
                        func_args.get("format", "markdown")
                    )
                
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": func_name,
                    "content": json.dumps(result, ensure_ascii=False)
                })
            
            messages.append(message)
            messages.extend(tool_results)
            
            final_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            
            return final_response.choices[0].message.content
        
        return message.content

agent = ResearchAssistantAgent(api_key="your-api-key")
print(agent.chat("帮我研究人工智能在教育领域的应用"))
print(agent.chat("记录一条笔记：AI可以个性化学习路径，分类：应用场景"))
print(agent.chat("生成一份研究报告，标题是《AI在教育领域的应用研究》"))
```

## 项目 4：自动化工作流 Agent

### 项目描述

一个能够自动化执行多步骤工作流的 Agent，如数据处理流程、报告生成流程等。

### 代码实现

```python
from openai import OpenAI
import json
from typing import Dict, List, Callable
from datetime import datetime

class WorkflowAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.workflows = {}
        self.execution_history = []
    
    def register_workflow(self, name: str, steps: List[Dict]):
        self.workflows[name] = steps
    
    def execute_workflow(self, name: str, inputs: Dict) -> Dict:
        if name not in self.workflows:
            return {"success": False, "error": "工作流不存在"}
        
        workflow = self.workflows[name]
        results = []
        context = inputs.copy()
        
        for i, step in enumerate(workflow):
            step_result = self._execute_step(step, context)
            results.append(step_result)
            
            if not step_result["success"]:
                return {
                    "success": False,
                    "failed_step": i + 1,
                    "error": step_result["error"],
                    "partial_results": results
                }
            
            context.update(step_result.get("output", {}))
        
        return {
            "success": True,
            "results": results,
            "final_output": context
        }
    
    def _execute_step(self, step: Dict, context: Dict) -> Dict:
        step_type = step.get("type")
        
        if step_type == "llm":
            prompt = step["prompt"].format(**context)
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                "success": True,
                "output": {"result": response.choices[0].message.content}
            }
        
        elif step_type == "function":
            func = step["function"]
            args = {k: context.get(v[1:], v) if v.startswith("$") else v 
                   for k, v in step.get("args", {}).items()}
            
            result = func(**args)
            
            return {
                "success": True,
                "output": result
            }
        
        elif step_type == "condition":
            condition = step["condition"].format(**context)
            
            if eval(condition):
                return {"success": True, "output": {"branch": step["if_true"]}}
            else:
                return {"success": True, "output": {"branch": step.get("if_false", "default")}}
        
        return {"success": False, "error": f"未知步骤类型: {step_type}"}
    
    def create_workflow_from_description(self, description: str) -> Dict:
        prompt = f"""
        根据以下描述创建工作流定义：
        
        描述：{description}
        
        请以 JSON 格式返回工作流定义，格式如下：
        {{
            "name": "工作流名称",
            "steps": [
                {{"type": "llm", "prompt": "提示词模板"}},
                {{"type": "function", "function": "函数名", "args": {{"参数": "$变量名"}}}},
                {{"type": "condition", "condition": "条件表达式", "if_true": "分支A", "if_false": "分支B"}}
            ]
        }}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            workflow_def = json.loads(response.choices[0].message.content)
            self.register_workflow(workflow_def["name"], workflow_def["steps"])
            return {"success": True, "workflow": workflow_def}
        except:
            return {"success": False, "error": "无法解析工作流定义"}

agent = WorkflowAgent(api_key="your-api-key")

agent.register_workflow("content_creation", [
    {"type": "llm", "prompt": "研究主题：{topic}，列出5个关键点"},
    {"type": "llm", "prompt": "根据以下关键点撰写文章大纲：{result}"},
    {"type": "llm", "prompt": "根据以下大纲撰写完整文章：{result}"}
])

result = agent.execute_workflow("content_creation", {"topic": "人工智能的未来"})
print(result["final_output"]["result"])
```

## 项目总结

| 项目 | 复杂度 | 核心技术 | 适用场景 |
|------|--------|----------|----------|
| 智能客服 | ⭐⭐ | 工具调用、知识库 | 客户服务 |
| 代码助手 | ⭐⭐⭐ | 代码执行、错误分析 | 开发辅助 |
| 研究助手 | ⭐⭐⭐ | 信息检索、报告生成 | 研究分析 |
| 工作流 Agent | ⭐⭐⭐⭐ | 流程编排、条件分支 | 自动化任务 |

## 下一步

学习了实战项目后，了解 Agent 开发的最佳实践和常见陷阱：
→ [11-最佳实践与陷阱](../11-最佳实践与陷阱/)
