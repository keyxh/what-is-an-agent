# 工具调用详解

## 什么是工具调用？

工具调用（Tool Calling / Function Calling）让 AI 能够执行实际操作，如搜索网络、查询数据库、调用 API 等。

```
没有工具：AI 只能"说"
有了工具：AI 能够"做"
```

## 工作原理

```
用户请求 → AI 判断是否需要工具 → 选择工具并生成参数 → 执行工具 → 返回结果 → AI 整合回答
```

## OpenAI Function Calling

### 定义工具

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索网络获取信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```

### 调用模型

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "北京今天天气怎么样？"}
    ],
    tools=tools,
    tool_choice="auto"
)

message = response.choices[0].message

if message.tool_calls:
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"调用函数: {function_name}")
        print(f"参数: {arguments}")
```

### 执行工具并返回结果

```python
import json

def get_weather(city: str, unit: str = "celsius") -> dict:
    weather_data = {
        "北京": {"temp": 25, "condition": "晴天"},
        "上海": {"temp": 28, "condition": "多云"},
        "广州": {"temp": 30, "condition": "小雨"}
    }
    return weather_data.get(city, {"temp": 20, "condition": "未知"})

def search_web(query: str) -> str:
    return f"关于'{query}'的搜索结果..."

available_functions = {
    "get_weather": get_weather,
    "search_web": search_web
}

def run_conversation(user_input: str):
    messages = [{"role": "user", "content": user_input}]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    messages.append(message)
    
    if message.tool_calls:
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            function_result = available_functions[function_name](**function_args)
            
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": str(function_result)
            })
        
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        return final_response.choices[0].message.content
    
    return message.content

result = run_conversation("北京今天天气怎么样？")
print(result)
```

## 多工具调用

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "获取股票价格",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "获取相关新闻",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        }
    }
]
```

## 工具类型

### 1. 信息获取类

```python
tools = [
    {
        "name": "search_web",
        "description": "搜索网络",
        "examples": ["搜索最新新闻", "查询技术文档"]
    },
    {
        "name": "query_database",
        "description": "查询数据库",
        "examples": ["查询用户信息", "获取订单数据"]
    },
    {
        "name": "get_weather",
        "description": "获取天气",
        "examples": ["北京天气", "上海气温"]
    }
]
```

### 2. 操作执行类

```python
tools = [
    {
        "name": "send_email",
        "description": "发送邮件",
        "parameters": ["to", "subject", "body"]
    },
    {
        "name": "create_file",
        "description": "创建文件",
        "parameters": ["path", "content"]
    },
    {
        "name": "execute_code",
        "description": "执行代码",
        "parameters": ["language", "code"]
    }
]
```

### 3. 数据处理类

```python
tools = [
    {
        "name": "analyze_data",
        "description": "分析数据",
        "parameters": ["data", "analysis_type"]
    },
    {
        "name": "transform_data",
        "description": "转换数据格式",
        "parameters": ["data", "from_format", "to_format"]
    }
]
```

## 工具设计原则

### 1. 单一职责

```python
# 好的设计
def get_user_by_id(user_id: str): ...
def get_user_by_email(email: str): ...

# 不好的设计
def get_user(user_id=None, email=None, name=None): ...
```

### 2. 清晰的描述

```python
# 好的描述
"获取指定城市的当前天气信息，包括温度、湿度、天气状况"

# 不好的描述
"获取天气"
```

### 3. 合理的参数

```python
{
    "name": "search_products",
    "parameters": {
        "query": {
            "type": "string",
            "description": "搜索关键词"
        },
        "category": {
            "type": "string",
            "enum": ["electronics", "clothing", "books"],
            "description": "商品类别（可选）"
        },
        "price_range": {
            "type": "object",
            "properties": {
                "min": {"type": "number"},
                "max": {"type": "number"}
            },
            "description": "价格范围（可选）"
        }
    },
    "required": ["query"]
}
```

## 错误处理

```python
def safe_tool_call(tool_func, **kwargs):
    try:
        result = tool_func(**kwargs)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_with_retry(tool_func, max_retries=3, **kwargs):
    for i in range(max_retries):
        try:
            return tool_func(**kwargs)
        except Exception as e:
            if i == max_retries - 1:
                return f"执行失败: {str(e)}"
            continue
```

## 工具调用流程图

```
┌─────────────┐
│  用户请求   │
└──────┬──────┘
       ↓
┌─────────────┐
│  AI 分析    │
└──────┬──────┘
       ↓
   需要工具？ ─否─→ 直接回答
       │
       是
       ↓
┌─────────────┐
│ 选择工具    │
│ 生成参数    │
└──────┬──────┘
       ↓
┌─────────────┐
│ 执行工具    │
└──────┬──────┘
       ↓
┌─────────────┐
│ 获取结果    │
└──────┬──────┘
       ↓
   需要更多？ ─是─→ 继续调用
       │
       否
       ↓
┌─────────────┐
│ 整合回答    │
└─────────────┘
```

## 实战示例：智能助手

```python
class SmartAssistant:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.tools = self._define_tools()
        self.functions = {
            "get_weather": self._get_weather,
            "search_web": self._search_web,
            "calculate": self._calculate,
            "send_email": self._send_email
        }
    
    def _define_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取城市天气",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"}
                        },
                        "required": ["city"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "搜索网络信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "执行数学计算",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"}
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]
    
    def _get_weather(self, city: str) -> str:
        return f"{city}今天晴天，气温25°C"
    
    def _search_web(self, query: str) -> str:
        return f"关于'{query}'的搜索结果..."
    
    def _calculate(self, expression: str) -> str:
        try:
            return str(eval(expression))
        except:
            return "计算错误"
    
    def _send_email(self, to: str, subject: str, body: str) -> str:
        return f"邮件已发送至 {to}"
    
    def chat(self, user_input: str) -> str:
        messages = [{"role": "user", "content": user_input}]
        
        while True:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            messages.append(message)
            
            if not message.tool_calls:
                return message.content
            
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                result = self.functions[func_name](**func_args)
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": func_name,
                    "content": result
                })

assistant = SmartAssistant(api_key="your-api-key")
print(assistant.chat("北京天气怎么样？"))
print(assistant.chat("帮我计算 123 * 456"))
```

## 下一步

学会了工具调用后，学习如何构建 Agent 的记忆系统：
→ [07-记忆系统](../07-记忆系统/)
