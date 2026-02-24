# Agent 框架

## 为什么需要 Agent 框架？

```
从零实现 Agent：
- 需要自己处理消息管理
- 需要自己实现工具调用
- 需要自己管理记忆系统
- 需要自己处理错误和重试

使用 Agent 框架：
- 开箱即用的组件
- 标准化的接口
- 社区支持和文档
- 快速开发和迭代
```

## 主流 Agent 框架对比

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| LangChain | 最流行，生态丰富 | 通用 Agent 开发 |
| AutoGen | 微软出品，多 Agent | 多 Agent 协作 |
| CrewAI | 角色扮演，易用 | 团队协作模拟 |
| LlamaIndex | 数据索引强 | RAG 应用 |
| Haystack | 生产就绪 | 企业级应用 |
| Semantic Kernel | 微软出品 | 企业集成 |

## LangChain

### 安装

```bash
pip install langchain langchain-openai
```

### 基础使用

```python
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = OpenAI(api_key="your-api-key")

prompt = PromptTemplate(
    input_variables=["topic"],
    template="写一首关于{topic}的诗"
)

chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run("人工智能")
print(result)
```

### 使用 Agent

```python
from langchain_openai import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

llm = OpenAI(api_key="your-api-key")

def get_word_length(word: str) -> str:
    return str(len(word))

def multiply(numbers: str) -> str:
    a, b = map(int, numbers.split(","))
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

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("单词'artificial'有多少个字母？")
print(result)
```

### 使用 Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

print(conversation.predict(input="你好，我叫张三"))
print(conversation.predict(input="我叫什么名字？"))
```

### 使用 RAG

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

loader = TextLoader("document.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(api_key="your-api-key")
vectorstore = Chroma.from_documents(texts, embeddings)

retriever = vectorstore.as_retriever()

from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

result = qa.run("文档的主要内容是什么？")
```

## AutoGen

### 安装

```bash
pip install pyautogen
```

### 基础使用

```python
import autogen

config_list = [
    {
        "model": "gpt-4",
        "api_key": "your-api-key"
    }
]

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": config_list
    }
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding"}
)

user_proxy.initiate_chat(
    assistant,
    message="写一个Python程序计算斐波那契数列"
)
```

### 多 Agent 协作

```python
import autogen

config_list = [{"model": "gpt-4", "api_key": "your-api-key"}]

planner = autogen.AssistantAgent(
    name="planner",
    system_message="你是一个任务规划专家，负责分解任务",
    llm_config={"config_list": config_list}
)

coder = autogen.AssistantAgent(
    name="coder",
    system_message="你是一个程序员，负责编写代码",
    llm_config={"config_list": config_list}
)

reviewer = autogen.AssistantAgent(
    name="reviewer",
    system_message="你是一个代码审查员，负责检查代码质量",
    llm_config={"config_list": config_list}
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding"}
)

groupchat = autogen.GroupChat(
    agents=[user_proxy, planner, coder, reviewer],
    messages=[],
    max_round=12
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": config_list}
)

user_proxy.initiate_chat(
    manager,
    message="开发一个简单的待办事项应用"
)
```

## CrewAI

### 安装

```bash
pip install crewai
```

### 基础使用

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="研究员",
    goal="研究并收集信息",
    backstory="你是一位经验丰富的研究员，擅长收集和分析数据",
    verbose=True
)

writer = Agent(
    role="作者",
    goal="撰写高质量文章",
    backstory="你是一位专业作家，擅长将复杂信息转化为易懂的文章",
    verbose=True
)

research_task = Task(
    description="研究人工智能的最新发展趋势",
    agent=researcher
)

write_task = Task(
    description="根据研究结果撰写一篇文章",
    agent=writer
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    verbose=True
)

result = crew.kickoff()
print(result)
```

### 使用工具

```python
from crewai_tools import SerperDevTool, FileReadTool

search_tool = SerperDevTool()
file_tool = FileReadTool()

researcher = Agent(
    role="研究员",
    goal="搜索最新信息",
    backstory="你擅长使用搜索工具",
    tools=[search_tool]
)

analyst = Agent(
    role="分析师",
    goal="分析文档内容",
    backstory="你擅长分析各种文档",
    tools=[file_tool]
)
```

## LlamaIndex

### 安装

```bash
pip install llama-index
```

### RAG 应用

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI

documents = SimpleDirectoryReader("./data").load_data()

Settings.llm = OpenAI(api_key="your-api-key")

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("文档的主要内容是什么？")
print(response)
```

### Agent 使用

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

def multiply(a: int, b: int) -> int:
    return a * b

def add(a: int, b: int) -> int:
    return a + b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

agent = ReActAgent.from_tools(
    [multiply_tool, add_tool],
    llm=OpenAI(api_key="your-api-key"),
    verbose=True
)

response = agent.chat("123 * 456 是多少？")
print(response)
```

## Semantic Kernel

### 安装

```bash
pip install semantic-kernel
```

### 基础使用

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()

kernel.add_chat_service(
    "chat-gpt",
    OpenAIChatCompletion("gpt-4", "your-api-key")
)

prompt = kernel.create_semantic_function(
    "写一首关于{{$topic}}的诗"
)

result = await kernel.run_async(
    prompt,
    variables=sk.ContextVariables(variables={"topic": "人工智能"})
)
print(result)
```

### 使用插件

```python
from semantic_kernel import SKContext

def get_weather(city: str) -> str:
    return f"{city}今天晴天，25°C"

kernel.create_semantic_function(
    "获取{{$city}}的天气信息",
    function_name="get_weather"
)

kernel.register_custom_function(
    function=get_weather,
    plugin_name="weather"
)
```

## 框架选择指南

```
需求：简单对话应用
推荐：LangChain 基础链

需求：需要 RAG 能力
推荐：LlamaIndex 或 LangChain + 向量数据库

需求：多 Agent 协作
推荐：AutoGen 或 CrewAI

需求：企业级应用
推荐：Semantic Kernel 或 Haystack

需求：快速原型开发
推荐：CrewAI
```

## 框架对比总结

| 特性 | LangChain | AutoGen | CrewAI | LlamaIndex |
|------|-----------|---------|--------|------------|
| 学习曲线 | 中等 | 中等 | 简单 | 中等 |
| 文档质量 | 优秀 | 良好 | 良好 | 优秀 |
| 社区活跃度 | 最高 | 高 | 中等 | 高 |
| 多 Agent | 支持 | 核心功能 | 核心功能 | 支持 |
| RAG 能力 | 强 | 一般 | 一般 | 最强 |
| 生产就绪 | 良好 | 良好 | 中等 | 优秀 |

## 下一步

了解了主流框架后，学习如何构建多 Agent 系统：
→ [09-多Agent系统](../09-多Agent系统/)
