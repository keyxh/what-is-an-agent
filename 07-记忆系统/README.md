# 记忆系统

## 为什么 Agent 需要记忆？

```
没有记忆：每次对话都是全新的，无法记住之前的内容
有了记忆：能够记住历史、保持上下文、积累知识
```

## 记忆类型

### 1. 短期记忆（Short-term Memory）

当前对话的上下文，存储在消息列表中。

```python
class ShortTermMemory:
    def __init__(self, max_messages: int = 10):
        self.messages = []
        self.max_messages = max_messages
    
    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context(self) -> list:
        return self.messages
    
    def clear(self):
        self.messages = []
```

### 2. 长期记忆（Long-term Memory）

持久化存储，可以跨会话使用。

```python
import json
from datetime import datetime

class LongTermMemory:
    def __init__(self, storage_path: str = "memory.json"):
        self.storage_path = storage_path
        self.memories = self._load()
    
    def _load(self) -> list:
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def _save(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)
    
    def add(self, content: str, metadata: dict = None):
        memory = {
            "id": len(self.memories),
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.memories.append(memory)
        self._save()
    
    def search(self, query: str) -> list:
        return [m for m in self.memories if query.lower() in m["content"].lower()]
    
    def get_recent(self, n: int = 5) -> list:
        return self.memories[-n:]
```

### 3. 向量记忆（Vector Memory）

使用向量数据库进行语义搜索。

```python
from typing import List
import numpy as np

class VectorMemory:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.vectors = []
        self.texts = []
        self.metadata = []
    
    def _get_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.embed(text)
    
    def add(self, text: str, meta: dict = None):
        vector = self._get_embedding(text)
        self.vectors.append(vector)
        self.texts.append(text)
        self.metadata.append(meta or {})
    
    def search(self, query: str, top_k: int = 5) -> List[dict]:
        query_vector = self._get_embedding(query)
        
        similarities = []
        for i, v in enumerate(self.vectors):
            sim = np.dot(query_vector, v) / (np.linalg.norm(query_vector) * np.linalg.norm(v))
            similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, sim in similarities[:top_k]:
            results.append({
                "text": self.texts[i],
                "score": sim,
                "metadata": self.metadata[i]
            })
        
        return results
```

## 记忆架构

```
┌─────────────────────────────────────────────┐
│                 Agent 记忆系统               │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────┐    ┌─────────────┐         │
│  │  短期记忆   │    │  长期记忆   │         │
│  │ (对话上下文)│    │ (持久存储)  │         │
│  └──────┬──────┘    └──────┬──────┘         │
│         │                  │                │
│         └────────┬─────────┘                │
│                  ↓                          │
│         ┌─────────────┐                     │
│         │  向量记忆   │                     │
│         │ (语义搜索)  │                     │
│         └─────────────┘                     │
│                                             │
└─────────────────────────────────────────────┘
```

## 完整记忆系统实现

```python
from openai import OpenAI
import json
from datetime import datetime
from typing import List, Optional

class MemorySystem:
    def __init__(self, api_key: str, max_short_term: int = 10):
        self.client = OpenAI(api_key=api_key)
        
        self.short_term = []
        self.max_short_term = max_short_term
        
        self.long_term = []
        
        self.working_memory = {}
    
    def add_message(self, role: str, content: str):
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.short_term.append(message)
        
        if len(self.short_term) > self.max_short_term:
            self._compress_short_term()
    
    def _compress_short_term(self):
        if len(self.short_term) <= self.max_short_term:
            return
        
        old_messages = self.short_term[:-self.max_short_term]
        summary = self._summarize(old_messages)
        
        self.long_term.append({
            "type": "summary",
            "content": summary,
            "timestamp": datetime.now().isoformat()
        })
        
        self.short_term = self.short_term[-self.max_short_term:]
    
    def _summarize(self, messages: List[dict]) -> str:
        text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "请简洁总结以下对话内容："},
                {"role": "user", "content": text}
            ]
        )
        
        return response.choices[0].message.content
    
    def add_fact(self, fact: str, category: str = "general"):
        self.long_term.append({
            "type": "fact",
            "content": fact,
            "category": category,
            "timestamp": datetime.now().isoformat()
        })
    
    def search_memory(self, query: str) -> List[dict]:
        results = []
        
        for msg in self.short_term:
            if query.lower() in msg["content"].lower():
                results.append({**msg, "source": "short_term"})
        
        for memory in self.long_term:
            if query.lower() in memory["content"].lower():
                results.append({**memory, "source": "long_term"})
        
        return results
    
    def get_context_for_query(self, query: str) -> str:
        relevant = self.search_memory(query)
        
        context_parts = []
        
        if relevant:
            context_parts.append("相关历史记忆：")
            for r in relevant[:3]:
                context_parts.append(f"- {r['content']}")
        
        if self.short_term:
            context_parts.append("\n当前对话：")
            for msg in self.short_term[-5:]:
                context_parts.append(f"{msg['role']}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def set_working(self, key: str, value):
        self.working_memory[key] = value
    
    def get_working(self, key: str):
        return self.working_memory.get(key)
    
    def clear_short_term(self):
        self.short_term = []
    
    def save_to_file(self, filepath: str):
        data = {
            "short_term": self.short_term,
            "long_term": self.long_term,
            "working_memory": self.working_memory
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_from_file(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.short_term = data.get("short_term", [])
            self.long_term = data.get("long_term", [])
            self.working_memory = data.get("working_memory", {})
        except FileNotFoundError:
            pass
```

## 使用示例

```python
memory = MemorySystem(api_key="your-api-key")

memory.add_message("user", "我叫张三")
memory.add_message("assistant", "你好张三！很高兴认识你。")
memory.add_message("user", "我喜欢编程")
memory.add_message("assistant", "编程是一项很有趣的技能！")

memory.add_fact("用户姓名：张三", category="personal")
memory.add_fact("用户爱好：编程", category="preferences")

print(memory.search_memory("张三"))
print(memory.get_context_for_query("我的爱好是什么？"))

memory.save_to_file("agent_memory.json")
```

## 记忆策略

### 1. 滑动窗口

```python
def sliding_window(messages: list, window_size: int = 10) -> list:
    return messages[-window_size:]
```

### 2. 摘要压缩

```python
def summarize_old_messages(messages: list, llm) -> str:
    text = "\n".join([m["content"] for m in messages])
    summary = llm.generate(f"总结以下内容：{text}")
    return summary
```

### 3. 重要性评分

```python
def score_importance(message: str, llm) -> float:
    prompt = f"""
    评估以下信息的重要性（0-1分）：
    信息：{message}
    
    只返回分数，不要其他内容。
    """
    score = llm.generate(prompt)
    return float(score)
```

### 4. 遗忘机制

```python
class ForgetfulMemory:
    def __init__(self, decay_rate: float = 0.1):
        self.memories = []
        self.decay_rate = decay_rate
    
    def add(self, content: str, importance: float = 1.0):
        self.memories.append({
            "content": content,
            "importance": importance,
            "access_count": 0
        })
    
    def access(self, index: int):
        self.memories[index]["access_count"] += 1
        self.memories[index]["importance"] += 0.1
    
    def decay(self):
        for m in self.memories:
            m["importance"] *= (1 - self.decay_rate)
        
        self.memories = [m for m in self.memories if m["importance"] > 0.1]
```

## 向量数据库集成

### 使用 ChromaDB

```python
import chromadb
from chromadb.utils import embedding_functions

class ChromaMemory:
    def __init__(self, collection_name: str = "agent_memory"):
        self.client = chromadb.Client()
        self.embedder = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.create_collection(name=collection_name)
    
    def add(self, text: str, metadata: dict = None):
        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[str(hash(text))]
        )
    
    def search(self, query: str, n_results: int = 5):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
```

### 使用 Pinecone

```python
import pinecone

class PineconeMemory:
    def __init__(self, api_key: str, environment: str, index_name: str):
        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)
    
    def add(self, id: str, vector: list, metadata: dict = None):
        self.index.upsert([(id, vector, metadata or {})])
    
    def search(self, vector: list, top_k: int = 5):
        return self.index.query(vector, top_k=top_k, include_metadata=True)
```

## 记忆系统最佳实践

| 实践 | 说明 |
|------|------|
| 分层存储 | 短期记忆用于当前对话，长期记忆用于持久知识 |
| 定期压缩 | 避免短期记忆无限增长 |
| 重要性筛选 | 只存储重要信息到长期记忆 |
| 语义搜索 | 使用向量搜索提高检索效率 |
| 隐私保护 | 敏感信息加密存储或不上传 |

## 下一步

学会了记忆系统后，学习主流的 Agent 框架：
→ [08-Agent框架](../08-Agent框架/)
