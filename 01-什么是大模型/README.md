# 01-什么是大模型

大家好！今天我们来聊聊什么是大模型。

## 一、从聊天开始理解大模型

### 1.1 你已经在用大模型了！

想想你平时用 ChatGPT、文心一言、通义千问的场景：

```
你：帮我写一封请假邮件
AI：好的，请问请假原因是？...
```

**这就是大模型！** 它能理解你的话，然后给出回答。

### 1.2 大模型到底是什么？

简单说：**大模型就是一个"文字接龙"高手**

```
输入：今天天气真
大模型：好！
```

它通过阅读海量的文本（互联网上的文章、书籍、代码...），学会了预测下一个字应该是什么。

### 1.3 为什么叫"大"模型？

因为它真的很大！

| 模型 | 参数量 | 有多大？ |
|------|--------|----------|
| GPT-3 | 1750 亿 | 如果用脑子装，需要 1750 亿个神经元 |
| Llama 3 | 700 亿 | 70B 版本需要 140GB 显存 |
| Qwen2.5 | 720 亿 | 阿里开源的中文模型 |

**参数量是什么？** 可以理解为模型的"脑细胞"数量，越多越聪明。

## 二、大模型能做什么？

### 2.1 日常对话

```python
import openai

client = openai.OpenAI(api_key="你的 API Key", base_url="https://integrate.api.nvidia.com/v1")

response = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ]
)

print(response.choices[0].message.content)
```

**输出：**
```
你好！我是一个人工智能助手，可以帮助你回答问题、写作、编程...
```

### 2.2 写作文、写代码、写任何东西

```python
response = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[
        {"role": "user", "content": "帮我写一首关于春天的诗"}
    ]
)

print(response.choices[0].message.content)
```

**输出：**
```
春风拂面花自开，
柳絮飘飘满江南。
燕子归来寻旧主，
桃花含笑迎客来。
```

### 2.3 翻译

```python
response = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[
        {"role": "system", "content": "你是一个翻译助手"},
        {"role": "user", "content": "把这句话翻译成英文：你好，很高兴认识你"}
    ]
)

print(response.choices[0].message.content)
# 输出：Hello, nice to meet you
```

### 2.4 批改作业（JSON 格式输出）

这个很实用！比如让 AI 批改英文作文：

```python
import openai
import json
import json_repair

client = openai.OpenAI(api_key="你的 API Key", base_url="https://integrate.api.nvidia.com/v1")

essay = """
My Self-Introduction
Hello, everyone! My name is Li Ming.
I'm ten years old. I study at Green Tree Primary School.
I like reading books and playing football.
"""

# 关键：让 AI 输出 JSON 格式
prompt = """
用小学四年级的标准给我的英文作文打分
满分 100 分，请用这样的格式输出
{"分数":"xxx","评语":"xxxx"}
直接输出 JSON，不要有多余的话
"""

response = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": essay}
    ]
)

# 解析结果
result = response.choices[0].message.content
rjson = json.loads(json_repair.repair_json(result))

print(f"分数：{rjson['分数']}")
print(f"评语：{rjson['评语']}")
```

**输出：**
```
分数：85
评语：作文结构清晰，语法基本正确。注意大小写规范，继续加油！
```

**为什么用 JSON？** 因为 JSON 可以被代码识别和处理！这样我们就能把 AI 的输出用到程序里。

## 三、多模态：不止是文字

### 3.1 什么是多模态？

传统大模型只能处理文字，而**多模态大模型**可以处理：
- 📝 文字
- 🖼️ 图片
- 🎵 音频
- 🎥 视频

### 3.2 举个例子

**单模态（只能文字）：**
```
你：猫是什么？
AI：猫是一种哺乳动物...
```

**多模态（可以看图）：**
```
你：[上传一张猫的照片] 这是什么？
AI：这是一只橘色的猫，正在阳光下睡觉
```

### 3.3 多模态模型有哪些？

| 模型 | 能处理什么 | 特点 |
|------|------------|------|
| GPT-4o | 文字、图片、声音 | 最强多模态 |
| Gemini | 文字、图片、视频、音频 | Google 出品 |
| Claude 3 | 文字、图片 | 图像理解好 |
| Qwen-VL | 文字、图片 | 中文友好 |

### 3.4 多模态代码示例

用 Gemini 分析图片：

```python
import google.generativeai as genai

genai.configure(api_key="你的 API Key")

model = genai.GenerativeModel('gemini-2.0-flash')

response = model.generate_content([
    "这张图片里有什么？",
    {
        "mime_type": "image/jpeg",
        "data": open("cat.jpg", "rb").read()
    }
])

print(response.text)
```

## 四、主流大模型介绍

### 4.1 商业模型（需要 API Key）

| 模型 | 公司 | 特点 | 适合什么 |
|------|------|------|----------|
| GPT-4o | OpenAI | 最强 | 各种场景 |
| GPT-4o-mini | OpenAI | 便宜快速 | 日常使用 |
| Claude 3.5 | Anthropic | 安全、长文本 | 文档分析 |
| Gemini | Google | 多模态强 | 图像视频 |

### 4.2 开源模型（可以本地部署）

| 模型 | 公司 | 特点 |
|------|------|------|
| Llama 3.1 | Meta | 开源可商用 |
| Qwen2.5 | 阿里 | 中文能力强 |
| DeepSeek | 深度求索 | 性价比高 |
| Mistral | Mistral AI | 高效 |

## 五、动手试试！

### 练习 1：让 AI 自我介绍

```python
import openai

client = openai.OpenAI(api_key="你的 API Key", base_url="https://integrate.api.nvidia.com/v1")

response = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[
        {"role": "user", "content": "请用一句话介绍你自己"}
    ]
)

print(response.choices[0].message.content)
```

### 练习 2：让 AI 讲个笑话

```python
response = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[
        {"role": "user", "content": "讲个笑话"}
    ]
)

print(response.choices[0].message.content)
```

### 练习 3：让 AI 帮你写代码

```python
response = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[
        {"role": "user", "content": "用 Python 写一个计算斐波那契数列的函数"}
    ]
)

print(response.choices[0].message.content)
```

## 六、大模型的局限

大模型虽然厉害，但也有缺点：

1. **会胡说八道**（幻觉问题）
   ```
   问：鲁迅为什么打周树人？
   答：（其实鲁迅就是周树人，但 AI 可能会编原因）
   ```

2. **知识有截止时间**
   - 训练数据是过去的，不知道最新新闻

3. **数学可能算错**
   - 复杂的计算题容易出错

4. **需要花钱**
   - API 调用要钱（虽然有免费额度）

## 七、总结

今天学到了：

✅ 大模型是什么（文字接龙高手）
✅ 大模型能做什么（对话、写作、翻译、批改作业）
✅ 什么是多模态（能看图、听声音）
✅ 主流模型有哪些
✅ 如何用代码调用大模型

## 下一步

学会了调用大模型，接下来我们学习：
→ [02-AI 调用基础](../02-AI 调用基础/)

深入学习 API 调用的各种参数和技巧！
