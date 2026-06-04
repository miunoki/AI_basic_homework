# 校园智能问答助手 —— 基于 CC98 论坛的新生入学指南

**《人工智能基础》课程大作业 · 方向一：校园智能问答助手（知识库 + RAG）**

| 项目 | 内容 |
|------|------|
| 课程 | 人工智能基础 |
| 选题 | 方向一：校园智能问答助手 |
| 题目 | 基于 CC98 论坛的新生入学指南智能问答助手 |
| 组号 | 16 |
| 成员 | 游尚洲、马野、陈俊希 |
| 日期 | 2026 年 5 月 |

---

## 一、项目概述

### 1.1 背景与问题

每年浙江大学新生入学都会面临大量校园生活问题：图书馆怎么预约？校医院哪个科室靠谱？食堂什么好吃？这些问题官方渠道信息不够具体，而 CC98 论坛上沉淀了大量真实、实用、经过实践检验的经验帖。通用大模型（如 DeepSeek）对浙大校园具体情况并不了解，直接提问容易产生"幻觉"——编造不存在的规章制度和设施。

### 1.2 解决方案

本项目基于**检索增强生成（RAG）**技术，构建了一个面向浙大新生的智能问答助手。系统从 CC98 论坛爬取了 9 个核心版块约 3640 条真实帖子构建知识库，用户提问时先检索最相关的帖子，再交给 DeepSeek 大模型基于这些真实资料生成回答，做到**准确、有据可依、来源可追溯**。

### 1.3 核心功能

| 功能 | 说明 | 状态 |
|------|------|------|
| 知识库 | 约 3640 条 CC98 论坛帖子，覆盖 9 个版块，运行时加载 3643 个知识条目 | ✅ |
| 关键词检索 | 校园同义词扩展 + 中文 2-gram 分词 + Jaccard 相似度 + 主题词标题加权 | ✅ |
| 语义检索（进阶） | sentence-transformers 向量嵌入 + 余弦相似度，含缓存与自动回退 | ✅ |
| RAG 检索增强 | 检索相关帖子 → 拼接 Prompt → DeepSeek 生成回答 | ✅ |
| 多轮对话 | 上下文记忆，连续追问不丢失；追问时自动增强检索查询 | ✅ |
| 幻觉抑制 | 知识库外问题如实说不知道，不编造 | ✅ |
| 来源标注（进阶） | 回答末尾标注参考帖子标题 | ✅ |
| 命令行界面 | 交互式 CLI 问答 | ✅ |
| Gradio 网页（进阶） | 可视化聊天界面，支持访问码或个人 API Key 两种进入方式 | ✅ |
| 对比实验（进阶） | 12 题 RAG vs 纯大模型效果对比，输出量化报告 | ✅ |

---

## 二、系统架构

```
用户提问 → 中文分词 → 知识库检索(Jaccard/语义) → Prompt拼接 → DeepSeek API → 生成回答
```

选用经典 RAG（检索增强生成）架构，核心流程：

1. **检索**：用户问题先经过校园同义词扩展，再经 2-gram 中文分词后，在 3643 条知识条目中进行 Jaccard 相似度匹配（或 sentence-transformers 语义向量检索），召回 top-8 相关帖子
2. **增强**：选取前 4 条相关帖子拼接成结构化 Prompt，注入 System Prompt 和来源标注信息
3. **生成**：调用 DeepSeek `deepseek-chat` 模型，基于检索资料生成口语化、有据可依的回答
4. **对话管理**：多轮对话中保存历史上下文，追问时自动用上一轮问题增强检索查询

---

## 三、项目结构

```
大作业/
├── README.md                # 本文件（项目文档）
├── requirements.txt         # Python 依赖
├── config.py                # 本地配置文件（不提交，可配置 API Key / 访问码）
├── .gitignore               # Git 忽略规则
├── llm.py                  # DeepSeek API 封装
├── retriever.py            # 知识库加载与检索（关键词 + 语义双模式）
├── main.py                 # 命令行入口（多轮对话 + 上下文增强检索）
├── app.py                  # Gradio 网页界面（含 API 配置面板）
├── compare.py              # 对比实验：RAG vs 纯大模型
├── scrape_cc98.py           # CC98 论坛爬虫
├── knowledge/              # 知识库文本文件（3643 条）
└── docs/                   # 技术栈、实验报告、分工说明、答辩 PPT
    └── 技术栈.md             # 当前技术方案与技术变更记录
```

---

## 四、环境与配置

### 4.1 环境要求

- **操作系统**：Windows / macOS / Linux 均可
- **Python**：建议 3.10+（命令行模式在 3.13 下也可运行）
- **网络**：调用 DeepSeek API 需要联网
- **API Key**：维护者可配置服务端 DeepSeek API Key；已有个人 Key 的用户也可在网页内临时输入
- **可选**：语义检索首次运行需要访问 HuggingFace 下载模型（约 120MB）

### 4.2 安装依赖

```bash
pip install -r requirements.txt
```

如安装较慢可换国内镜像：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4.3 配置 API Key 与访问码

在项目根目录创建 `config.py`，维护者可填入服务端 DeepSeek API Key 和统一访问码：

```python
DEEPSEEK_API_KEY = "你的 DeepSeek API Key"
ACCESS_CODE = "给新生使用的访问码"
```

也可以通过环境变量配置：

```bash
set DEEPSEEK_API_KEY=你的 DeepSeek API Key
set ACCESS_CODE=给新生使用的访问码
```

> ⚠️ `config.py` 已被 `.gitignore` 忽略，不要提交真实密钥或访问码。网页模式下，已有个人 Key 的用户也可在界面内临时输入 Key，无需访问码。

---

## 五、运行方式

### 5.1 命令行模式

```bash
python main.py
```

正常启动时显示：

```
知识库已加载（3643 个知识条目）。
检索模式: keyword
输入你的问题开始对话，输入 q 退出。
```

**演示示例：**

| 问题 | 预期回答 |
|------|---------|
| 图书馆怎么预约？ | 基于帖子给出预约网址、签到规则、违约惩罚等具体信息 |
| 预约后多久签到？ | 基于上下文理解"预约"指图书馆预约，回答半小时内签到 |
| 苹果手机和华为哪个好？ | 知识库无此内容，回答"暂未找到相关信息"，不编造 |
| 校医院牙科怎么样？ | 引用多个帖子，给出医生评价、费用、预约难度等综合信息 |

### 5.2 Gradio 网页模式

```bash
python app.py
```

或在 Windows 下双击：

```bash
start_web.bat
```

浏览器会自动打开 `http://127.0.0.1:7860`。网页支持两种进入方式：

- **新生**：输入项目方提供的访问码，使用服务端配置的 API。
- **已有 DeepSeek API Key 的用户**：直接输入个人 API Key，无需访问码。
- 如果同时填写访问码和个人 API Key，优先使用个人 API Key。

### 5.3 对比实验

```bash
python compare.py
```

测试 12 个校园生活问题，逐题对比"带知识库 RAG"与"不带知识库纯大模型"的回答质量，输出量化统计和详细报告。

---

## 六、检索模式

在 `main.py` 中修改：

```python
RETRIEVAL_MODE = "keyword"   # 同义词扩展 + 关键词检索，启动快，无需额外模型
RETRIEVAL_MODE = "semantic"  # 语义向量检索，需 sentence-transformers，首次需下载模型
```

默认配置为召回 top-8 相关帖子，并将前 4 条写入 Prompt。这样比只引用 top-2 覆盖更多经验帖，同时通过单条截断控制 Prompt 长度。

语义检索首次初始化约 1-2 分钟，后续从缓存加载可秒级启动。若 HuggingFace 不可访问，系统自动回退关键词模式。

---

## 七、常见问题

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| `No such file or directory: main.py` | 当前目录不是项目目录 | `cd` 到项目根目录 |
| `No module named openai` | 依赖未安装 | `pip install -r requirements.txt` |
| 未找到 API Key | 服务端未配置 Key，且网页内未输入个人 Key | 维护者配置 `DEEPSEEK_API_KEY`，或用户在网页内输入个人 Key |
| 访问码不正确 | 输入的访问码与维护者配置不一致 | 检查 `ACCESS_CODE` 或联系维护者 |
| 知识库加载 0 条 | `knowledge/` 文件夹缺失 | 确认 `knowledge/` 与 `main.py` 在同一目录下 |
| 网页打不开 | `app.py` 未运行或端口不同 | 查看终端输出的 local URL |
| 语义检索下载失败 | 网络无法访问 HuggingFace | 改用 `RETRIEVAL_MODE = "keyword"` |
| 回答较慢 | API 响应耗时 | 正常 2-5 秒，避免连续高频调用 |

---

- [DeepSeek 开放平台](https://platform.deepseek.com)
- Gradio 官方文档：https://www.gradio.app/docs
- CC98 论坛：https://www.cc98.org/
- Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, 2020
