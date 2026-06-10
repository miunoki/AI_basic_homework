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

本项目基于**检索增强生成（RAG）**技术，构建了一个面向浙大新生的智能问答助手。系统从 CC98 论坛 9 个校园相关版块整理出 3643 个文本知识条目，用户提问时先检索最相关的帖子，再交给 DeepSeek 大模型基于这些真实资料生成回答，做到**准确、有据可依、来源可追溯**。

### 1.3 核心功能

| 功能 | 说明 | 状态 |
|------|------|------|
| 知识库 | 3643 个 CC98 论坛文本知识条目，覆盖 9 个版块 | ✅ |
| 关键词检索 | 校园同义词扩展 + 中文 2-gram 分词 + Jaccard 相似度 + 标题加权 + 低相关阈值 | ✅ |
| 语义检索（进阶） | sentence-transformers 向量嵌入 + 余弦相似度，含缓存与自动回退 | ✅ |
| RAG 检索增强 | 检索相关帖子 → 拼接 Prompt → DeepSeek 生成回答 | ✅ |
| 多轮对话 | 上下文记忆，连续追问不丢失；追问时自动拼接上一轮问题、来源标题和回答摘要增强检索 | ✅ |
| 幻觉抑制 | 知识库外问题拒绝编造，并建议通过学校相关部门或官方渠道确认 | ✅ |
| 来源标注（进阶） | 程序根据实际检索结果在回答末尾追加参考帖子标题 | ✅ |
| 命令行界面 | 交互式 CLI 问答 | ✅ |
| Gradio 网页（进阶） | 可视化聊天界面，支持访问码或个人 API Key、可轮换示例问题和检索调试折叠面板 | ✅ |
| 对比实验（进阶） | 离线检索基准 + 配置 API Key 后的 RAG/纯模型逐题对比 | ✅ |

---

## 二、系统架构

```
用户提问 → 中文分词 → 知识库检索(Jaccard/语义) → Prompt拼接 → DeepSeek API → 生成回答
```

选用经典 RAG（检索增强生成）架构，核心流程：

1. **检索**：用户问题先经过校园同义词扩展，再经 2-gram 中文分词后，在 3643 条知识条目中进行 Jaccard 相似度匹配（或 sentence-transformers 语义向量检索），召回 top-8 相关帖子
2. **增强**：选取前 4 条相关帖子拼接成结构化 Prompt，注入 System Prompt
3. **生成**：调用 DeepSeek `deepseek-chat` 模型，基于检索资料生成口语化、有据可依的回答
4. **来源追加**：程序根据实际进入 Prompt 的检索结果，在回答末尾统一追加参考来源
5. **对话管理**：多轮对话中保存历史上下文，短追问会自动结合上一轮问题、来源标题和回答摘要增强检索查询

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
├── context_utils.py         # 多轮追问识别与检索上下文增强
├── main.py                 # 命令行入口（多轮对话 + 上下文增强检索）
├── app.py                  # Gradio 网页界面（含 API 配置面板）
├── compare.py              # 对比实验：RAG vs 纯大模型
├── scrape_cc98.py           # CC98 论坛爬虫
├── knowledge/              # 知识库文本文件（3643 条）
└── docs/                   # 技术栈、实验报告、分工说明、答辩 PPT
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

也可以通过环境变量配置。PowerShell：

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
$env:ACCESS_CODE="给新生使用的访问码"
```

Windows CMD：

```bat
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
| 正常成年人每天应该喝多少毫升水？ | 知识库无此内容，拒绝编造具体数值并建议通过可靠渠道确认 |
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

该脚本始终先运行无需 API Key 的离线检索基准；配置 `DEEPSEEK_API_KEY` 后，还会逐题输出“带知识库 RAG”和“不带知识库纯大模型”的回答，供人工比较事实依据、校园针对性和来源可追溯性。

当前关键词检索离线基准结果：

- 校园问题 Top-3 命中：11/11（100%）
- 知识库外问题正确拒绝：5/5（100%）

> 离线基准用于验证检索与边界判断，不等同于对生成回答质量的自动评分。在线回答对比需要有效的 DeepSeek API Key，并由人工结合帖子内容审阅。

---

## 六、检索模式

通过环境变量切换检索模式，无需修改源码。PowerShell：

```powershell
$env:RETRIEVAL_MODE="keyword"
python main.py

$env:RETRIEVAL_MODE="semantic"
python main.py
```

可选值为 `keyword` 和 `semantic`；未设置或填写无效值时自动回退为 `keyword`。默认召回 top-8 相关帖子，并将前 4 条写入 Prompt。关键词检索会过滤低于 50 分的弱相关结果；如果没有候选帖子，系统会直接拒答并给出官方咨询建议，不再调用模型硬答。

语义检索首次初始化约 1-2 分钟，后续从缓存加载可秒级启动。若 HuggingFace 不可访问，系统自动回退关键词模式。

---

## 七、常见问题

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| `No such file or directory: main.py` | 当前目录不是项目目录 | `cd` 到项目根目录 |
| `No module named openai` | 依赖未安装 | `pip install -r requirements.txt` |
| 未找到 API Key | 服务端未配置 Key，且网页内未输入个人 Key | 维护者配置 `DEEPSEEK_API_KEY`，或用户在网页内输入个人 Key |
| 访问码不正确 | 输入的访问码与维护者配置不一致 | 检查 `ACCESS_CODE` 或联系维护者 |
| 知识库为空或不存在 | `knowledge/` 缺失或没有 UTF-8 `.txt` 文件 | 检查项目根目录下的 `knowledge/`；程序会输出实际解析路径 |
| 网页打不开 | `app.py` 未运行或端口不同 | 查看终端输出的 local URL |
| 语义检索下载失败 | 网络无法访问 HuggingFace | 设置 `RETRIEVAL_MODE=keyword`；程序也会自动回退 |
| API 调用失败 | Key 无效、余额不足、限流或网络异常 | 根据界面中的安全提示检查配置后重试 |

---

## 八、参考资料

- [DeepSeek 开放平台](https://platform.deepseek.com)
- [Gradio 官方文档](https://www.gradio.app/docs)
- [CC98 论坛](https://www.cc98.org/)
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020
