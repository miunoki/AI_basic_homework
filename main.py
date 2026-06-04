"""校园智能问答助手 —— RAG + 多轮对话 + 来源标注。"""

from llm import chat
from retriever import load_knowledge, retrieve

SYSTEM_PROMPT = """你是浙江大学「校园生活助手」，专门为新生解答校园生活中的各种问题。
你的知识全部来自 CC98 论坛上浙大学生们的真实讨论和经验分享。

回答规则：
1. 认真阅读【参考帖子】中的内容，提取有用信息来回答
2. 如果帖子里确实有相关信息，就直接引用回答，语气自然友好
3. 只有当所有参考帖子都和问题完全不相关时，才说「抱歉，资料库中暂未找到相关信息」
4. 回答要简洁、口语化，像一个热心的学长/学姐
5. 回答末尾用「📎 参考来源：」标注你依据了哪些帖子的标题
"""

RETRIEVAL_MODE = "keyword"  # "keyword" | "semantic"


def build_prompt(question, related, show_sources=True):
    """把检索到的帖子和用户问题拼成 Prompt。"""
    parts = []
    sources = []

    for i, c in enumerate(related[:2], 1):
        text = c["text"]

        if len(text) > 1500:
            text = text[:1500] + "\n...(内容过长，已截断)"

        parts.append(f"参考帖子 {i}:\n{text}")

        if c.get("title"):
            sources.append(f"帖子{i}: {c['title']}")

    refs = "\n\n---\n\n".join(parts)

    prompt = f"""【参考帖子】
{refs}

【用户问题】
{question}
"""

    if show_sources and sources:
        prompt += f"\n（可参考的来源：{'；'.join(sources)}）"

    return prompt


def main():
    chunks = load_knowledge()

    print(f"知识库已加载（{len(chunks)} 个知识条目）。")
    print(f"检索模式: {RETRIEVAL_MODE}")
    print("输入你的问题开始对话，输入 q 退出。\n")

    history = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    while True:
        question = input("你：").strip()

        if question.lower() in ("q", "quit", "exit"):
            print("再见！")
            break

        if not question:
            continue

        # ===== 上下文增强检索 =====
        search_query = question

        if len(history) >= 3:
            try:
                last_question = history[-2]["content"]

                if len(question) <= 15:
                    search_query = last_question + " " + question

            except Exception:
                pass

        related = retrieve(
            search_query,
            chunks,
            mode=RETRIEVAL_MODE
        )

        user_msg = build_prompt(
            question,
            related
        )

        messages = history + [
            {
                "role": "user",
                "content": user_msg
            }
        ]

        answer = chat(messages)

        history.append(
            {
                "role": "user",
                "content": question
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        print(f"\n助手：{answer}\n")


if __name__ == "__main__":
    main()
