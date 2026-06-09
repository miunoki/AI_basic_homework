"""对比实验：带知识库(RAG) vs 不带知识库。"""
import time
from llm import chat, is_configured, user_error_message
from retriever import is_query_in_scope, load_knowledge, retrieve
from main import (
    append_sources,
    build_prompt,
    SYSTEM_PROMPT,
    RETRIEVAL_MODE,
    RETRIEVAL_TOP_K,
    NO_RELEVANT_ANSWER,
)

TEST_QUESTIONS = [
    "图书馆怎么预约？",
    "校医院牙科怎么样？",
    "紫金港有什么好吃的？",
    "电动车在哪买？",
    "宿舍可以用大功率电器吗？",
    "绩点和记点有什么区别？",
    "军训要注意什么？",
    "怎么申请奖学金？",
    "校园卡丢了怎么办？",
    "附近有海底捞吗？",
    "转专业难不难？",
    "英语四级没过怎么办？",
]

RETRIEVAL_CASES = [
    ("图书馆怎么预约？", ("图书馆",)),
    ("校医院牙科怎么样？", ("牙", "校医院")),
    ("紫金港有什么好吃的？", ("紫金港", "zjg", "好吃")),
    ("电动车在哪买？", ("电动车",)),
    ("宿舍可以用大功率电器吗？", ("宿舍", "大功率")),
    ("绩点和记点有什么区别？", ("绩点", "记点")),
    ("怎么申请奖学金？", ("评奖", "奖学金")),
    ("校园卡丢了怎么办？", ("校园卡", "补办")),
    ("转专业难不难？", ("转专业",)),
    ("英语四级没过怎么办？", ("四级", "CET")),
    ("补考怎么申请？", ("补考", "缓考")),
]

OUT_OF_SCOPE_QUESTIONS = [
    "Python 项目怎么做？",
    "普通医院怎么挂号？",
    "密码怎么设置更安全？",
    "如何报名马拉松比赛？",
    "股票怎么买？",
]

NO_KB_SYSTEM = """你是一个通用AI助手，请回答用户的问题。
如果不知道，请如实说不知道，不要编造。"""


def ask_with_kb(question, chunks):
    """带知识库的 RAG 回答。"""
    related = retrieve(question, chunks, top_k=RETRIEVAL_TOP_K, mode=RETRIEVAL_MODE)
    if not related:
        return NO_RELEVANT_ANSWER
    user_msg = build_prompt(question, related)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return append_sources(chat(messages), related)


def ask_without_kb(question):
    """不带知识库的通用回答。"""
    messages = [
        {"role": "system", "content": NO_KB_SYSTEM},
        {"role": "user", "content": question},
    ]
    return chat(messages)


def run_case(answer_func):
    """执行单项实验，避免一次 API 异常中断整场对比。"""
    started_at = time.time()
    try:
        return answer_func(), time.time() - started_at, ""
    except Exception as exc:
        return "", time.time() - started_at, user_error_message(exc)


def classify_result(answer, error=""):
    """只标记可客观判断的状态，回答质量留给人工核验。"""
    if error:
        return "调用失败"

    refuse_keywords = ["抱歉", "未找到", "不知道", "没有相关信息"]
    has_refuse = any(kw in answer for kw in refuse_keywords)
    if has_refuse:
        return "已拒答"
    return "待人工核验"


def run_retrieval_benchmark(chunks, top_k=3):
    """运行无需 API 的检索命中与边界拒答基准。"""
    hits = 0
    print("离线检索基准")
    print("-" * 80)

    for question, expected_terms in RETRIEVAL_CASES:
        related = retrieve(
            question,
            chunks,
            top_k=top_k,
            mode=RETRIEVAL_MODE,
        )
        titles = [
            (item.get("title") or item.get("source") or "")
            for item in related
        ]
        matched = any(
            term.lower() in title.lower()
            for term in expected_terms
            for title in titles
        )
        hits += int(matched)
        status = "命中" if matched else "未命中"
        print(f"[{status}] {question}")
        for title in titles:
            print(f"       - {title}")

    rejected = sum(
        not is_query_in_scope(question)
        for question in OUT_OF_SCOPE_QUESTIONS
    )
    total = len(RETRIEVAL_CASES)
    boundary_total = len(OUT_OF_SCOPE_QUESTIONS)
    print("\n离线基准汇总")
    print(f"  校园问题 Top-{top_k} 命中：{hits}/{total} ({hits / total:.1%})")
    print(
        "  知识库外问题正确拒绝："
        f"{rejected}/{boundary_total} ({rejected / boundary_total:.1%})"
    )
    return {
        "retrieval_hits": hits,
        "retrieval_total": total,
        "boundary_rejected": rejected,
        "boundary_total": boundary_total,
    }


def main():
    chunks = load_knowledge()
    print(f"知识库：{len(chunks)} 条 | 测试问题：{len(TEST_QUESTIONS)} 个")
    print(f"检索模式：{RETRIEVAL_MODE} | 召回 top-{RETRIEVAL_TOP_K}\n")
    run_retrieval_benchmark(chunks)

    if not is_configured():
        print("\n未配置 DeepSeek API Key，跳过在线回答对比。")
        print("配置 DEEPSEEK_API_KEY 后重新运行，可查看 RAG 与无知识库回答。")
        return

    print("\n在线回答对比（回答质量需人工核验）")
    print("=" * 80)

    results = []
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] Q: {q}")

        a_kb, t_kb, error_kb = run_case(
            lambda: ask_with_kb(q, chunks)
        )
        a_no, t_no, error_no = run_case(
            lambda: ask_without_kb(q)
        )

        q_kb = classify_result(a_kb, error_kb)
        q_no = classify_result(a_no, error_no)
        display_kb = a_kb or error_kb
        display_no = a_no or error_no

        print(f"  [RAG] ({t_kb:.1f}s) [{q_kb}] {display_kb[:120]}...")
        print(f"  [无KB] ({t_no:.1f}s) [{q_no}] {display_no[:120]}...")

        results.append({
            "question": q,
            "rag_answer": a_kb,
            "no_kb_answer": a_no,
            "rag_status": q_kb,
            "no_kb_status": q_no,
            "rag_error": error_kb,
            "no_kb_error": error_no,
            "rag_time": t_kb,
            "no_kb_time": t_no,
        })

        time.sleep(0.5)

    # 汇总
    print("\n" + "=" * 80)
    print("对比实验总结")
    print("=" * 80)
    rag_review = sum(1 for r in results if r["rag_status"] == "待人工核验")
    no_review = sum(1 for r in results if r["no_kb_status"] == "待人工核验")
    rag_refuse = sum(1 for r in results if r["rag_status"] == "已拒答")
    no_refuse = sum(1 for r in results if r["no_kb_status"] == "已拒答")
    rag_failed = sum(1 for r in results if r["rag_status"] == "调用失败")
    no_failed = sum(1 for r in results if r["no_kb_status"] == "调用失败")

    print(f"{'':<20} {'RAG(带知识库)':<20} {'无知识库':<20}")
    print(f"{'待人工核验':<20} {rag_review:<20} {no_review:<20}")
    print(f"{'已拒答':<20} {rag_refuse:<20} {no_refuse:<20}")
    print(f"{'调用失败':<20} {rag_failed:<20} {no_failed:<20}")
    print(f"{'平均耗时':<20} {sum(r['rag_time'] for r in results)/len(results):.1f}s{'':<13} {sum(r['no_kb_time'] for r in results)/len(results):.1f}s")

    print("\n说明：")
    print("  脚本只记录可客观识别的拒答和调用失败。")
    print("  回答是否准确、是否有依据，需要结合参考来源人工核验。")
    print("  本次运行不会自动写入或覆盖 docs 中的报告。")


if __name__ == "__main__":
    main()
