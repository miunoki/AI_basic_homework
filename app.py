"""校园智能问答助手 —— Gradio 网页聊天界面（美化版 + 访问码/API 配置）。"""
import os
import gradio as gr
from llm import chat, init_client, is_configured
from retriever import load_knowledge, retrieve

RETRIEVAL_MODE = "keyword"
RETRIEVAL_TOP_K = 8
PROMPT_TOP_K = 4
MAX_SOURCE_CHARS = 900
NO_RELEVANT_ANSWER = "抱歉，资料库中暂未找到相关信息。"
THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Noto Sans SC"),
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="0px",
    block_shadow="0 1px 3px 0 rgb(0 0 0 / 0.06)",
    button_primary_background_fill="#1565C0",
    button_primary_background_fill_hover="#0D47A1",
    button_primary_text_color="white",
    border_color_primary="#1565C0",
    loader_color="#1565C0",
)

SYSTEM_PROMPT = """你是浙江大学「校园生活助手」，专门为新生解答校园生活中的各种问题。
你的知识全部来自 CC98 论坛上浙大学生们的真实讨论和经验分享。

回答规则：
1. 认真阅读【参考帖子】中的内容，提取有用信息来回答
2. 如果帖子里确实有相关信息，就直接引用回答，语气自然友好
3. 只有当所有参考帖子都和问题完全不相关时，才说「抱歉，资料库中暂未找到相关信息」
4. 回答要简洁、口语化，像一个热心的学长/学姐
5. 回答末尾标注参考来源"""

CSS = """
.gradio-container { max-width: 880px !important; margin: 0 auto !important; }
.header-bar {
    background: linear-gradient(135deg, #0D47A1 0%, #1565C0 40%, #1976D2 100%);
    border-radius: 16px; padding: 28px 32px; margin-bottom: 20px; color: white;
}
.header-bar h1 { font-size: 1.6rem; font-weight: 700; margin: 0 0 6px 0; color: white; }
.header-bar p  { font-size: 0.9rem; opacity: 0.85; margin: 0; }
.header-tags { display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
.header-tags span {
    background: rgba(255,255,255,0.15); backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.2); border-radius: 20px;
    padding: 4px 14px; font-size: 0.78rem; color: white;
}
.api-accordion { margin-bottom: 16px !important; }
.api-row { align-items: end !important; gap: 10px !important; }
.api-row button { align-self: end !important; border-radius: 10px !important; font-weight: 600 !important; }
.api-row .wrap { gap: 8px !important; }
.api-status-ok { color: #16a34a !important; font-weight: 600 !important; }
.api-status-err { color: #dc2626 !important; font-weight: 600 !important; }
.chatbot-wrapper { border-radius: 16px; overflow: hidden; }
.chatbot-wrapper .bubble-wrap { padding: 8px 0; }
.input-row textarea {
    border-radius: 12px !important; border: 1.5px solid #e2e8f0 !important;
    padding: 12px 16px !important; font-size: 0.95rem !important;
    transition: border-color 0.2s;
}
.input-row textarea:focus {
    border-color: #1565C0 !important;
    box-shadow: 0 0 0 3px rgba(21,101,192,0.1) !important;
}
.send-btn { border-radius: 12px !important; font-weight: 600 !important; padding: 10px 24px !important; }
.footer-bar { text-align: center; padding: 24px 0 4px; color: #94a3b8; font-size: 0.78rem; }
.footer-bar span { margin: 0 10px; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }
"""

HEADER_HTML = """
<div class="header-bar">
    <h1>🎓 浙大新生入学指南 · 智能问答助手</h1>
    <p>基于 CC98 论坛真实帖子，用 RAG 检索增强生成，为新生提供可靠的校园生活解答</p>
    <div class="header-tags">
        <span>📚 3640 条知识条目</span>
        <span>🔍 RAG 检索增强</span>
        <span>🤖 DeepSeek 驱动</span>
        <span>🏫 9 大校园版块</span>
    </div>
</div>
"""

FOOTER_HTML = """
<div class="footer-bar">
    浙江大学 · 人工智能通识基础课程项目<span>|</span>方向一：校园智能问答助手
</div>
"""


def build_prompt(question, related):
    parts = []
    for i, c in enumerate(related[:PROMPT_TOP_K], 1):
        text = c["text"]
        if len(text) > MAX_SOURCE_CHARS:
            text = text[:MAX_SOURCE_CHARS] + "\n...(内容过长，已截断)"
        parts.append(f"参考帖子 {i}:\n{text}")
    refs = "\n\n---\n\n".join(parts)
    return f"【参考帖子】\n{refs}\n\n【用户问题】{question}"


chunks = load_knowledge()
print(
    f"知识库已加载（{len(chunks)} 个知识条目），检索模式: {RETRIEVAL_MODE}，"
    f"召回 top-{RETRIEVAL_TOP_K}，Prompt 引用 top-{PROMPT_TOP_K}"
)

SERVER_API_CONFIGURED = is_configured()


# ===== 访问码 / API 配置逻辑 =====


def get_access_code():
    """从环境变量或 config.py 读取统一访问码。"""
    code = os.getenv("ACCESS_CODE", "").strip()
    if code:
        return code
    try:
        from config import ACCESS_CODE
        return str(ACCESS_CODE).strip()
    except (ImportError, AttributeError):
        return ""


ACCESS_CODE = get_access_code()


def empty_auth_state():
    return {
        "access_granted": False,
        "personal_key_configured": False,
    }


def submit_access(access_code, api_key, auth_state):
    """访问码和个人 API Key 双通道入口。个人 API Key 优先。"""
    auth_state = dict(auth_state or empty_auth_state())
    access_code = (access_code or "").strip()
    key = (api_key or "").strip()

    if key:
        if len(key) < 10 or not key.startswith("sk-"):
            return (
                gr.Markdown("❌ **API Key 格式不正确**（应以 `sk-` 开头）"),
                gr.Accordion(open=True),
                auth_state,
            )
        try:
            ok = init_client(key)
            if ok:
                auth_state["personal_key_configured"] = True
                return (
                    gr.Markdown("✅ **个人 API Key 配置成功**，无需访问码，可以开始对话。"),
                    gr.Accordion(open=False),
                    auth_state,
                )
            return gr.Markdown("❌ **API Key 配置失败，请重试**"), gr.Accordion(open=True), auth_state
        except Exception as e:
            return gr.Markdown(f"❌ **连接失败**：{e}"), gr.Accordion(open=True), auth_state

    if access_code:
        if not ACCESS_CODE:
            return (
                gr.Markdown("❌ **服务端尚未配置访问码**。请改用个人 DeepSeek API Key，或联系维护者。"),
                gr.Accordion(open=True),
                auth_state,
            )
        if access_code != ACCESS_CODE:
            return gr.Markdown("❌ **访问码不正确，请检查后重试。**"), gr.Accordion(open=True), auth_state
        auth_state["access_granted"] = True
        if SERVER_API_CONFIGURED:
            return (
                gr.Markdown("✅ **访问码验证成功**，将使用项目方配置的服务端 API。"),
                gr.Accordion(open=False),
                auth_state,
            )
        return (
            gr.Markdown("⚠️ **访问码正确，但服务端未配置 API Key。** 请改用个人 DeepSeek API Key。"),
            gr.Accordion(open=True),
            auth_state,
        )

    return (
        gr.Markdown("⚠️ **请输入访问码，或输入你自己的 DeepSeek API Key。**"),
        gr.Accordion(open=True),
        auth_state,
    )


def get_start_status(auth_state):
    """页面加载时显示进入方式说明。"""
    auth_state = dict(auth_state or empty_auth_state())
    if auth_state.get("personal_key_configured"):
        return gr.Markdown("✅ **个人 API Key 已配置**，可以开始对话。"), gr.Accordion(open=False), auth_state
    if auth_state.get("access_granted") and SERVER_API_CONFIGURED:
        return gr.Markdown("✅ **访问码已验证**，可以开始对话。"), gr.Accordion(open=False), auth_state
    if SERVER_API_CONFIGURED:
        return (
            gr.Markdown("ℹ️ **有访问码的新生可输入访问码使用；已有 DeepSeek API Key 的用户可直接输入 Key。**"),
            gr.Accordion(open=True),
            auth_state,
        )
    return (
        gr.Markdown("⚠️ **服务端未配置 API Key。** 请输入个人 DeepSeek API Key，或联系维护者配置服务端 Key。"),
        gr.Accordion(open=True),
        auth_state,
    )


def can_chat(auth_state):
    auth_state = dict(auth_state or empty_auth_state())
    return auth_state.get("personal_key_configured") or (
        auth_state.get("access_granted") and SERVER_API_CONFIGURED
    )


def missing_access_message():
    if SERVER_API_CONFIGURED:
        return (
            "⚠️ **请先输入访问码或个人 API Key。**\n\n"
            "有访问码的新生可以在上方输入访问码开始使用；已有 DeepSeek API Key 的用户可以直接输入自己的 Key，无需访问码。"
        )
    return (
        "⚠️ **请先输入个人 DeepSeek API Key。**\n\n"
        "当前服务端未配置项目方 API Key，访问码无法直接调用模型。"
    )


def save_api_key(api_key):
    """兼容旧的 API Key 配置入口。"""
    return submit_access("", api_key, empty_auth_state())


def get_api_status():
    """兼容旧的页面加载状态入口。"""
    return get_start_status(empty_auth_state())


# ===== 对话逻辑 =====

def handle_chat(history, message, auth_state):
    """处理一轮对话：添加用户消息 + 调用 LLM 生成回复。"""
    history = list(history) if history else []
    message = message.strip()
    if not message:
        return history, ""
    if not isinstance(history, list):
        history = []

    history.append({"role": "user", "content": message})

    if not can_chat(auth_state):
        history.append({"role": "assistant", "content": missing_access_message()})
        return history, ""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[:-1]:
        messages.append({"role": h["role"], "content": h["content"]})
    related = retrieve(
        message,
        chunks,
        top_k=RETRIEVAL_TOP_K,
        mode=RETRIEVAL_MODE,
    )

    if not related:
        history.append({"role": "assistant", "content": NO_RELEVANT_ANSWER})
        return history, ""

    prompt = build_prompt(message, related)
    messages.append({"role": "user", "content": prompt})

    try:
        reply = chat(messages)
    except Exception as e:
        reply = f"❌ **调用失败**：{e}\n\n请检查 API Key 是否正确、账户是否有余额。"

    history.append({"role": "assistant", "content": reply})
    return history, ""


# ===== 界面构建 =====

with gr.Blocks(title="浙大智能问答助手", fill_height=True) as demo:
    auth_state = gr.State(empty_auth_state())
    gr.HTML(HEADER_HTML)

    # ━━━━ 开始使用区 ━━━━
    api_accordion = gr.Accordion("⚙️ 开始使用", open=True, elem_classes="api-accordion")
    with api_accordion:
        api_status = gr.Markdown("")
        gr.Markdown(
            "有访问码的新生：输入访问码开始使用。已有 DeepSeek API Key：可直接输入 API Key，无需访问码。"
        )
        with gr.Row(elem_classes="api-row"):
            access_code_input = gr.Textbox(
                label="访问码",
                placeholder="输入课程/项目方提供的访问码",
                type="password",
                scale=3,
            )
            api_key_input = gr.Textbox(
                label="个人 DeepSeek API Key（可选）",
                placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                type="password",
                scale=4,
            )
            save_btn = gr.Button("开始使用", variant="primary", scale=1, min_width=120)
        gr.Markdown(
            "> 若同时填写访问码和个人 API Key，将优先使用个人 API Key。"
            "个人 Key 仅在当前会话有效，不会被存储到磁盘。"
        )

    # ━━━━ 聊天区 ━━━━
    chatbot = gr.Chatbot(
        value=[],
        elem_classes="chatbot-wrapper",
        layout="bubble",
        height=500,
        scale=1,
    )

    with gr.Row(elem_classes="input-row"):
        msg = gr.Textbox(
            placeholder="输入你的问题，比如：紫金港有什么好吃的？",
            show_label=False,
            scale=9,
            container=False,
        )
        send_btn = gr.Button("发送", variant="primary", scale=1, elem_classes="send-btn")

    gr.Markdown("##### 💡 试试这些问题")
    gr.Examples(
        examples=[
            "图书馆怎么预约？",
            "校医院牙科怎么样？",
            "紫金港有什么好吃的？",
            "新生宿舍条件如何？",
            "浙大附近有哪些好玩的地方？",
        ],
        inputs=msg,
        label="",
        examples_per_page=5,
    )

    gr.HTML(FOOTER_HTML)

    # ━━━━ 事件绑定 ━━━━

    demo.load(
        fn=get_start_status,
        inputs=auth_state,
        outputs=[api_status, api_accordion, auth_state],
    )

    save_btn.click(
        fn=submit_access,
        inputs=[access_code_input, api_key_input, auth_state],
        outputs=[api_status, api_accordion, auth_state],
    )

    msg.submit(
        fn=handle_chat,
        inputs=[chatbot, msg, auth_state],
        outputs=[chatbot, msg],
    )

    send_btn.click(
        fn=handle_chat,
        inputs=[chatbot, msg, auth_state],
        outputs=[chatbot, msg],
    )

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, inbrowser=True)
