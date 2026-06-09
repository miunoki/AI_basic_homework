"""DeepSeek API 统一封装 —— 支持运行时动态配置 API Key。"""
import os

BASE_URL = "https://api.deepseek.com"
_client = None


def _get_api_key():
    """尝试从 config.py 或环境变量读取 API Key。"""
    try:
        from config import DEEPSEEK_API_KEY
        key = DEEPSEEK_API_KEY
        if key and "在这" not in key and "填入" not in key:
            return key
    except ImportError:
        pass
    return os.getenv("DEEPSEEK_API_KEY", "")


def init_client(api_key=None):
    """初始化命令行和服务端共用的默认客户端。"""
    global _client
    key = api_key or _get_api_key()
    if key:
        _client = create_client(key)
        return True
    return False


def create_client(api_key):
    """创建独立客户端，不修改默认客户端。"""
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def is_configured():
    return _client is not None


# 尝试自动加载
init_client()


def _clean_text(text):
    if text is None:
        return ""
    if isinstance(text, list):
        return "\n".join(_clean_text(item) for item in text)
    if isinstance(text, dict):
        if text.get("type") == "text" and "text" in text:
            return _clean_text(text["text"])
        for key in ("content", "text", "value"):
            if key in text:
                return _clean_text(text[key])
        return str(text)
    if not isinstance(text, str):
        text = str(text)
    return ''.join(ch for ch in text if ord(ch) < 0xD800 or ord(ch) > 0xDFFF)


def _clean_messages(messages):
    return [
        {**msg, "content": _clean_text(msg["content"])} for msg in messages
    ]


def chat(messages, model="deepseek-chat", temperature=0.7,
         api_key=None, **kwargs):
    client = create_client(api_key) if api_key else _client
    if client is None:
        raise RuntimeError("请先在网页中配置 API Key（点击「API 配置」展开设置）。")
    messages = _clean_messages(messages)
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, **kwargs
    )
    return resp.choices[0].message.content


def ask(prompt, system="你是一个乐于助人的助手。", **kwargs):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    return chat(messages, **kwargs)
