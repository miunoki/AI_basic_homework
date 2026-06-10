"""CC98 论坛爬虫 —— 爬取目标版块的高价值帖子，生成知识库文本。"""
import requests
import re
import os
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== 配置 =====
# ACCESS_TOKEN 请从环境变量 CC98_ACCESS_TOKEN 或本地 config.py 读取
# 获取方式：浏览器登录 cc98.org → DevTools → Application → localStorage → access_token

def _load_token():
    """优先从环境变量读取，其次读取本地 config.py。"""
    env_token = os.getenv("CC98_ACCESS_TOKEN", "").strip()
    if env_token:
        return env_token
    try:
        from config import CC98_ACCESS_TOKEN
        token = str(CC98_ACCESS_TOKEN).strip()
        if token and "填入" not in token:
            return token
    except (ImportError, AttributeError):
        pass
    return ""

ACCESS_TOKEN = _load_token()
if ACCESS_TOKEN and not ACCESS_TOKEN.startswith("Bearer "):
    ACCESS_TOKEN = "Bearer " + ACCESS_TOKEN

HEADERS = {
    "Accept": "application/json",
    "Origin": "https://www.cc98.org",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
if ACCESS_TOKEN:
    HEADERS["Authorization"] = ACCESS_TOKEN

API_BASE = "https://api.cc98.org"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "knowledge_cc98")
TOPICS_PER_BOARD = 5000  # 每个版块最多取多少条（全量）
REPLIES_PER_TOPIC = 30  # 每个帖子最多取多少条回复
MIN_REPLIES = 1          # 至少要有回复才抓取正文

# 新生入学指南相关版块（全量爬取）
TARGET_BOARDS = {
    198: "新生宝典",
    100: "校园信息",
    184: "论坛指南",
    515: "住房信息",
    229: "美食天地",
    261: "健康贴士",
    101: "个性生活",
    226: "电脑医院",
    15: "体育运动",
}


def clean_bbcode(text):
    """清除 BBCode 标签，保留纯文本。"""
    if not text:
        return ""
    # 移除 [b][/b] [color=...][/color] [url][/url] [img][/img] 等
    text = re.sub(r'\[color[^\]]*\]', '', text)
    text = re.sub(r'\[/color\]', '', text)
    text = re.sub(r'\[b\]', '', text)
    text = re.sub(r'\[/b\]', '', text)
    text = re.sub(r'\[url[^\]]*\]', '', text)
    text = re.sub(r'\[/url\]', '', text)
    text = re.sub(r'\[img[^\]]*\]', '', text)
    text = re.sub(r'\[/img\]', '', text)
    text = re.sub(r'\[i\]', '', text)
    text = re.sub(r'\[/i\]', '', text)
    text = re.sub(r'\[quote[^\]]*\]', '', text)
    text = re.sub(r'\[/quote\]', '', text)
    text = re.sub(r'\[size[^\]]*\]', '', text)
    text = re.sub(r'\[/size\]', '', text)
    text = re.sub(r'\[list[^\]]*\]', '', text)
    text = re.sub(r'\[/list\]', '', text)
    text = re.sub(r'\[\*\]', '· ', text)
    # 移除其他所有 BBCode 标签
    text = re.sub(r'\[[^\]]+\]', '', text)
    # 清理多余空白
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def api_get(path, params=None, retries=3):
    """封装 API 调用，带重试。"""
    if not ACCESS_TOKEN:
        raise RuntimeError(
            "未配置 CC98_ACCESS_TOKEN，请在 config.py 或环境变量中设置后重试。"
        )

    url = f"{API_BASE}{path}"
    for attempt in range(retries):
        try:
            resp = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (401, 403):
                raise RuntimeError(
                    "CC98 Access Token 无效或已过期，请重新获取后重试。"
                )

            print(
                f"  API {resp.status_code}: {resp.url} "
                f"(attempt {attempt + 1})"
            )
            retry_after = resp.headers.get("Retry-After")
            wait_seconds = (
                float(retry_after)
                if retry_after and retry_after.isdigit()
                else 1 + attempt
            )
        except RuntimeError:
            raise
        except (requests.RequestException, ValueError) as e:
            print(f"  Error: {e} (attempt {attempt + 1})")
            wait_seconds = 1 + attempt

        if attempt < retries - 1:
            time.sleep(wait_seconds)
    return None


def fetch_board_topics(board_id, max_count=TOPICS_PER_BOARD):
    """获取指定版块的帖子列表（按热度/最新排序）"""
    all_topics = []
    for offset in range(0, max_count, 20):
        batch = api_get(f"/board/{board_id}/topic", {"from": offset, "size": min(20, max_count - offset)})
        if not batch:
            break
        all_topics.extend(batch)
        if len(batch) < 20:
            break
        time.sleep(0.15)  # 温和限速
    return all_topics


def fetch_topic_posts(topic_id, max_replies=REPLIES_PER_TOPIC):
    """获取帖子的所有回复（含楼主正文）"""
    all_posts = []
    for offset in range(0, max_replies, 20):
        batch = api_get(f"/topic/{topic_id}/post", {"from": offset, "size": min(20, max_replies - offset)})
        if not batch:
            break
        all_posts.extend(batch)
        if len(batch) < 20:
            break
        time.sleep(0.2)
    return all_posts


def format_post_as_qa(topic, posts):
    """将帖子格式化为「问题 + 回答」的知识库条目。"""
    title = topic.get("title", "无标题")
    board = TARGET_BOARDS.get(topic.get("boardId"), "未知版面")

    # 楼主帖 = 问题 / 核心内容
    lz_post = next((p for p in posts if p.get("isLZ")), None)
    question = clean_bbcode(lz_post.get("content", "")) if lz_post else ""

    # 非楼主的帖子 = 回答
    answers = []
    for p in posts:
        if not p.get("isLZ") and not p.get("isDeleted"):
            content = clean_bbcode(p.get("content", ""))
            if content and len(content) > 10:  # 过滤太短的水帖
                answers.append(f"[{p.get('userName', '匿名')} 的回答]\n{content}")

    # 组装
    parts = [f"【来源】CC98论坛「{board}」版块"]
    parts.append(f"【标题】{title}")
    if question:
        parts.append(f"【帖子正文】\n{question}")

    if answers:
        parts.append(f"\n【精选回复】（共 {len(answers)} 条）")
        for i, ans in enumerate(answers[:10], 1):  # 最多保留 10 条高质量回复
            parts.append(f"\n--- 回复 {i} ---\n{ans}")

    return "\n\n".join(parts)


def sort_topics_by_value(topics):
    """按信息价值排序：精华帖优先，其次按回复数+收藏数+热度"""
    def score(t):
        return (
            t.get("bestState", 0) * 100 +
            t.get("replyCount", 0) * 2 +
            t.get("favoriteCount", 0) * 3 +
            t.get("hitCount", 0) * 0.01
        )
    return sorted(topics, key=score, reverse=True)


def main():
    if not ACCESS_TOKEN:
        raise RuntimeError(
            "未配置 CC98_ACCESS_TOKEN，请在 config.py 或环境变量中设置后重试。"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    entry_num = 0

    for board_id, board_name in TARGET_BOARDS.items():
        print(f"\n{'='*60}")
        print(f"爬取版块 [{board_name}] (ID: {board_id})...")

        # 获取帖子列表
        topics = fetch_board_topics(board_id)
        print(f"  获取到 {len(topics)} 个帖子")

        if not topics:
            print(f"  [!] 版块无数据，跳过")
            continue

        # 按价值排序
        valued_topics = sort_topics_by_value(topics)[:TOPICS_PER_BOARD]
        print(f"  选取前 {len(valued_topics)} 个高价值帖子")

        # 过滤0回复帖子
        topics_with_replies = [t for t in valued_topics if t.get("replyCount", 0) >= MIN_REPLIES]
        print(f"  其中有回复的帖子: {len(topics_with_replies)} 个")

        # 逐个爬取帖子内容并直接保存为独立文件
        for i, topic in enumerate(topics_with_replies):
            tid = topic["id"]
            title = topic.get("title", "")[:50]
            print(f"  [{i+1}/{len(topics_with_replies)}] {title}...", end=" ", flush=True)

            posts = fetch_topic_posts(tid)
            if not posts:
                print("无回复，跳过")
                continue

            text = format_post_as_qa(topic, posts)

            # 直接保存为独立文件
            entry_num += 1
            safe_title = re.sub(r'[\\/*?:"<>|\n\r\t]', '', topic.get("title", "untitled"))[:50]
            fname = f"{entry_num:04d}_{board_name}_{safe_title}.txt"
            filepath = os.path.join(OUTPUT_DIR, fname)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"OK ({len(posts)} 条回复)")
            time.sleep(0.15)

        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"[OK] 爬取完成！")
    print(f"   总知识条目: {entry_num}")
    print(f"   输出目录: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"\n   将 knowledge_cc98/ 目录下的 .txt 文件复制到 knowledge/ 即可使用。")


if __name__ == "__main__":
    main()
