"""多轮对话的检索上下文增强工具。"""

STRONG_CONTEXT_HINTS = (
    "这个", "那个", "这些", "那些", "它", "那里", "上面", "前面",
    "刚才", "继续", "详细", "具体", "展开", "补充", "还有", "另外",
    "那", "呢",
)

WEAK_FOLLOWUP_HINTS = (
    "吗", "需要", "可以", "多久", "多少", "几个",
    "怎么去", "在哪里", "在哪", "流程", "预约", "费用", "价格",
)

EXPLICIT_TOPIC_HINTS = (
    "图书馆", "自习室", "校医院", "医院", "牙科", "医保", "心理",
    "宿舍", "寝室", "空调", "热水", "洗澡", "洗衣", "大功率",
    "食堂", "餐厅", "外卖", "超市", "校园卡", "补办", "挂失",
    "电动车", "自行车", "充电", "交通", "公交", "地铁", "校车", "军训",
    "奖学金", "助学金", "贫困生", "资助", "转专业", "绩点", "综测",
    "四级", "六级", "选课", "教务", "学在", "教材", "快递", "社团",
    "活动", "体育", "体测", "健身", "VPN", "WebVPN", "校网", "校园网",
    "统一身份认证", "打印", "租房", "保研",
    "考研", "实习", "竞赛", "实验", "请假", "证件", "邮箱",
    "紫金港", "玉泉", "西溪", "华家池", "之江", "海宁",
    "学校", "校园", "校内", "校区", "学院", "附近", "周边",
)

SOURCE_TITLE_LIMIT = 3
ANSWER_EXCERPT_CHARS = 160
MAX_CONTEXT_QUERY_CHARS = 500


def empty_context_state():
    return {
        "last_user_question": "",
        "last_source_titles": [],
        "last_answer_excerpt": "",
    }


def should_use_context(question):
    """判断当前问题是否像追问，追问才拼接上一轮上下文。"""
    question = (question or "").strip()
    if not question:
        return False
    if any(hint in question for hint in STRONG_CONTEXT_HINTS):
        return True
    if any(hint in question for hint in EXPLICIT_TOPIC_HINTS):
        return False
    return any(hint in question for hint in WEAK_FOLLOWUP_HINTS)


def extract_source_titles(related, limit=SOURCE_TITLE_LIMIT):
    """从检索结果里提取去重后的标题，作为下一轮最近话题。"""
    titles = []
    seen = set()
    for chunk in (related or [])[:limit]:
        title = (chunk.get("title") or chunk.get("source") or "").strip()
        if not title or title in seen:
            continue
        titles.append(title)
        seen.add(title)
    return titles


def build_context_query(question, context_state=None):
    """为短追问构造带上下文的检索 query，不额外调用大模型。"""
    question = (question or "").strip()
    state = dict(empty_context_state(), **(context_state or {}))

    if not should_use_context(question):
        return question

    parts = []
    last_question = state.get("last_user_question", "").strip()
    source_titles = state.get("last_source_titles") or []
    answer_excerpt = state.get("last_answer_excerpt", "").strip()

    if last_question:
        parts.append(last_question)
    if source_titles:
        parts.append(" ".join(source_titles))
    if answer_excerpt:
        parts.append(answer_excerpt[:ANSWER_EXCERPT_CHARS])

    parts.append(question)
    query = " ".join(part for part in parts if part).strip()
    return query[-MAX_CONTEXT_QUERY_CHARS:]


def update_context_state(question, related, answer=""):
    """用本轮成功检索到的内容更新下一轮检索上下文。"""
    answer = (answer or "").strip()
    return {
        "last_user_question": (question or "").strip(),
        "last_source_titles": extract_source_titles(related),
        "last_answer_excerpt": answer[:ANSWER_EXCERPT_CHARS],
    }
