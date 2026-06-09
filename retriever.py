"""知识库检索 —— 关键词 + 语义向量（进阶）。

支持两种模式：
  keyword  - 中文2-gram + Jaccard（基础版，无需额外依赖）
  semantic - sentence-transformers 语义向量检索（进阶版）
"""
import os
import glob
import hashlib
import re
import pickle

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def _project_path(path):
    """把项目资源的相对路径固定解析到代码所在目录。"""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_DIR, path)


# 语义检索（可选依赖）
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False


LOCATION_SYNONYM_GROUPS = [
    ("紫金港", "紫金港校区", "zjg"),
    ("玉泉", "玉泉校区", "yq"),
    ("西溪", "西溪校区", "xx"),
    ("华家池", "华家池校区", "hjc"),
    ("之江", "之江校区", "zj"),
    ("海宁", "海宁校区", "国际校区", "haining"),
    ("杭州", "杭州市", "市区"),
]

TOPIC_SYNONYM_GROUPS = [
    ("新生", "大一", "入学", "报到", "迎新", "开学", "新生报到", "入学指南"),
    ("图书馆", "图书馆预约", "座位预约", "预约座位", "研讨间", "自习室"),
    ("借书", "还书", "续借", "馆际互借", "文献传递"),
    ("校医院", "浙大校医院", "医院", "看病", "挂号", "预约挂号"),
    ("牙科", "口腔科", "补牙", "拔牙", "洗牙", "牙医"),
    ("医保", "医疗保险", "大学生医保", "报销", "保险"),
    ("心理", "心理咨询", "心理健康", "情绪", "焦虑", "压力"),
    ("宿舍", "寝室", "住宿", "宿管", "生活园区", "楼栋"),
    ("大功率", "大功率电器", "违章电器", "电吹风", "锅", "热得快"),
    ("空调", "制冷", "制热", "空调费", "遥控器"),
    ("热水", "洗澡", "浴室", "澡堂", "淋浴"),
    ("洗衣", "洗衣机", "烘干机", "洗衣房", "烘干"),
    ("食堂", "餐厅", "吃饭", "好吃", "美食", "夜宵"),
    ("外卖", "点外卖", "配送", "取外卖"),
    ("超市", "便利店", "全家", "罗森", "购物"),
    ("校园卡", "饭卡", "一卡通", "学生卡", "校卡"),
    ("补办", "挂失", "丢了", "遗失", "找回"),
    ("电动车", "电瓶车", "自行车", "共享单车", "买车", "骑行"),
    ("充电", "充电桩", "充电站", "充电柜", "电瓶"),
    ("交通", "公交", "地铁", "校车", "班车", "打车", "通勤", "坐车", "路线", "怎么去", "出行"),
    ("军训", "新生军训", "训练", "训服"),
    ("奖学金", "奖助学金", "助学金", "评奖评优"),
    ("贫困生", "困难生", "家庭经济困难", "资助", "勤工助学"),
    ("转专业", "转专", "跨专业", "专业分流"),
    ("绩点", "记点", "GPA", "学分绩点"),
    ("综测", "综合测评", "综素", "德育分", "加分"),
    ("四级", "六级", "CET4", "CET-4", "CET6", "CET-6", "英语等级考试"),
    ("选课", "抢课", "退课", "补退选", "培养方案"),
    ("教务", "教务系统", "现代教务", "成绩", "课表", "考试安排"),
    ("挂科", "不及格", "补考", "重修", "缓考", "旷考"),
    ("学在浙大", "学在", "课程平台", "作业", "网课"),
    ("教材", "课本", "买书", "二手书", "教材费"),
    ("快递", "驿站", "菜鸟", "取件"),
    ("社团", "学生组织", "招新", "百团大战"),
    ("活动", "讲座", "报名", "志愿者", "志愿服务"),
    ("体育", "体育课", "体测", "体育测试", "跑步", "晨跑"),
    ("健身", "健身房", "体育馆", "场馆预约", "羽毛球", "篮球", "游泳"),
    ("VPN", "WebVPN", "校网", "校园网", "zjuwlan", "无线网", "网络认证"),
    ("统一身份认证", "统一认证", "浙大通行证", "账号", "密码", "通行证"),
    ("打印", "复印", "扫描", "打印店", "文印"),
    ("租房", "找房", "短租", "长租", "合租", "整租", "房东", "中介"),
    ("保研", "推免", "免试", "夏令营", "预推免"),
    ("考研", "研究生考试", "复习", "备考"),
    ("实习", "就业", "求职", "简历", "招聘"),
    ("竞赛", "比赛", "大创", "科研", "项目"),
    ("实验", "实验课", "实验室", "lab"),
    ("请假", "销假", "病假", "事假", "假条"),
    ("证件", "学生证", "身份证", "护照", "证明", "在读证明"),
    ("邮箱", "浙大邮箱", "邮件", "mail"),
]

SYNONYM_GROUPS = LOCATION_SYNONYM_GROUPS + TOPIC_SYNONYM_GROUPS
TOPIC_BOOST_GROUPS = TOPIC_SYNONYM_GROUPS

CAMPUS_SCOPE_TERMS = (
    "浙大", "浙江大学", "学校", "校园", "校内", "校区", "学院",
    "紫金港", "玉泉", "西溪", "华家池", "之江", "海宁",
    "附近", "周边",
)

AMBIGUOUS_SCOPE_TERMS = {
    "医院", "看病", "挂号", "预约挂号", "保险",
    "账号", "密码", "项目", "比赛", "报名",
}

QUERY_STOP_TOKENS = {
    "请问", "求问", "有没有", "有没", "没有", "哪里", "哪儿", "在哪",
    "怎么", "么用", "如何", "什么", "有什", "么办", "咋办", "为什",
    "哪个", "哪家", "多少", "可以", "能不", "能否", "吗", "呢", "啊",
}

BROAD_TITLE_TOKENS = {
    "新生", "大一", "校园", "校区", "浙大", "紫金港", "zjg", "玉泉",
    "yq", "西溪", "华家池", "之江", "海宁", "宿舍", "寝室", "住宿",
}

KEYWORD_MIN_SCORE = 50.0
SEMANTIC_MIN_SCORE = 0.35


def _knowledge_fingerprint(chunks):
    """根据知识条目的来源和内容生成稳定指纹，用于校验向量缓存。"""
    digest = hashlib.sha256()
    for chunk in chunks:
        for key in ("source", "title", "text"):
            digest.update(str(chunk.get(key, "")).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def is_query_in_scope(query):
    """判断问题是否属于校园生活知识库覆盖范围。"""
    query = (query or "").strip()
    if not query:
        return False

    query_lower = query.lower()
    matched_terms = {
        term
        for group in TOPIC_SYNONYM_GROUPS
        for term in group
        if term.lower() in query_lower
    }
    if any(term not in AMBIGUOUS_SCOPE_TERMS for term in matched_terms):
        return True

    return any(term.lower() in query_lower for term in CAMPUS_SCOPE_TERMS)


def _expand_query(query):
    """基于校园生活同义词表扩展检索 query。"""
    expanded = [query]
    seen = {query.lower()}

    query_lower = query.lower()
    for group in SYNONYM_GROUPS:
        if any(term.lower() in query_lower for term in group):
            for term in group:
                key = term.lower()
                if key not in seen:
                    expanded.append(term)
                    seen.add(key)

    return " ".join(expanded)


def _matched_topic_groups(query):
    """返回 query 命中的主题同义词组，用于标题重排序加分。"""
    query_lower = query.lower()
    return [
        group for group in TOPIC_BOOST_GROUPS
        if any(term.lower() in query_lower for term in group)
    ]


def _tokenize(text):
    """中文分词：去标点 + 2-gram。"""
    text = text.lower()
    text = re.sub(r'[^一-鿿\w]', ' ', text)
    words = [w for w in text.split() if len(w) >= 1]
    bigrams = []
    for w in words:
        if len(w) >= 2:
            bigrams.extend(w[i:i+2] for i in range(len(w)-1))
    return set(words + bigrams)


def _filter_query_tokens(tokens):
    """去掉泛问题词，避免“怎么/什么/请问”等通用问法主导排序。"""
    return {t for t in tokens if t not in QUERY_STOP_TOKENS}


def load_knowledge(folder="knowledge"):
    """读取 knowledge/ 下所有 .txt 文件，每个文件作为一个知识块。"""
    folder = _project_path(folder)
    chunks = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            # 提取标题用于来源标注
            title = ""
            for line in text.split("\n"):
                if line.startswith("【标题】"):
                    title = line.replace("【标题】", "").strip()
                    break
            chunks.append({"source": name, "title": title, "text": text})
    if not chunks:
        raise RuntimeError(
            f"知识库为空或不存在：{folder}。"
            "请确认 knowledge 目录中包含 UTF-8 编码的 .txt 文件。"
        )
    return chunks


def score_keyword_chunks(query, chunks):
    """返回关键词检索的打分结果，按分数从高到低排序。"""
    expanded_query = _expand_query(query)
    matched_topic_groups = _matched_topic_groups(query)
    original_tokens = _filter_query_tokens(_tokenize(query))
    q_tokens = _filter_query_tokens(_tokenize(expanded_query))

    def score(chunk):
        text = chunk["text"]
        text_lower = text.lower()
        title = chunk.get("title", "")
        title_lower = title.lower()
        title_tokens = _tokenize(title)
        body_tokens = _tokenize(text)
        original_title_hits = len(original_tokens & title_tokens)
        specific_original_title_hits = len(
            (original_tokens - BROAD_TITLE_TOKENS) & title_tokens
        )
        title_hits = len(q_tokens & title_tokens)
        topic_title_hits = sum(
            1 for group in matched_topic_groups
            if any(term.lower() in title_lower for term in group)
        )
        if q_tokens | body_tokens:
            jaccard = len(q_tokens & body_tokens) / len(q_tokens | body_tokens)
        else:
            jaccard = 0
        phrase_bonus = sum(1 for t in q_tokens if len(t) >= 2 and t in text_lower)
        return (
            specific_original_title_hits * 24
            + original_title_hits * 8
            + title_hits * 10
            + topic_title_hits * 36
            + jaccard * 3
            + phrase_bonus * 0.5
        )

    return sorted(
        ((score(chunk), chunk) for chunk in chunks),
        key=lambda item: item[0],
        reverse=True,
    )


def retrieve_keyword(query, chunks, top_k=3, min_score=KEYWORD_MIN_SCORE):
    """关键词检索：同义词扩展 + 2-gram + Jaccard + 标题加权 + 低相关过滤。"""
    if not is_query_in_scope(query):
        return []

    ranked = score_keyword_chunks(query, chunks)
    return [chunk for score, chunk in ranked if score >= min_score][:top_k]


# ===== 语义向量检索 =====

class SemanticRetriever:
    """基于 sentence-transformers 的语义向量检索。

    首次使用会自动下载模型（约 120MB），并将嵌入向量缓存到磁盘。
    """

    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2",
                 cache_dir="knowledge"):
        if not HAS_SBERT:
            raise ImportError(
                "请先安装 sentence-transformers: pip install sentence-transformers")
        self.model_name = model_name
        self.cache_path = os.path.join(
            _project_path(cache_dir),
            ".embeddings_cache.pkl",
        )
        self.model = None
        self.chunks = None
        self.embeddings = None

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def build_index(self, chunks):
        """对所有知识块编码，存入缓存。"""
        self.chunks = chunks
        self._load_model()

        texts = [c["title"] + " " + c["text"][:500] for c in chunks]
        print(f"  正在向量化 {len(texts)} 条知识条目（首次较慢，约1-2分钟）...")
        self.embeddings = self.model.encode(
            texts, show_progress_bar=True, convert_to_numpy=True)

        # 保存缓存
        with open(self.cache_path, "wb") as f:
            pickle.dump({
                "model_name": self.model_name,
                "knowledge_fingerprint": _knowledge_fingerprint(chunks),
                "embeddings": self.embeddings,
            }, f)
        print(f"  向量缓存已保存到 {self.cache_path}")

    def load_cache(self, chunks):
        """尝试从缓存加载已编码的向量。"""
        self.chunks = chunks
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    data = pickle.load(f)
                if data.get("model_name") == self.model_name and \
                   data.get("knowledge_fingerprint") == \
                   _knowledge_fingerprint(chunks) and \
                   len(data["embeddings"]) == len(chunks):
                    self.embeddings = data["embeddings"]
                    print(f"  已从缓存加载 {len(self.embeddings)} 条向量")
                    return True
            except Exception:
                pass
        return False

    def search(self, query, top_k=3, min_score=SEMANTIC_MIN_SCORE):
        """语义检索：余弦相似度排序。"""
        if self.embeddings is None:
            return retrieve_keyword(query, self.chunks, top_k)

        self._load_model()
        q_vec = self.model.encode([query], convert_to_numpy=True)[0]

        # 余弦相似度
        from numpy import dot
        from numpy.linalg import norm
        scores = dot(self.embeddings, q_vec) / \
            (norm(self.embeddings, axis=1) * norm(q_vec) + 1e-10)

        ranked = sorted(zip(scores, self.chunks),
                        key=lambda x: x[0], reverse=True)
        return [c for score, c in ranked if score >= min_score][:top_k]


# ===== 统一检索接口 =====

_semantic_retriever = None


def retrieve(query, chunks, top_k=3, mode="keyword"):
    """统一检索接口。

    mode: "keyword" | "semantic"
    如果 semantic 模式初始化失败（如 HuggingFace 不可访问），自动回退 keyword。
    """
    if not is_query_in_scope(query):
        return []

    if mode == "semantic" and HAS_SBERT:
        global _semantic_retriever
        try:
            if _semantic_retriever is None:
                _semantic_retriever = SemanticRetriever()
                if not _semantic_retriever.load_cache(chunks):
                    _semantic_retriever.build_index(chunks)
            return _semantic_retriever.search(_expand_query(query), top_k)
        except Exception as e:
            print(f"  [语义检索初始化失败: {e}]")
            print(f"  [自动回退到关键词模式]")

    return retrieve_keyword(query, chunks, top_k)
