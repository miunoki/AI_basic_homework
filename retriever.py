"""知识库检索 —— 关键词 + 语义向量（进阶）。

支持两种模式：
  keyword  - 中文2-gram + Jaccard（基础版，无需额外依赖）
  semantic - sentence-transformers 语义向量检索（进阶版）
"""
import os
import glob
import re
import pickle

# 语义检索（可选依赖）
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False


def _tokenize(text):
    """中文分词：去标点 + 2-gram。"""
    text = re.sub(r'[^一-鿿\w]', ' ', text)
    words = [w for w in text.split() if len(w) >= 1]
    bigrams = []
    for w in words:
        if len(w) >= 2:
            bigrams.extend(w[i:i+2] for i in range(len(w)-1))
    return set(words + bigrams)


def load_knowledge(folder="knowledge"):
    """读取 knowledge/ 下所有 .txt 文件，每个文件作为一个知识块。"""
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
    return chunks


def retrieve_keyword(query, chunks, top_k=3):
    """关键词检索：2-gram + Jaccard + 标题加权。"""
    q_tokens = _tokenize(query)

    def score(chunk):
        text = chunk["text"]
        title_tokens = _tokenize(chunk.get("title", ""))
        body_tokens = _tokenize(text)
        title_hits = len(q_tokens & title_tokens)
        if q_tokens | body_tokens:
            jaccard = len(q_tokens & body_tokens) / len(q_tokens | body_tokens)
        else:
            jaccard = 0
        phrase_bonus = sum(1 for t in q_tokens if len(t) >= 2 and t in text)
        return title_hits * 10 + jaccard * 3 + phrase_bonus * 0.5

    return sorted(chunks, key=score, reverse=True)[:top_k]


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
        self.cache_path = os.path.join(cache_dir, ".embeddings_cache.pkl")
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
                   len(data["embeddings"]) == len(chunks):
                    self.embeddings = data["embeddings"]
                    print(f"  已从缓存加载 {len(self.embeddings)} 条向量")
                    return True
            except Exception:
                pass
        return False

    def search(self, query, top_k=3):
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
        return [c for _, c in ranked[:top_k]]


# ===== 统一检索接口 =====

_semantic_retriever = None


def retrieve(query, chunks, top_k=3, mode="keyword"):
    """统一检索接口。

    mode: "keyword" | "semantic"
    如果 semantic 模式初始化失败（如 HuggingFace 不可访问），自动回退 keyword。
    """
    if mode == "semantic" and HAS_SBERT:
        global _semantic_retriever
        try:
            if _semantic_retriever is None:
                _semantic_retriever = SemanticRetriever()
                if not _semantic_retriever.load_cache(chunks):
                    _semantic_retriever.build_index(chunks)
            return _semantic_retriever.search(query, top_k)
        except Exception as e:
            print(f"  [语义检索初始化失败: {e}]")
            print(f"  [自动回退到关键词模式]")

    return retrieve_keyword(query, chunks, top_k)
