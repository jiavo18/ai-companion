"""
===============================================================================
AI伴侣 —— 具备长期记忆、自我进化与情感感知的智能对话系统
===============================================================================
技术栈: Streamlit + LangChain + ChromaDB + DeepSeek API
作者: AI Companion Project
版本: 1.0.0
===============================================================================
"""

# ============================================================================
# 关键：必须在导入 chromadb 之前设置，解决 protobuf 兼容性问题
# 参考: https://developers.google.com/protocol-buffers/docs/news/2022-05-06
# ============================================================================
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import chromadb
import json
import uuid
import re
from datetime import datetime, timedelta
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 用于 embedding 的本地模型（轻量、免费、无需 API Key）
from chromadb.utils import embedding_functions

# 可选：情绪曲线图表
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# 环境变量支持
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# 页面配置
# ============================================================================
st.set_page_config(
    page_title="AI伴侣 · 懂你的智能伙伴",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# 自定义 CSS 样式 —— 仅通过类选择器微调，不触碰 Streamlit 内部 DOM
# ============================================================================
st.markdown("""
<style>
    /* 全局字体 */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    }
    /* 按钮微调 */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 常量定义
# ============================================================================

# —— 情感关键词库 ——
EMOTION_KEYWORDS = {
    "正面": [
        "开心", "高兴", "快乐", "幸福", "满足", "激动", "期待",
        "棒", "赞", "好", "喜欢", "爱", "太棒了", "优秀", "完美",
        "感谢", "谢谢", "哈哈", "嘿嘿", "嘻嘻", "不错", "厉害",
        "牛", "爽", "温暖", "感动", "惊喜", "欣慰", "自豪",
    ],
    "负面": [
        "难过", "伤心", "失落", "生气", "愤怒", "讨厌", "烦",
        "累", "疲惫", "焦虑", "害怕", "担心", "哭", "痛苦",
        "难受", "郁闷", "烦躁", "崩溃", "绝望", "无助", "委屈",
        "沮丧", "孤独", "寂寞", "失望", "后悔", "愧疚",
    ],
}

# —— 主动关心触发关键词 ——
CONCERN_PATTERNS = {
    "熬夜": "注意到你最近经常提到熬夜，身体是革命的本钱，记得早点休息。",
    "加班": "你最近似乎工作很忙，别忘了给自己留一些喘息的时间。",
    "累": "感觉你状态有些疲惫，要不要听听轻音乐放松一下？",
    "压力": "压力大的时候，深呼吸或者出去走走都会有帮助。",
    "焦虑": "你提到了焦虑，我想告诉你，这种感觉很正常，慢慢来。",
}

# —— 记忆遗忘阈值（天） ——
MEMORY_FORGET_DAYS = 30
MEMORY_WEAKEN_DAYS = 14

# —— 持久化路径 ——
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
LAST_ACTIVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_active.json")

# ============================================================================
# 资源缓存（使用 st.cache_resource 避免重复初始化）
# ============================================================================

@st.cache_resource
def get_embedding_function():
    """
    获取嵌入函数。
    使用 SentenceTransformer 的轻量中文支持模型，本地运行无需 API Key。
    首次运行会自动下载模型（约 80MB），后续使用缓存。
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )


@st.cache_resource
def get_chroma_client():
    """获取 ChromaDB 持久化客户端"""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


@st.cache_resource
def get_collections():
    """
    获取或创建 ChromaDB 集合。
    返回:
        memory_collection: 存储用户记忆
        feedback_collection: 存储用户反馈
    """
    client = get_chroma_client()
    ef = get_embedding_function()

    memory_collection = client.get_or_create_collection(
        name="user_memories",
        embedding_function=ef,
        metadata={"description": "用户长期记忆存储"},
    )

    feedback_collection = client.get_or_create_collection(
        name="feedback_collection",
        embedding_function=ef,
        metadata={"description": "用户反馈与风格偏好"},
    )

    return memory_collection, feedback_collection


def get_llm(api_key: str) -> ChatOpenAI:
    """
    获取 DeepSeek ChatOpenAI 实例。
    使用 st.cache_resource 缓存，仅 api_key 变化时重建。
    """
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0.72,
        max_tokens=1024,
        timeout=45,
    )

# ============================================================================
# API Key 获取
# ============================================================================

def get_api_key() -> str | None:
    """
    获取 DeepSeek API Key。
    优先级: 用户界面输入 > Streamlit Secrets > 环境变量(.env)
    """
    # 优先使用用户在界面中输入的 Key
    user_key = st.session_state.get("user_api_key", "").strip()
    if user_key:
        return user_key

    # 其次 Streamlit Cloud Secrets
    try:
        key = st.secrets.get("DEEPSEEK_API_KEY", "")
        if key:
            return key
    except (KeyError, FileNotFoundError):
        pass

    # 再次环境变量
    env_key = os.getenv("DEEPSEEK_API_KEY", "")
    if env_key:
        return env_key

    return None

# ============================================================================
# 情感分析模块
# ============================================================================

def detect_emotion(text: str) -> dict:
    """
    检测文本中的情感倾向。

    参数:
        text: 用户输入文本

    返回:
        {
            "polarity": "正面" | "负面" | "中性",
            "score": float (0~1, 情感强度),
            "matched_keywords": [匹配到的关键词列表],
        }
    """
    positive_count = 0
    negative_count = 0
    matched_pos = []
    matched_neg = []

    for keyword in EMOTION_KEYWORDS["正面"]:
        if keyword in text:
            count = text.count(keyword)
            positive_count += count
            matched_pos.append(keyword)

    for keyword in EMOTION_KEYWORDS["负面"]:
        if keyword in text:
            count = text.count(keyword)
            negative_count += count
            matched_neg.append(keyword)

    total = positive_count + negative_count

    if total == 0:
        return {"polarity": "中性", "score": 0.0, "matched_keywords": []}

    if positive_count > negative_count:
        score = positive_count / total
        return {"polarity": "正面", "score": score, "matched_keywords": matched_pos}
    elif negative_count > positive_count:
        score = negative_count / total
        return {"polarity": "负面", "score": score, "matched_keywords": matched_neg}
    else:
        return {"polarity": "中性", "score": 0.5, "matched_keywords": matched_pos + matched_neg}


def get_emotion_response_guide(polarity: str) -> str:
    """
    根据情感极性返回回应语气指引。
    """
    if polarity == "正面":
        return "用户情绪积极。请在回应中保持热情、活泼的语气，可以适当使用感叹号来表达共鸣。"
    elif polarity == "负面":
        return "用户情绪偏向负面。请使用温和、共情、安抚的语气。先表达理解与关心，再给出建议。避免过于轻快的表达。"
    else:
        return "用户情绪中性。请保持常规的友好、专业语气。"

# ============================================================================
# 长期记忆管理模块
# ============================================================================

def store_memory(content: str) -> str:
    """
    将一条记忆存入 ChromaDB。

    参数:
        content: 记忆文本内容

    返回:
        memory_id: 记忆的唯一标识
    """
    memory_collection, _ = get_collections()
    memory_id = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()

    memory_collection.add(
        ids=[memory_id],
        documents=[content],
        metadatas=[{
            "created_at": now,
            "last_accessed": now,
            "access_count": 1,
            "type": "user_memory",
        }],
    )

    return memory_id


def retrieve_memories(query: str, n_results: int = 3) -> list[dict]:
    """
    检索与当前查询最相关的记忆，并应用遗忘权重。

    参数:
        query: 查询文本
        n_results: 返回的记忆数量

    返回:
        [{"content": str, "created_at": str, "access_count": int, "distance": float}, ...]
    """
    memory_collection, _ = get_collections()
    now = datetime.now()

    # 检索候选记忆（多取一些用于遗忘权重排序）
    fetch_count = max(n_results * 3, 10)
    try:
        results = memory_collection.query(
            query_texts=[query],
            n_results=fetch_count,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    if not results["ids"] or not results["ids"][0]:
        return []

    memories = []
    for i, mem_id in enumerate(results["ids"][0]):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        # —— 遗忘权重计算 ——
        try:
            last_accessed = datetime.fromisoformat(meta.get("last_accessed", meta.get("created_at", now.isoformat())))
            days_since_access = (now - last_accessed).days

            # 超过遗忘阈值，加大距离（降低相关性）
            if days_since_access > MEMORY_FORGET_DAYS:
                distance *= 2.5
            elif days_since_access > MEMORY_WEAKEN_DAYS:
                distance *= 1.5
        except (ValueError, TypeError):
            pass

        memories.append({
            "id": mem_id,
            "content": doc,
            "created_at": meta.get("created_at", ""),
            "last_accessed": meta.get("last_accessed", ""),
            "access_count": meta.get("access_count", 1),
            "distance": distance,
        })

    # 按调整后的距离排序（距离越小越相关）
    memories.sort(key=lambda m: m["distance"])
    selected = memories[:n_results]

    # —— 更新访问记录 ——
    memory_collection, _ = get_collections()
    for mem in selected:
        try:
            memory_collection.update(
                ids=[mem["id"]],
                metadatas=[{
                    "last_accessed": now.isoformat(),
                    "access_count": mem["access_count"] + 1,
                }],
            )
        except Exception:
            pass

    return selected


def get_all_memories(limit: int = 20) -> list[dict]:
    """获取所有已存储的记忆（用于可视化面板）"""
    memory_collection, _ = get_collections()
    try:
        results = memory_collection.get(
            include=["documents", "metadatas"],
            limit=limit,
        )
    except Exception:
        return []

    if not results["ids"]:
        return []

    memories = []
    for i, mem_id in enumerate(results["ids"]):
        doc = results["documents"][i]
        meta = results["metadatas"][i]
        memories.append({
            "id": mem_id,
            "content": doc,
            "created_at": meta.get("created_at", ""),
            "access_count": meta.get("access_count", 1),
        })

    # 按创建时间倒序
    memories.sort(key=lambda m: m["created_at"], reverse=True)
    return memories


def delete_memory(memory_id: str) -> bool:
    """删除单条记忆"""
    try:
        memory_collection, _ = get_collections()
        memory_collection.delete(ids=[memory_id])
        return True
    except Exception:
        return False


def clear_all_memories() -> bool:
    """清空所有记忆"""
    try:
        memory_collection, feedback_collection = get_collections()
        # 获取所有 ID 并删除
        all_mem = memory_collection.get(include=[])
        if all_mem["ids"]:
            memory_collection.delete(ids=all_mem["ids"])
        all_fb = feedback_collection.get(include=[])
        if all_fb["ids"]:
            feedback_collection.delete(ids=all_fb["ids"])
        return True
    except Exception:
        return False

# ============================================================================
# 反馈学习模块
# ============================================================================

def store_feedback(content: str) -> str:
    """
    将用户反馈存入反馈集合。

    参数:
        content: 反馈内容（如"回答太长了，精简一点"）

    返回:
        feedback_id: 反馈标识
    """
    _, feedback_collection = get_collections()
    feedback_id = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()

    feedback_collection.add(
        ids=[feedback_id],
        documents=[content],
        metadatas=[{
            "created_at": now,
        }],
    )

    return feedback_id


def retrieve_feedback(n_results: int = 3) -> list[str]:
    """
    检索最新的用户反馈。
    使用一个通用查询词来获取所有反馈，按时间排序取最近 N 条。
    """
    _, feedback_collection = get_collections()
    try:
        results = feedback_collection.get(
            include=["documents", "metadatas"],
        )
    except Exception:
        return []

    if not results["ids"]:
        return []

    # 按创建时间排序
    items = []
    for i, fb_id in enumerate(results["ids"]):
        doc = results["documents"][i]
        meta = results["metadatas"][i]
        items.append({
            "id": fb_id,
            "content": doc,
            "created_at": meta.get("created_at", ""),
        })

    items.sort(key=lambda x: x["created_at"], reverse=True)
    return [item["content"] for item in items[:n_results]]


def parse_feedback_to_constraints(feedbacks: list[str]) -> str:
    """
    将用户反馈解析为系统约束文本。
    提取常见模式并生成具体的约束指令。
    """
    if not feedbacks:
        return ""

    constraints = []
    for fb in feedbacks:
        fb_lower = fb.lower()

        # 规则匹配
        if any(w in fb_lower for w in ["精简", "啰嗦", "太长", "简短", "简洁", "少说"]):
            constraints.append("回答应尽量简洁精炼，控制在100字以内，避免冗余表达。")
        elif any(w in fb_lower for w in ["详细", "多说", "展开", "深入"]):
            constraints.append("回答应更加详细深入，提供充分的解释和背景信息。")
        if any(w in fb_lower for w in ["温柔", "温和", "语气", "态度好"]):
            constraints.append("请使用更加温和、柔和的语气。")
        if any(w in fb_lower for w in ["幽默", "搞笑", "有趣"]):
            constraints.append("请在回答中适当加入幽默感，让对话更轻松。")
        if any(w in fb_lower for w in ["专业", "严谨", "认真"]):
            constraints.append("请保持专业严谨的表达风格。")

        # 如果没有匹配到规则，保留原文作为软约束
        if not constraints or constraints[-1] not in [
            "回答应尽量简洁精炼，控制在100字以内，避免冗余表达。",
            "回答应更加详细深入，提供充分的解释和背景信息。",
            "请使用更加温和、柔和的语气。",
            "请在回答中适当加入幽默感，让对话更轻松。",
            "请保持专业严谨的表达风格。",
        ]:
            constraints.append(f"用户偏好：{fb}")

    # 去重
    seen = set()
    unique_constraints = []
    for c in constraints:
        if c not in seen:
            seen.add(c)
            unique_constraints.append(c)

    return "\n".join(f"- {c}" for c in unique_constraints)

# ============================================================================
# 主动关心模块
# ============================================================================

def get_last_active_time() -> datetime | None:
    """从本地文件读取上次活跃时间"""
    try:
        if os.path.exists(LAST_ACTIVE_FILE):
            with open(LAST_ACTIVE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return datetime.fromisoformat(data["last_active"])
    except Exception:
        pass
    return None


def update_last_active_time():
    """更新本地活跃时间戳"""
    try:
        with open(LAST_ACTIVE_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_active": datetime.now().isoformat()}, f, ensure_ascii=False)
    except Exception:
        pass


def generate_proactive_care(conversation_history: list[dict]) -> str | None:
    """
    生成主动关心语句。

    参数:
        conversation_history: 最近的对话历史

    返回:
        关心语句文本，若无触发条件则返回 None
    """
    care_messages = []

    # 检查1: 距离上次对话的时间
    last_active = get_last_active_time()
    if last_active:
        hours_since = (datetime.now() - last_active).total_seconds() / 3600
        if hours_since > 24:
            days = int(hours_since / 24)
            care_messages.append(
                f"已经{days}天没聊了，最近过得怎么样？有什么想和我分享的吗？"
            )

    # 检查2: 从最近对话中检测压力/疲劳关键词
    if conversation_history:
        recent_texts = []
        for msg in conversation_history[-10:]:  # 最近10条
            if msg.get("role") == "user":
                recent_texts.append(msg.get("content", ""))

        combined_text = " ".join(recent_texts)

        for keyword, care_msg in CONCERN_PATTERNS.items():
            if keyword in combined_text:
                care_messages.append(care_msg)
                break  # 只取第一个匹配的关心模式

    if care_messages:
        return " ".join(care_messages)

    return None

# ============================================================================
# 系统提示词构建器
# ============================================================================

def build_system_prompt(
    memories: list[dict],
    feedback_constraints: str,
    emotion_guide: str,
    proactive_care: str | None,
) -> str:
    """
    动态构建系统提示词，整合所有上下文信息。

    参数:
        memories: 检索到的用户记忆
        feedback_constraints: 反馈约束文本
        emotion_guide: 情感回应指引
        proactive_care: 主动关心语句（可选）

    返回:
        完整的系统提示词
    """
    prompt_parts = [
        "你是一位善解人意、温和体贴的AI伴侣。你的名字是「禾苗」。",
        "",
        "## 核心原则",
        "- 你是用户值得信赖的朋友，而非冷冰冰的工具。",
        "- 回答真诚自然，像一个了解你的朋友在聊天。",
        "- 如果用户询问建议，结合你对用户的了解给出个性化建议。",
        "- 不要使用过于机械或模板化的表达。",
        "",
    ]

    # —— 用户反馈约束 ——
    if feedback_constraints:
        prompt_parts.append("## 用户偏好（请严格遵守）")
        prompt_parts.append(feedback_constraints)
        prompt_parts.append("")

    # —— 相关记忆 ——
    if memories:
        prompt_parts.append("## 关于用户的重要记忆")
        for i, mem in enumerate(memories, 1):
            created = mem.get("created_at", "")[:10] if mem.get("created_at") else ""
            prompt_parts.append(f"{i}. {mem['content']}（记录于 {created}）")
        prompt_parts.append("请在回答中自然地融入这些记忆，让用户感受到你真正记得关于TA的事。")
        prompt_parts.append("")

    # —— 情感指引 ——
    prompt_parts.append("## 当前回应策略")
    prompt_parts.append(emotion_guide)
    prompt_parts.append("")

    # —— 主动关心 ——
    if proactive_care:
        prompt_parts.append("## 主动关心")
        prompt_parts.append(f"请在回答中自然地加入以下关心：{proactive_care}")
        prompt_parts.append("不要生硬地插入，要自然地融入回答中。")
        prompt_parts.append("")

    return "\n".join(prompt_parts)

# ============================================================================
# 会话状态初始化
# ============================================================================

def init_session_state():
    """初始化所有会话状态变量"""
    defaults = {
        "messages": [],                # 对话历史 [{"role": "user"|"assistant", "content": str}, ...]
        "emotion_history": [],         # 情绪记录 [{"time": str, "polarity": str, "score": float}, ...]
        "api_key_verified": False,     # API Key 是否已验证
        "memory_count": 0,             # 当前记忆总数
        "user_api_key": "",            # 用户在界面输入的 API Key
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# ============================================================================
# 消息处理主逻辑
# ============================================================================

def process_user_message(user_input: str, llm: ChatOpenAI) -> str:
    """
    处理用户输入并生成 AI 回复。

    处理流程:
    1. 判断是否为记忆/反馈指令
    2. 情感检测
    3. 记忆检索
    4. 反馈检索
    5. 主动关心生成
    6. 构建系统提示词
    7. 调用 LLM
    """
    user_input = user_input.strip()

    # —— 步骤0: 指令识别 ——
    # 记忆指令: "记住：xxx" 或 "记住:xxx"
    remember_match = re.match(r"^记住[：:]\s*(.+)", user_input)
    if remember_match:
        content = remember_match.group(1).strip()
        if content:
            mem_id = store_memory(content)
            update_last_active_time()
            st.session_state.memory_count += 1
            return f"已牢记：「{content}」。以后你需要的时候，我会想起这个信息。"

    # 反馈指令: "反馈：xxx" 或 "反馈:xxx"
    feedback_match = re.match(r"^反馈[：:]\s*(.+)", user_input)
    if feedback_match:
        content = feedback_match.group(1).strip()
        if content:
            fb_id = store_feedback(content)
            update_last_active_time()
            return f"已收到你的反馈：「{content}」。我会在后续对话中调整我的表达方式。"

    # —— 步骤1: 情感检测 ——
    emotion_result = detect_emotion(user_input)
    emotion_guide = get_emotion_response_guide(emotion_result["polarity"])

    # 记录情绪
    st.session_state.emotion_history.append({
        "time": datetime.now().isoformat(),
        "polarity": emotion_result["polarity"],
        "score": emotion_result["score"],
    })
    # 只保留最近 100 条情绪记录
    if len(st.session_state.emotion_history) > 100:
        st.session_state.emotion_history = st.session_state.emotion_history[-100:]

    # —— 步骤2: 记忆检索 ——
    memories = retrieve_memories(user_input, n_results=3)

    # —— 步骤3: 反馈检索 ——
    feedbacks = retrieve_feedback(n_results=3)
    feedback_constraints = parse_feedback_to_constraints(feedbacks)

    # —— 步骤4: 主动关心 ——
    proactive_care = generate_proactive_care(st.session_state.messages)

    # —— 步骤5: 构建系统提示词 ——
    system_prompt = build_system_prompt(
        memories=memories,
        feedback_constraints=feedback_constraints,
        emotion_guide=emotion_guide,
        proactive_care=proactive_care,
    )

    # —— 步骤6: 构建消息列表 ——
    messages = [SystemMessage(content=system_prompt)]

    # 加入最近 20 轮对话历史
    recent_history = st.session_state.messages[-20:]
    for msg in recent_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # 当前用户消息
    messages.append(HumanMessage(content=user_input))

    # —— 步骤7: 调用 LLM ——
    try:
        response = llm.invoke(messages)
        reply = response.content
    except Exception as e:
        reply = f"抱歉，我暂时无法回应。请检查网络连接或API配置。（错误详情：{str(e)}）"

    # —— 步骤8: 更新活跃时间 ——
    update_last_active_time()

    return reply

# ============================================================================
# 验证 API Key 是否有效
# ============================================================================

def verify_api_key(api_key: str) -> bool:
    """快速验证 API Key 是否有效"""
    try:
        llm = get_llm(api_key)
        response = llm.invoke([HumanMessage(content="你好，请回复一个字：好")])
        return bool(response.content.strip())
    except Exception:
        return False

# ============================================================================
# 界面渲染 —— 侧边栏
# ============================================================================

def render_sidebar():
    """渲染侧边栏：API配置 + 记忆面板 + 情绪图表"""

    with st.sidebar:
        # —— 品牌标识 —— 使用原生组件，避免 DOM 冲突
        st.title("禾 苗")
        st.caption("AI Companion")

        st.markdown("---")

        # —— API Key 配置 ——
        st.markdown("### API 配置")

        # 检测是否已有 Secrets 中的 Key
        has_secret_key = False
        try:
            has_secret_key = bool(st.secrets.get("DEEPSEEK_API_KEY", ""))
        except (KeyError, FileNotFoundError):
            pass

        if has_secret_key:
            st.success("已使用云端配置的 API Key")
        elif st.session_state.get("user_api_key", "") and st.session_state.api_key_verified:
            st.success("API 已连接")
        else:
            with st.form("api_key_form", clear_on_submit=False):
                user_key_input = st.text_input(
                    "DeepSeek API Key",
                    type="password",
                    value=st.session_state.get("user_api_key", ""),
                    placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    help="输入后点击「保存」即可使用。Key 仅保存在当前会话中。",
                )
                col_save, col_clear = st.columns(2)
                with col_save:
                    save_clicked = st.form_submit_button("保存", use_container_width=True, type="primary")
                with col_clear:
                    clear_clicked = st.form_submit_button("清除", use_container_width=True)

            if save_clicked and user_key_input.strip():
                st.session_state.user_api_key = user_key_input.strip()
                st.session_state.api_key_verified = False  # 触发重新验证
                st.rerun()

            if clear_clicked:
                st.session_state.user_api_key = ""
                st.session_state.api_key_verified = False
                st.rerun()

            # 验证用户输入的 Key
            if st.session_state.get("user_api_key", "") and not st.session_state.api_key_verified:
                with st.spinner("正在验证 API Key..."):
                    if verify_api_key(st.session_state.user_api_key):
                        st.session_state.api_key_verified = True
                        st.rerun()
                    else:
                        st.error("API Key 无效，请检查")

            with st.expander("如何获取 API Key？"):
                st.markdown("""
                1. 访问 [platform.deepseek.com](https://platform.deepseek.com)
                2. 注册 / 登录账号
                3. 进入 **API Keys** 页面
                4. 点击「创建 API Key」并复制
                5. 粘贴到上方输入框，点击「保存」

                > 新用户通常有免费额度
                """)

        st.markdown("---")

        # —— 记忆可视化面板 ——
        st.markdown("### 记忆档案")
        memories = get_all_memories(limit=20)

        col_count, col_clear = st.columns([2, 1])
        with col_count:
            st.metric("已存储记忆", len(memories))
        with col_clear:
            if memories:
                if st.button("清空全部", type="secondary", use_container_width=True):
                    if clear_all_memories():
                        st.session_state.memory_count = 0
                        st.rerun()

        if memories:
            with st.container(height=320):
                for mem in memories:
                    created_str = mem["created_at"][:16].replace("T", " ") if mem["created_at"] else "未知"
                    with st.expander(f"{mem['content'][:40]}...", expanded=False):
                        st.caption(f"存储时间：{created_str}")
                        st.caption(f"访问次数：{mem['access_count']}")
                        if st.button("删除此记忆", key=f"del_{mem['id']}", type="secondary"):
                            if delete_memory(mem["id"]):
                                st.session_state.memory_count = max(0, st.session_state.memory_count - 1)
                                st.rerun()
        else:
            st.caption("暂无记忆。对我说「记住：xxx」来添加记忆。")

        st.markdown("---")

        # —— 情绪曲线图表 ——
        st.markdown("### 情绪脉动")

        if st.session_state.emotion_history and HAS_PLOTLY:
            # 准备数据
            times = []
            scores = []
            for record in st.session_state.emotion_history[-30:]:  # 最近30条
                try:
                    t = datetime.fromisoformat(record["time"])
                    times.append(t)
                    if record["polarity"] == "正面":
                        scores.append(record["score"])
                    elif record["polarity"] == "负面":
                        scores.append(-record["score"])
                    else:
                        scores.append(0)
                except (ValueError, KeyError):
                    continue

            if times:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=times,
                    y=scores,
                    mode="lines+markers",
                    name="情绪值",
                    line=dict(color="#5b7fff", width=1.5),
                    marker=dict(
                        size=6,
                        color=[
                            "#4caf50" if s > 0 else "#f44336" if s < 0 else "#9e9e9e"
                            for s in scores
                        ],
                    ),
                    fill="tozeroy",
                    fillcolor="rgba(91,127,255,0.08)",
                ))
                fig.add_hline(y=0, line_dash="dot", line_color="#ccc", opacity=0.5)
                fig.update_layout(
                    height=180,
                    margin=dict(l=0, r=0, t=8, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                    yaxis=dict(
                        showgrid=False,
                        showticklabels=False,
                        zeroline=False,
                        range=[-1.2, 1.2],
                    ),
                    showlegend=False,
                    hovermode="x",
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.caption("最近情绪波动曲线（绿=正面 / 红=负面）")
        elif not HAS_PLOTLY:
            st.caption("安装 Plotly 可显示情绪曲线：`pip install plotly`")
        else:
            st.caption("开始对话后将自动记录情绪变化。")

        st.markdown("---")

        # —— 关于 ——
        with st.expander("关于禾苗"):
            st.markdown("""
            **禾苗 AI 伴侣** v1.0

            核心能力：
            - 长期记忆存储与检索
            - 情感感知与语气适配
            - 反馈学习与自我进化
            - 主动关心与模式检测
            - 记忆遗忘机制

            技术栈：
            Streamlit · LangChain · ChromaDB · DeepSeek
            """)

# ============================================================================
# 界面渲染 —— 主区域
# ============================================================================

def render_main():
    """渲染主对话区域"""

    # —— 标题 —— 使用原生组件
    st.title("禾 苗")
    st.caption("你的 AI 伴侣")

    # —— 首次使用提示 ——
    api_key = get_api_key()
    if not api_key:
        st.info(
            "请先配置 DeepSeek API Key。在项目根目录创建 `.env` 文件并添加 "
            "`DEEPSEEK_API_KEY=sk-xxxxx`，或在 Streamlit Cloud 的 Secrets 中设置。\n\n"
            "获取 Key：[platform.deepseek.com](https://platform.deepseek.com)"
        )
        return

    if not st.session_state.api_key_verified:
        with st.spinner("正在验证 API 连接..."):
            if verify_api_key(api_key):
                st.session_state.api_key_verified = True
                st.rerun()
            else:
                st.error("API Key 验证失败，请检查后刷新页面。")
                return

    # —— 首次欢迎语 ——
    if not st.session_state.messages:
        last_active = get_last_active_time()
        if last_active:
            hours_since = (datetime.now() - last_active).total_seconds() / 3600
            if hours_since > 24:
                days = int(hours_since / 24)
                welcome_msg = f"欢迎回来！已经 {days} 天没见了，最近过得怎么样？有什么想聊的或者需要我帮忙的吗？"
            else:
                welcome_msg = "你好！我是禾苗，你的AI伴侣。你可以和我聊天、让我记住关于你的事，我会越来越了解你。"
        else:
            welcome_msg = (
                "你好，我是**禾苗**，你的 AI 伴侣。\n\n"
                "我可以：\n"
                "- 记住关于你的一切（试试说「记住：我喜欢吃辣」）\n"
                "- 感知你的情绪并调整回应方式\n"
                "- 接收反馈并持续进化（试试说「反馈：回答请精简一些」）\n"
                "- 在需要时主动关心你\n\n"
                "让我们开始对话吧！"
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_msg,
        })

    # —— 渲染对话历史 ——
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # —— 输入区域 ——
    user_input = st.chat_input("输入消息...（支持指令：记住：xxx / 反馈：xxx）")

    if user_input:
        # 检查 API Key
        if not st.session_state.api_key_verified:
            st.warning("API 连接尚未就绪，请刷新页面重试。")
            return

        llm = get_llm(api_key)

        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # 生成 AI 回复
        with st.chat_message("assistant"):
            with st.spinner(""):
                reply = process_user_message(user_input, llm)

            st.markdown(reply)

        # 保存 AI 回复
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    init_session_state()
    render_sidebar()
    render_main()

    # 页脚
    st.divider()
    st.caption("HeMiao AI Companion · Powered by Streamlit + LangChain + ChromaDB + DeepSeek")
