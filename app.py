"""
AI伴侣 v2.0 —— 模块化架构
UI 层(app.py) → 编排层(src/companion.py) → 功能模块(src/*.py)
"""

import os, streamlit as st
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from src.chroma_client import create_embedding_function, create_chroma_client, get_or_create_collections, create_llm
from src.memory import get_all_memories, delete_memory, clear_all_memories
from src.proactive import get_last_active_time, DEFAULT_LAST_ACTIVE_FILE
from src.companion import Companion

# ============================================================================
# 页面配置 & 样式
# ============================================================================
st.set_page_config(page_title="AI伴侣 · 懂你的智能伙伴", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>html,body,[class*='css']{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif}.stButton>button{border-radius:8px;font-weight:500;transition:all .2s ease}</style>", unsafe_allow_html=True)

CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# ============================================================================
# 缓存资源
# ============================================================================
@st.cache_resource
def _cached_embedding_function():
    return create_embedding_function()

@st.cache_resource
def _cached_chroma_client():
    return create_chroma_client(persist_path=CHROMA_DB_PATH)

@st.cache_resource
def _cached_companion():
    client = _cached_chroma_client()
    ef = _cached_embedding_function()
    mem_col, fb_col = get_or_create_collections(client, ef)
    return Companion(memory_collection=mem_col, feedback_collection=fb_col, ai_name=st.session_state.get("ai_name", "禾苗"))

# ============================================================================
# API Key
# ============================================================================
def get_api_key() -> str | None:
    """获取 API Key。优先级: 界面输入 > Secrets > .env"""
    key = st.session_state.get("user_api_key", "").strip()
    if key: return key
    try:
        key = st.secrets.get("DEEPSEEK_API_KEY", "")
        if key: return key
    except: pass
    return os.getenv("DEEPSEEK_API_KEY", "") or None


# ============================================================================
# 会话状态
# ============================================================================
def init_session_state():
    D = {"messages":[],"emotion_history":[],"api_key_verified":False,"memory_count":0,
         "user_api_key":"","last_proactive_time":{},"user_avatar":"👤","ai_avatar":"🌱","user_name":"我","ai_name":"禾苗"}
    for k,v in D.items():
        if k not in st.session_state: st.session_state[k] = v

# ============================================================================
# 记忆面板
# ============================================================================
def render_memory_panel():
    st.markdown("### 记忆档案")
    companion = _cached_companion()
    memories = get_all_memories(companion.memory_collection, limit=20)
    c1,c2 = st.columns([2,1])
    with c1: st.metric("已存储记忆", len(memories))
    with c2:
        if memories and st.button("清空全部", type="secondary", use_container_width=True):
            if clear_all_memories(companion.memory_collection, companion.feedback_collection):
                st.session_state.memory_count = 0; st.rerun()
    if memories:
        with st.container(height=320):
            for mem in memories:
                ts = mem["created_at"][:16].replace("T"," ") if mem["created_at"] else "未知"
                with st.expander(f"{mem['content'][:40]}...", expanded=False):
                    st.caption(f"存储时间：{ts}"); st.caption(f"访问次数：{mem['access_count']}")
                    if st.button("删除此记忆", key=f"del_{mem['id']}", type="secondary"):
                        if delete_memory(companion.memory_collection, mem["id"]):
                            st.session_state.memory_count = max(0, st.session_state.memory_count - 1); st.rerun()
    else:
        st.caption("暂无记忆。对我说「记住：xxx」来添加记忆。")


# ============================================================================
# 侧边栏渲染
# ============================================================================

def render_sidebar():
    """渲染侧边栏：品牌、API配置、个性化设置、记忆面板、情绪图表"""
    with st.sidebar:
        # —— 品牌标识 ——
        st.title("禾 苗")
        st.caption("AI Companion")

        st.markdown("---")

        # —— API Key 配置 ——
        st.markdown("### API 配置")

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
                st.session_state.api_key_verified = False
                st.rerun()

            if clear_clicked:
                st.session_state.user_api_key = ""
                st.session_state.api_key_verified = False
                st.rerun()

            # 验证 Key
            if st.session_state.get("user_api_key", "") and not st.session_state.api_key_verified:
                with st.spinner("正在验证 API Key..."):
                    if Companion.verify_api_key(
                        st.session_state.user_api_key, create_llm
                    ):
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

        # —— 个性化设置 ——
        with st.expander("个性化设置"):
            AVATAR_PRESETS = {
                "用户": ["👤", "🧑", "👩", "👨", "😊", "🐱", "🐶", "🦊", "🐼", "⭐"],
                "AI":   ["🌱", "🤖", "✨", "🌟", "💡", "🌸", "🍀", "🎵", "💎", "🔥"],
            }

            st.caption("用户头像")
            cols = st.columns(5)
            for i, emoji in enumerate(AVATAR_PRESETS["用户"]):
                with cols[i % 5]:
                    if st.button(emoji, key=f"ua_{emoji}", use_container_width=True):
                        st.session_state.user_avatar = emoji
                        st.rerun()

            st.caption("AI 头像")
            cols = st.columns(5)
            for i, emoji in enumerate(AVATAR_PRESETS["AI"]):
                with cols[i % 5]:
                    if st.button(emoji, key=f"aa_{emoji}", use_container_width=True):
                        st.session_state.ai_avatar = emoji
                        st.rerun()

            col1, col2 = st.columns(2)
            with col1:
                new_name = st.text_input("你的昵称", value=st.session_state.user_name, max_chars=10)
                if new_name and new_name != st.session_state.user_name:
                    st.session_state.user_name = new_name
                    st.rerun()
            with col2:
                new_ai = st.text_input("AI 昵称", value=st.session_state.ai_name, max_chars=10)
                if new_ai and new_ai != st.session_state.ai_name:
                    st.session_state.ai_name = new_ai
                    st.rerun()

            st.caption(
                f"预览：{st.session_state.user_avatar} {st.session_state.user_name}"
                f"  ⇄  {st.session_state.ai_avatar} {st.session_state.ai_name}"
            )

        st.markdown("---")

        # —— 记忆可视化面板 ——
        render_memory_panel()

        st.markdown("---")

        # —— 情绪曲线 ——
        st.markdown("### 情绪脉动")

        if st.session_state.emotion_history and HAS_PLOTLY:
            times, scores = [], []
            for record in st.session_state.emotion_history[-30:]:
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
                    x=times, y=scores, mode="lines+markers", name="情绪值",
                    line=dict(color="#5b7fff", width=1.5),
                    marker=dict(size=6, color=[
                        "#4caf50" if s > 0 else "#f44336" if s < 0 else "#9e9e9e"
                        for s in scores
                    ]),
                    fill="tozeroy", fillcolor="rgba(91,127,255,0.08)",
                ))
                fig.add_hline(y=0, line_dash="dot", line_color="#ccc", opacity=0.5)
                fig.update_layout(
                    height=180, margin=dict(l=0, r=0, t=8, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-1.2, 1.2]),
                    showlegend=False, hovermode="x",
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
            **禾苗 AI 伴侣** v2.0

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
# 主区域渲染
# ============================================================================

def render_main():
    """渲染主对话区域"""
    st.title("禾 苗")
    st.caption("你的 AI 伴侣")

    api_key = get_api_key()
    if not api_key:
        st.info(
            "请先配置 DeepSeek API Key。在侧边栏输入 Key，"
            "或在项目根目录创建 `.env` 文件并添加 `DEEPSEEK_API_KEY=sk-xxxxx`。\n\n"
            "获取 Key：[platform.deepseek.com](https://platform.deepseek.com)"
        )
        return

    if not st.session_state.api_key_verified:
        with st.spinner("正在验证 API 连接..."):
            if Companion.verify_api_key(api_key, create_llm):
                st.session_state.api_key_verified = True
                st.rerun()
            else:
                st.error("API Key 验证失败，请检查后刷新页面。")
                return

    # —— 首次欢迎语 ——
    if not st.session_state.messages:
        last_active = get_last_active_time(DEFAULT_LAST_ACTIVE_FILE)
        if last_active:
            hours_since = (datetime.now() - last_active).total_seconds() / 3600
            if hours_since > 24:
                days = int(hours_since / 24)
                welcome_msg = f"欢迎回来！已经 {days} 天没见了，最近过得怎么样？"
            else:
                welcome_msg = "你好！我是禾苗，你的AI伴侣。我会越来越了解你。"
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
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

    # —— 渲染对话历史 ——
    for msg in st.session_state.messages:
        avatar = st.session_state.user_avatar if msg["role"] == "user" else st.session_state.ai_avatar
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # —— 输入区域 ——
    user_input = st.chat_input("输入消息...（支持指令：记住：xxx / 反馈：xxx）")

    if user_input:
        if not st.session_state.api_key_verified:
            st.warning("API 连接尚未就绪，请刷新页面重试。")
            return

        llm = create_llm(api_key)
        companion = _cached_companion()

        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user", avatar=st.session_state.user_avatar):
            st.markdown(user_input)

        # 生成 AI 回复
        with st.chat_message("assistant", avatar=st.session_state.ai_avatar):
            with st.spinner(""):
                reply = companion.process_message(
                    user_input=user_input,
                    conversation_history=st.session_state.messages,
                    session_state=st.session_state,
                    llm=llm,
                )
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    init_session_state()
    render_sidebar()
    render_main()

    st.divider()
    st.caption("HeMiao AI Companion · Powered by Streamlit + LangChain + ChromaDB + DeepSeek")
