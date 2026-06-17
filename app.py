"""
AI伴侣 v2.0 —— 用户系统 + 对话持久化 + 模块化架构
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
from src.auth import register_user, login_user
from src.conversation import ConversationManager
from src.database import init_tables

st.set_page_config(page_title="禾苗 AI伴侣", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
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

def _get_companion_for_user(user_id: int):
    """为非 Streamlit 缓存的用户创建 Companion（登录后调用）"""
    client = _cached_chroma_client()
    ef = _cached_embedding_function()
    mem_col, fb_col = get_or_create_collections(client, ef, user_id=user_id)
    ai_name = st.session_state.get("ai_name", "禾苗")
    return Companion(memory_collection=mem_col, feedback_collection=fb_col, ai_name=ai_name, user_id=user_id)

def get_api_key() -> str | None:
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
    D = {
        "logged_in": False, "current_user": None, "current_session_id": "",
        "messages": [], "emotion_history": [], "api_key_verified": False,
        "auto_extract_enabled": True,  # 自动记忆提取开关
        "emotion_mode": "keyword",    # 情感检测模式: keyword / llm
        "memory_count": 0, "user_api_key": "", "last_proactive_time": {},
        "user_avatar": "👤", "ai_avatar": "🌱", "user_name": "我", "ai_name": "禾苗",
        "companion": None,
    }
    for k, v in D.items():
        if k not in st.session_state: st.session_state[k] = v

# ============================================================================
# 登录/注册页面
# ============================================================================
def render_login_page():
    st.title("禾 苗")
    st.caption("你的 AI 伴侣")

    tab_login, tab_register = st.tabs(["登录", "注册"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("用户名", placeholder="输入用户名", key="login_user")
            password = st.text_input("密码", type="password", placeholder="输入密码", key="login_pass")
            submitted = st.form_submit_button("登录", type="primary", use_container_width=True)
            if submitted and username and password:
                user = login_user(username, password)
                if user:
                    st.session_state.current_user = user
                    st.session_state.logged_in = True
                    st.session_state.current_session_id = ""
                    st.session_state.messages = []
                    st.session_state.companion = _get_companion_for_user(user["id"])
                    st.rerun()
                else:
                    st.error("用户名或密码错误")

    with tab_register:
        with st.form("register_form"):
            new_user = st.text_input("用户名", placeholder="至少3个字符", key="reg_user")
            new_pass = st.text_input("密码", type="password", placeholder="至少4个字符", key="reg_pass")
            new_pass2 = st.text_input("确认密码", type="password", placeholder="再输一次", key="reg_pass2")
            submitted = st.form_submit_button("注册", type="primary", use_container_width=True)
            if submitted:
                if not new_user or len(new_user.strip()) < 3:
                    st.error("用户名至少需要 3 个字符")
                elif not new_pass or len(new_pass) < 4:
                    st.error("密码至少需要 4 个字符")
                elif new_pass != new_pass2:
                    st.error("两次密码不一致")
                else:
                    user = register_user(new_user.strip(), new_pass)
                    if user:
                        st.success(f"注册成功！请切换到「登录」标签登录。")
                    else:
                        st.error("用户名已存在或格式不合法")

# ============================================================================
# 记忆面板
# ============================================================================
def render_memory_panel(companion):
    st.markdown("### 记忆档案")
    memories = get_all_memories(companion.memory_collection, limit=20)
    c1, c2 = st.columns([2, 1])
    with c1: st.metric("已存储记忆", len(memories))
    with c2:
        if memories and st.button("清空全部", type="secondary", use_container_width=True):
            if clear_all_memories(companion.memory_collection, companion.feedback_collection):
                st.session_state.memory_count = 0; st.rerun()
    if memories:
        with st.container(height=320):
            for mem in memories:
                ts = mem["created_at"][:16].replace("T", " ") if mem["created_at"] else "未知"
                with st.expander(f"{mem['content'][:40]}...", expanded=False):
                    st.caption(f"存储时间：{ts}"); st.caption(f"访问次数：{mem['access_count']}")
                    if st.button("删除此记忆", key=f"del_{mem['id']}", type="secondary"):
                        if delete_memory(companion.memory_collection, mem["id"]):
                            st.session_state.memory_count = max(0, st.session_state.memory_count - 1); st.rerun()
    else:
        st.caption("暂无记忆。对我说「记住：xxx」来添加记忆。")

# ============================================================================
# 侧边栏
# ============================================================================
def render_sidebar():
    with st.sidebar:
        if st.session_state.logged_in and st.session_state.current_user:
            user = st.session_state.current_user
            # 用户信息
            st.markdown(f"### 👤 {user['username']}")
            st.caption(f"用户 ID: {user['id']}")

            if st.button("退出登录", type="secondary", use_container_width=True):
                # 清空所有会话状态
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

            st.markdown("---")

            # API Key 配置
            st.markdown("### API 配置")
            has_secret = False
            try: has_secret = bool(st.secrets.get("DEEPSEEK_API_KEY", ""))
            except: pass

            if has_secret:
                st.success("已使用云端 API Key")
            elif st.session_state.api_key_verified:
                st.success("API 已连接")
            else:
                with st.form("api_form", clear_on_submit=False):
                    uk = st.text_input("DeepSeek API Key", type="password", value=st.session_state.get("user_api_key", ""), placeholder="sk-xxx", key="api_key_input")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.form_submit_button("保存", type="primary", use_container_width=True) and uk.strip():
                            st.session_state.user_api_key = uk.strip(); st.session_state.api_key_verified = False; st.rerun()
                    with c2:
                        if st.form_submit_button("清除", use_container_width=True):
                            st.session_state.user_api_key = ""; st.session_state.api_key_verified = False; st.rerun()

                if st.session_state.get("user_api_key","") and not st.session_state.api_key_verified:
                    with st.spinner("验证 API Key..."):
                        if Companion.verify_api_key(st.session_state.user_api_key, create_llm):
                            st.session_state.api_key_verified = True; st.rerun()
                        else:
                            st.error("API Key 无效")

            st.markdown("---")

            # 对话列表
            st.markdown("### 对话")
            companion = st.session_state.get("companion")
            if companion:
                sessions = companion.conversation.list_sessions(user["id"])
                if st.button("+ 新建对话", use_container_width=True):
                    st.session_state.current_session_id = ""
                    st.session_state.messages = []
                    st.rerun()

                if sessions:
                    for sess in sessions:
                        is_active = st.session_state.current_session_id == sess["id"]
                        label = f"{'● ' if is_active else ''}{sess['title']}"
                        col_s, col_d = st.columns([4, 1])
                        with col_s:
                            if st.button(label, key=f"sess_{sess['id']}", use_container_width=True, type="primary" if is_active else "secondary"):
                                # 切换对话：从数据库加载历史
                                st.session_state.current_session_id = sess["id"]
                                history = companion.conversation.get_history(sess["id"], user["id"])
                                st.session_state.messages = history
                                st.rerun()
                        with col_d:
                            if st.button("✕", key=f"del_{sess['id']}"):
                                companion.conversation.delete_session(sess["id"], user["id"])
                                if st.session_state.current_session_id == sess["id"]:
                                    st.session_state.current_session_id = ""
                                    st.session_state.messages = []
                                st.rerun()
                else:
                    st.caption("暂无对话记录")

            st.markdown("---")

            # 个性化设置
            with st.expander("个性化设置"):
                AVATARS_USER = ["👤","🧑","👩","👨","😊","🐱","🐶","🦊","🐼","⭐"]
                AVATARS_AI   = ["🌱","🤖","✨","🌟","💡","🌸","🍀","🎵","💎","🔥"]
                st.caption("用户头像")
                cols = st.columns(5)
                for i, e in enumerate(AVATARS_USER):
                    with cols[i%5]:
                        if st.button(e, key=f"ua_{e}", use_container_width=True):
                            st.session_state.user_avatar = e; st.rerun()
                st.caption("AI 头像")
                cols = st.columns(5)
                for i, e in enumerate(AVATARS_AI):
                    with cols[i%5]:
                        if st.button(e, key=f"aa_{e}", use_container_width=True):
                            st.session_state.ai_avatar = e; st.rerun()
                c1, c2 = st.columns(2)
                with c1:
                    n = st.text_input("你的昵称", value=st.session_state.user_name, max_chars=10, key="uname")
                    if n and n != st.session_state.user_name: st.session_state.user_name = n; st.rerun()
                with c2:
                    n = st.text_input("AI 昵称", value=st.session_state.ai_name, max_chars=10, key="ainame")
                    if n and n != st.session_state.ai_name: st.session_state.ai_name = n; st.rerun()
                st.caption(f"预览：{st.session_state.user_avatar} {st.session_state.user_name}  ⇄  {st.session_state.ai_avatar} {st.session_state.ai_name}")

            # 自动记忆提取开关
            st.checkbox("自动记忆提取", value=st.session_state.auto_extract_enabled,
                        key="auto_extract_toggle",
                        help="开启后AI会自动从对话中识别并记住你的信息。关闭后仅手动「记住：xxx」指令生效。",
                        on_change=lambda: setattr(st.session_state, "auto_extract_enabled", st.session_state.auto_extract_toggle))

            # 情感检测模式
            st.radio("情感检测模式", options=["keyword", "llm"],
                     index=0 if st.session_state.emotion_mode == "keyword" else 1,
                     format_func=lambda x: "关键词匹配（快）" if x == "keyword" else "LLM 分析（准）",
                     key="emotion_mode_radio",
                     help="关键词：零延迟零成本。LLM：更准确，能理解「烦死了今天怎么这么开心」是正面情绪。",
                     on_change=lambda: setattr(st.session_state, "emotion_mode", st.session_state.emotion_mode_radio))

            st.markdown("---")

            # 记忆面板
            if companion:
                render_memory_panel(companion)

            st.markdown("---")

            # 情绪曲线
            st.markdown("### 情绪脉动")
            if st.session_state.emotion_history and HAS_PLOTLY:
                times, scores = [], []
                for r in st.session_state.emotion_history[-30:]:
                    try:
                        t = datetime.fromisoformat(r["time"]); times.append(t)
                        if r["polarity"] == "正面": scores.append(r["score"])
                        elif r["polarity"] == "负面": scores.append(-r["score"])
                        else: scores.append(0)
                    except: continue
                if times:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=times, y=scores, mode="lines+markers", line=dict(color="#5b7fff", width=1.5),
                        marker=dict(size=6, color=["#4caf50" if s>0 else "#f44336" if s<0 else "#9e9e9e" for s in scores]),
                        fill="tozeroy", fillcolor="rgba(91,127,255,0.08)"))
                    fig.add_hline(y=0, line_dash="dot", line_color="#ccc", opacity=0.5)
                    fig.update_layout(height=180, margin=dict(l=0,r=0,t=8,b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-1.2,1.2]),
                        showlegend=False, hovermode="x")
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    st.caption("最近情绪波动（绿正 / 红负）")
            elif not HAS_PLOTLY:
                st.caption("安装 Plotly 显示情绪曲线")
            else:
                st.caption("开始对话后记录情绪变化。")

# ============================================================================
# 主区域
# ============================================================================
def render_main():
    if not st.session_state.logged_in:
        render_login_page()
        return

    user = st.session_state.current_user
    companion = st.session_state.get("companion")
    if companion is None:
        companion = _get_companion_for_user(user["id"])
        st.session_state.companion = companion

    st.title("禾 苗")
    st.caption(f"你好，{user['username']} — 你的 AI 伴侣")

    api_key = get_api_key()
    if not api_key:
        st.info("请在侧边栏配置 DeepSeek API Key。获取：[platform.deepseek.com](https://platform.deepseek.com)")
        return

    if not st.session_state.api_key_verified:
        with st.spinner("验证 API 连接..."):
            if Companion.verify_api_key(api_key, create_llm):
                st.session_state.api_key_verified = True; st.rerun()
            else:
                st.error("API Key 验证失败。"); return

    # 加载对话历史（从 SQLite）
    if not st.session_state.messages and st.session_state.current_session_id:
        history = companion.conversation.get_history(st.session_state.current_session_id, user["id"])
        st.session_state.messages = history

    # 首次欢迎语
    if not st.session_state.messages:
        last_active = get_last_active_time(DEFAULT_LAST_ACTIVE_FILE)
        if last_active:
            hours = (datetime.now() - last_active).total_seconds() / 3600
            if hours > 24:
                welcome = f"欢迎回来！已经 {int(hours/24)} 天没见了，最近过得怎么样？"
            else:
                welcome = "你好！我是禾苗，你的AI伴侣。"
        else:
            welcome = "你好，我是**禾苗**，你的 AI 伴侣。\n\n我可以记住关于你的一切、感知你的情绪、从你的反馈中进化。\n试试说「记住：我喜欢吃辣」或「反馈：请精简回答」。"
        st.session_state.messages.append({"role": "assistant", "content": welcome})

    # 渲染对话
    for msg in st.session_state.messages:
        avatar = st.session_state.user_avatar if msg["role"] == "user" else st.session_state.ai_avatar
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # 输入
    prompt = st.chat_input("输入消息...（支持指令：记住：xxx / 反馈：xxx）")
    if prompt:
        if not st.session_state.api_key_verified:
            st.warning("API 尚未就绪。"); return

        # 确保当前有 session_id
        if not st.session_state.current_session_id:
            st.session_state.current_session_id = companion.conversation.create_session(user["id"])
            st.session_state.messages = []

        llm = create_llm(api_key)

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=st.session_state.user_avatar):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=st.session_state.ai_avatar):
            with st.spinner(""):
                reply = companion.process_message(
                    user_input=prompt,
                    conversation_history=st.session_state.messages,
                    session_state=st.session_state,
                    llm=llm,
                    session_id=st.session_state.current_session_id,
                )
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    init_tables()  # 确保数据库表存在
    init_session_state()
    render_sidebar()
    render_main()
    st.divider()
    st.caption("HeMiao AI Companion · Streamlit + LangChain + ChromaDB + DeepSeek")
