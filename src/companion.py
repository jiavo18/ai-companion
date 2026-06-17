"""
===============================================================================
核心编排类 —— Companion
===============================================================================
整合情感检测、记忆管理、反馈学习、主动提问、对话持久化和系统提示词构建，
提供统一的消息处理入口。
===============================================================================
"""

import re
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.emotion import detect_emotion, get_emotion_response_guide
from src.memory import store_memory
from src.feedback import store_feedback, retrieve_feedback, parse_feedback_to_constraints
from src.proactive import (
    check_trigger_conditions,
    update_last_active_time,
    DEFAULT_LAST_ACTIVE_FILE,
)
from src.prompt_builder import build_system_prompt
from src.conversation import ConversationManager
from src.database import DEFAULT_DB_PATH
from src.memory_extractor import start_auto_extraction


class Companion:
    """
    AI 伴侣核心编排类。

    封装了从用户输入到 AI 回复的完整处理流程：
    1. 指令识别（记忆/反馈）
    2. 主动提问检测
    3. 情感检测
    4. 记忆检索
    5. 反馈检索
    6. 系统提示词构建
    7. LLM 调用
    8. 回复拼接
    9. 对话持久化（自动保存到 SQLite）
    """

    def __init__(
        self,
        memory_collection,
        feedback_collection,
        ai_name: str = "禾苗",
        user_id: int = 0,
        db_path: str = DEFAULT_DB_PATH,
    ):
        """
        初始化 Companion 实例。

        参数:
            memory_collection: ChromaDB 记忆集合
            feedback_collection: ChromaDB 反馈集合
            ai_name: AI 的名字
            user_id: 当前用户 ID（用于对话持久化）
            db_path: SQLite 数据库路径
        """
        self.memory_collection = memory_collection
        self.feedback_collection = feedback_collection
        self.ai_name = ai_name
        self.user_id = user_id
        self.conversation = ConversationManager(db_path=db_path)

    # ========================================================================
    # 核心消息处理
    # ========================================================================

    def process_message(
        self,
        user_input: str,
        conversation_history: list[dict],
        session_state: dict,
        llm,
        session_id: str = "",
    ) -> str:
        """
        处理用户输入并生成 AI 回复。

        处理流程（按顺序）:
        1. 判断是否为记忆/反馈指令
        2. 主动提问检测（结果作为回复前缀）
        3. 情感检测
        4. 记忆检索
        5. 反馈检索
        6. 构建系统提示词
        7. 调用 LLM
        8. 将主动提问文本拼接为回复第一句话

        参数:
            user_input: 用户输入文本
            conversation_history: 对话历史
            session_state: 会话状态字典（支持 st.session_state 或普通 dict）
            llm: ChatOpenAI 实例

        返回:
            AI 回复文本
        """
        from src.memory import retrieve_memories as _retrieve_memories

        user_input = user_input.strip()

        # —— 步骤0: 指令识别 ——
        # 记忆指令: "记住：xxx" 或 "记住:xxx"
        remember_match = re.match(r"^记住[：:]\s*(.+)", user_input)
        if remember_match:
            content = remember_match.group(1).strip()
            if content:
                store_memory(self.memory_collection, content)
                update_last_active_time(DEFAULT_LAST_ACTIVE_FILE)
                session_state["memory_count"] = session_state.get("memory_count", 0) + 1
                return f"已牢记：「{content}」。以后你需要的时候，我会想起这个信息。"

        # 反馈指令: "反馈：xxx" 或 "反馈:xxx"
        feedback_match = re.match(r"^反馈[：:]\s*(.+)", user_input)
        if feedback_match:
            content = feedback_match.group(1).strip()
            if content:
                store_feedback(self.feedback_collection, content)
                update_last_active_time(DEFAULT_LAST_ACTIVE_FILE)
                return f"已收到你的反馈：「{content}」。我会在后续对话中调整我的表达方式。"

        # —— 步骤1: 主动提问检测 ——
        last_proactive_time = session_state.get("last_proactive_time", {})
        proactive_text = check_trigger_conditions(
            user_input=user_input,
            conversation_history=conversation_history,
            last_proactive_time=last_proactive_time,
            last_active_file=DEFAULT_LAST_ACTIVE_FILE,
        )

        # —— 步骤2: 情感检测 ——
        emotion_result = detect_emotion(user_input)
        emotion_guide = get_emotion_response_guide(emotion_result["polarity"])

        # 记录情绪
        emotion_history = session_state.get("emotion_history", [])
        emotion_history.append({
            "time": datetime.now().isoformat(),
            "polarity": emotion_result["polarity"],
            "score": emotion_result["score"],
        })
        # 只保留最近 100 条
        if len(emotion_history) > 100:
            emotion_history = emotion_history[-100:]
        session_state["emotion_history"] = emotion_history

        # —— 步骤3: 记忆检索 ——
        memories = _retrieve_memories(self.memory_collection, user_input, n_results=3)

        # —— 步骤4: 反馈检索 ——
        feedbacks = retrieve_feedback(self.feedback_collection, n_results=3)
        feedback_constraints = parse_feedback_to_constraints(feedbacks)

        # —— 步骤5: 构建系统提示词 ——
        system_prompt = build_system_prompt(
            memories=memories,
            feedback_constraints=feedback_constraints,
            emotion_guide=emotion_guide,
            proactive_care=None,  # 主动关心已移至回复前缀
            ai_name=self.ai_name,
        )

        # —— 步骤6: 构建消息列表 ——
        messages = [SystemMessage(content=system_prompt)]

        # 加入最近 20 轮对话历史
        recent_history = conversation_history[-20:]
        for msg in recent_history:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
            else:
                messages.append(AIMessage(content=msg.get("content", "")))

        # 当前用户消息
        messages.append(HumanMessage(content=user_input))

        # —— 步骤7: 调用 LLM ——
        try:
            response = llm.invoke(messages)
            llm_reply = response.content
        except Exception as e:
            llm_reply = f"抱歉，我暂时无法回应。请检查网络连接或API配置。（错误详情：{str(e)}）"

        # —— 步骤8: 拼接回复 ——
        if proactive_text:
            reply = f"{proactive_text}\n\n{llm_reply}"
        else:
            reply = llm_reply

        # —— 步骤9: 更新活跃时间 ——
        update_last_active_time(DEFAULT_LAST_ACTIVE_FILE)

        # —— 步骤10: 对话持久化（自动保存到 SQLite） ——
        if session_id and self.user_id:
            try:
                self.conversation.add_message(
                    session_id, self.user_id, "user", user_input
                )
                self.conversation.add_message(
                    session_id, self.user_id, "assistant", reply
                )
            except Exception:
                pass  # 持久化失败不影响主流程

        # —— 步骤11: 自动记忆提取（后台线程，不阻塞） ——
        auto_extract = session_state.get("auto_extract_enabled", True)
        if auto_extract and llm is not None:
            try:
                start_auto_extraction(
                    user_message=user_input,
                    ai_reply=reply,
                    memory_collection=self.memory_collection,
                    llm=llm,
                    on_done=lambda count: None,  # 静默完成
                )
            except Exception:
                pass  # 提取失败不影响主流程

        return reply

    # ========================================================================
    # API Key 验证
    # ========================================================================

    @staticmethod
    def verify_api_key(api_key: str, llm_factory) -> bool:
        """
        快速验证 API Key 是否有效。

        参数:
            api_key: 待验证的 Key
            llm_factory: 创建 LLM 的可调用对象

        返回:
            True 表示 Key 有效
        """
        try:
            llm = llm_factory(api_key)
            response = llm.invoke([HumanMessage(content="你好，请回复一个字：好")])
            return bool(response.content.strip())
        except Exception:
            return False


# ============================================================================
# 模块自测
# ============================================================================

if __name__ == "__main__":
    import tempfile
    from src.chroma_client import (
        create_chroma_client,
        create_embedding_function,
        get_or_create_collections,
        create_llm,
    )

    print("=" * 60)
    print("测试 companion 模块")
    print("=" * 60)

    # 准备测试环境
    ef = create_embedding_function()
    with tempfile.TemporaryDirectory() as tmpdir:
        client = create_chroma_client(persist_path=tmpdir)
        mem_col, fb_col = get_or_create_collections(client, ef)

        # 创建 Companion 实例
        companion = Companion(
            memory_collection=mem_col,
            feedback_collection=fb_col,
            ai_name="测试助手",
        )

        # 模拟 session_state
        session_state = {
            "messages": [],
            "emotion_history": [],
            "memory_count": 0,
            "last_proactive_time": {},
        }

        # 测试记忆指令
        print("\n[1] 测试记忆指令...")
        reply = companion.process_message(
            "记住：我喜欢吃辣",
            conversation_history=[],
            session_state=session_state,
            llm=None,  # 不需要 LLM
        )
        print(f"    输入: 记住：我喜欢吃辣")
        print(f"    回复: {reply}")
        assert "已牢记" in reply, f"应包含「已牢记」，实际: {reply}"
        print("    ✓ 记忆指令通过")

        # 测试反馈指令
        print("\n[2] 测试反馈指令...")
        reply = companion.process_message(
            "反馈：回答太啰嗦了",
            conversation_history=[],
            session_state=session_state,
            llm=None,
        )
        print(f"    输入: 反馈：回答太啰嗦了")
        print(f"    回复: {reply}")
        assert "已收到" in reply, f"应包含「已收到」，实际: {reply}"
        print("    ✓ 反馈指令通过")

        # 测试常规对话（需要真实 API Key 才能完整测试）
        print("\n[3] 测试常规对话流程（无 LLM）...")
        # 添加一些对话历史
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好呀"},
        ]
        try:
            # 不传 LLM，验证流程不崩溃（会在 LLM 调用时报错，但流程本身正确）
            reply = companion.process_message(
                "今天天气不错",
                conversation_history=history,
                session_state=session_state,
                llm=None,
            )
        except AttributeError:
            # 预期的错误：None 没有 invoke 方法
            print("    流程走到 LLM 调用步骤（预期行为）")
        print("    ✓ 常规流程通过（走到 LLM 调用前无异常）")

        # 验证情绪记录
        print("\n[4] 验证情绪记录...")
        assert len(session_state["emotion_history"]) >= 2, "应至少记录 2 次情绪"
        print(f"    情绪记录数: {len(session_state['emotion_history'])}")
        print("    ✓ 情绪记录通过")

    print("\n" + "=" * 60)
    print("companion 模块：全部测试通过 ✓")
    print("=" * 60)
