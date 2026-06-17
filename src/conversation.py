"""
===============================================================================
对话管理模块 —— 会话创建、消息存储、历史检索（按用户隔离）
===============================================================================
"""

import uuid
from datetime import datetime

from src.database import get_db, init_tables, DEFAULT_DB_PATH


class ConversationManager:
    """
    对话管理器 —— 所有操作都绑定 user_id，确保用户间数据隔离。

    用法:
        cm = ConversationManager()
        session_id = cm.create_session(user_id=1, title="闲聊")
        cm.add_message(session_id, user_id=1, role="user", content="你好")
        history = cm.get_history(session_id, user_id=1)
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        init_tables(db_path)

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------

    def create_session(self, user_id: int, title: str = "") -> str:
        """
        创建新对话。

        参数:
            user_id: 用户 ID
            title: 对话标题（可选，默认用第一条用户消息的前 20 字自动更新）

        返回:
            session_id: 8 位短 UUID
        """
        session_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        with get_db(self.db_path) as db:
            db.execute(
                "INSERT INTO companion_sessions (id, user_id, title, created_at) VALUES (?, ?, ?, ?)",
                (session_id, user_id, title, now),
            )
        return session_id

    # ------------------------------------------------------------------
    # 消息管理
    # ------------------------------------------------------------------

    def add_message(
        self, session_id: str, user_id: int, role: str, content: str
    ):
        """
        追加一条消息，并自动更新对话标题。

        标题规则：用第一条用户消息的前 20 字作为标题。

        参数:
            session_id: 对话 ID
            user_id: 用户 ID（用于权限校验）
            role: "user" 或 "assistant"
            content: 消息内容
        """
        now = datetime.now().isoformat()

        with get_db(self.db_path) as db:
            # 验证会话属于该用户
            session = db.execute(
                "SELECT id, title FROM companion_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()

            if session is None:
                raise ValueError(f"会话 {session_id} 不属于用户 {user_id}")

            # 插入消息
            db.execute(
                "INSERT INTO companion_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )

            # 自动更新标题：第一条用户消息的前 20 字
            if role == "user" and not session["title"]:
                title = content[:20] + ("..." if len(content) > 20 else "")
                db.execute(
                    "UPDATE companion_sessions SET title = ? WHERE id = ?",
                    (title, session_id),
                )

    # ------------------------------------------------------------------
    # 历史查询
    # ------------------------------------------------------------------

    def get_history(
        self, session_id: str, user_id: int, max_turns: int = 20
    ) -> list[dict]:
        """
        获取会话的最近 N 轮对话历史。

        参数:
            session_id: 对话 ID
            user_id: 用户 ID
            max_turns: 最多返回的轮数（一轮 = user + assistant 各一条）

        返回:
            [{"role": str, "content": str}, ...] 按时间正序排列
        """
        with get_db(self.db_path) as db:
            # 验证会话归属
            session = db.execute(
                "SELECT id FROM companion_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()

            if session is None:
                return []

            # 取最近 N 条消息（N = max_turns * 2，因为每轮两条）
            rows = db.execute(
                "SELECT role, content FROM companion_messages "
                "WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, max_turns * 2),
            ).fetchall()

        # 反转回时间正序
        rows = list(reversed(rows))
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    # ------------------------------------------------------------------
    # 会话列表与删除
    # ------------------------------------------------------------------

    def list_sessions(self, user_id: int) -> list[dict]:
        """
        列出用户的所有对话（按创建时间倒序）。

        参数:
            user_id: 用户 ID

        返回:
            [{"id": str, "title": str, "created_at": str, "message_count": int}, ...]
        """
        with get_db(self.db_path) as db:
            rows = db.execute(
                "SELECT s.id, s.title, s.created_at, "
                "(SELECT COUNT(*) FROM companion_messages m WHERE m.session_id = s.id) AS msg_count "
                "FROM companion_sessions s "
                "WHERE s.user_id = ? "
                "ORDER BY s.created_at DESC",
                (user_id,),
            ).fetchall()

        return [
            {
                "id": r["id"],
                "title": r["title"] or "新对话",
                "created_at": r["created_at"],
                "message_count": r["msg_count"],
            }
            for r in rows
        ]

    def delete_session(self, session_id: str, user_id: int) -> bool:
        """
        删除对话（验证属于该用户）。

        参数:
            session_id: 对话 ID
            user_id: 用户 ID

        返回:
            True 表示删除成功
        """
        with get_db(self.db_path) as db:
            cursor = db.execute(
                "DELETE FROM companion_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            )
            return cursor.rowcount > 0

    def get_message_count(self, session_id: str) -> int:
        """获取会话的消息条数"""
        with get_db(self.db_path) as db:
            row = db.execute(
                "SELECT COUNT(*) AS cnt FROM companion_messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row["cnt"] if row else 0


# ============================================================================
# 模块自测
# ============================================================================

if __name__ == "__main__":
    import tempfile, os

    print("=" * 60)
    print("测试 conversation 模块")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_conv.db")

        # 先创建测试用户（满足外键约束）
        from src.database import get_db as _getdb, init_tables as _init
        _init(db_path)
        with _getdb(db_path) as _db:
            _db.execute("INSERT INTO users (id, username, password_hash, created_at) VALUES (1, 'test_a', 'x', '2026-01-01')")
            _db.execute("INSERT INTO users (id, username, password_hash, created_at) VALUES (2, 'test_b', 'x', '2026-01-01')")

        cm = ConversationManager(db_path=db_path)
        user_id_a = 1
        user_id_b = 2

        # 测试创建会话
        print("\n[1] 创建会话...")
        sid_a = cm.create_session(user_id_a, "用户A的对话")
        sid_b = cm.create_session(user_id_b, "用户B的对话")
        print(f"    用户A会话: {sid_a}")
        print(f"    用户B会话: {sid_b}")
        assert sid_a != sid_b
        print("    OK: 会话创建通过")

        # 测试添加消息
        print("\n[2] 添加消息...")
        cm.add_message(sid_a, user_id_a, "user", "我喜欢吃辣")
        cm.add_message(sid_a, user_id_a, "assistant", "记住了，你喜欢吃辣")
        cm.add_message(sid_b, user_id_b, "user", "我不吃香菜")
        cm.add_message(sid_b, user_id_b, "assistant", "好的，记住了")
        print("    OK: 消息添加通过")

        # 测试跨用户隔离
        print("\n[3] 测试跨用户隔离...")
        # 用户B尝试访问用户A的会话
        try:
            cm.add_message(sid_a, user_id_b, "user", "越权")
            assert False, "应拒绝越权操作"
        except ValueError as e:
            print(f"    越权被拒绝: {e}")
        print("    OK: 跨用户隔离通过")

        # 测试历史查询
        print("\n[4] 测试历史查询...")
        hist_a = cm.get_history(sid_a, user_id_a)
        assert len(hist_a) == 2, f"应有2条消息，实际{len(hist_a)}"
        assert hist_a[0]["content"] == "我喜欢吃辣"
        assert hist_a[1]["content"] == "记住了，你喜欢吃辣"
        print(f"    用户A历史: {len(hist_a)} 条")

        # 测试用户B看不到用户A的历史
        hist_b_see_a = cm.get_history(sid_a, user_id_b)
        assert len(hist_b_see_a) == 0
        print(f"    用户B看A的对话: {len(hist_b_see_a)} 条 (应为0)")
        print("    OK: 历史隔离通过")

        # 测试对话列表
        print("\n[5] 测试对话列表...")
        sessions_a = cm.list_sessions(user_id_a)
        assert len(sessions_a) == 1
        print(f"    用户A对话列表: {len(sessions_a)} 个")
        for s in sessions_a:
            print(f"    - [{s['id']}] {s['title']} ({s['message_count']}条消息)")

        # 测试自动标题
        cm2 = ConversationManager(db_path=db_path)
        sid = cm2.create_session(user_id_a)
        cm2.add_message(sid, user_id_a, "user", "这是一个测试标题应该取前二十个字")
        sessions = cm2.list_sessions(user_id_a)
        for s in sessions:
            if s["id"] == sid:
                print(f"    自动标题: '{s['title']}'")
                assert "这是一个测试标题" in s["title"]
        print("    OK: 自动标题通过")

        # 测试删除（注意：前面自动标题测试已为 user_a 创建了第二个会话）
        print("\n[6] 测试删除...")
        ok = cm2.delete_session(sid, user_id_a)
        assert ok
        sessions_a = cm2.list_sessions(user_id_a)
        print(f"    删除后对话数: {len(sessions_a)} (应为0)")
        assert len(sessions_a) == 0
        print("    OK: 删除通过")

    print("\n" + "=" * 60)
    print("conversation 模块：全部测试通过")
    print("=" * 60)
