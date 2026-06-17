"""
===============================================================================
SQLite 数据库封装 —— 用户表 + 对话表 + 消息表的统一管理
===============================================================================
"""

import os
import sqlite3
from contextlib import contextmanager

# 数据库路径
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "companion.db"
)


@contextmanager
def get_db(db_path: str = DEFAULT_DB_PATH):
    """
    获取数据库连接（上下文管理器，自动提交和关闭）。

    用法:
        with get_db() as db:
            db.execute("SELECT ...")
    """
    # 确保 data 目录存在
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_tables(db_path: str = DEFAULT_DB_PATH):
    """
    初始化所有表（首次运行或表不存在时自动创建）。

    表结构:
      - users: 用户账号
      - companion_sessions: 对话会话（关联用户）
      - companion_messages: 对话消息（关联会话）
    """
    with get_db(db_path) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS companion_sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                title TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS companion_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES companion_sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_user
                ON companion_sessions(user_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON companion_messages(session_id, created_at);
        """)


# ============================================================================
# 模块自测
# ============================================================================

if __name__ == "__main__":
    import tempfile
    from datetime import datetime

    print("=" * 60)
    print("测试 database 模块")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        # 测试建表
        print("\n[1] 初始化数据库表...")
        init_tables(db_path)
        with get_db(db_path) as db:
            tables = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = [t["name"] for t in tables]
            print(f"    表列表: {table_names}")
            assert "users" in table_names
            assert "companion_sessions" in table_names
            assert "companion_messages" in table_names
        print("    ✓ 建表通过")

        # 测试插入用户
        print("\n[2] 测试用户表操作...")
        with get_db(db_path) as db:
            db.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                ("testuser", "hashed_xxx", datetime.now().isoformat()),
            )
            user = db.execute("SELECT * FROM users WHERE username = ?", ("testuser",)).fetchone()
            assert user is not None
            assert user["username"] == "testuser"
            print(f"    用户: id={user['id']}, username={user['username']}")

            # 测试唯一约束
            try:
                db.execute(
                    "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                    ("testuser", "dup", datetime.now().isoformat()),
                )
                assert False, "应触发唯一约束错误"
            except sqlite3.IntegrityError:
                print("    ✓ 唯一约束生效")
        print("    ✓ 用户表操作通过")

        # 测试会话和消息
        print("\n[3] 测试会话与消息...")
        with get_db(db_path) as db:
            db.execute(
                "INSERT INTO companion_sessions (id, user_id, title, created_at) VALUES (?, ?, ?, ?)",
                ("sess_001", 1, "测试对话", datetime.now().isoformat()),
            )
            db.execute(
                "INSERT INTO companion_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                ("sess_001", "user", "你好", datetime.now().isoformat()),
            )
            db.execute(
                "INSERT INTO companion_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                ("sess_001", "assistant", "你好呀", datetime.now().isoformat()),
            )
            msgs = db.execute(
                "SELECT * FROM companion_messages WHERE session_id = ? ORDER BY created_at",
                ("sess_001",),
            ).fetchall()
            assert len(msgs) == 2
            print(f"    消息数: {len(msgs)}")
            for m in msgs:
                print(f"    - [{m['role']}] {m['content']}")
        print("    ✓ 会话与消息通过")

        # 测试级联删除
        print("\n[4] 测试级联删除...")
        with get_db(db_path) as db:
            db.execute("DELETE FROM companion_sessions WHERE id = ?", ("sess_001",))
            msgs = db.execute(
                "SELECT * FROM companion_messages WHERE session_id = ?", ("sess_001",)
            ).fetchall()
            assert len(msgs) == 0, "级联删除应清除消息"
        print("    ✓ 级联删除通过")

    print("\n" + "=" * 60)
    print("database 模块：全部测试通过")
    print("=" * 60)
