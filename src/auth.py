"""
===============================================================================
用户认证模块 —— bcrypt 哈希 + 注册 + 登录
===============================================================================
密码使用 bcrypt 哈希存储，绝不存明文。
用户名至少 3 字符，密码至少 4 字符。
===============================================================================
"""

import re
from datetime import datetime

import bcrypt

from src.database import get_db, init_tables, DEFAULT_DB_PATH


def hash_password(password: str) -> str:
    """
    对密码进行 bcrypt 哈希。

    参数:
        password: 明文密码

    返回:
        bcrypt 哈希字符串
    """
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password_bytes, salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """
    验证密码是否匹配。

    参数:
        password: 明文密码
        password_hash: bcrypt 哈希

    返回:
        True 表示密码正确
    """
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def _validate_credentials(username: str, password: str) -> str | None:
    """
    校验用户名和密码格式。

    返回:
        None 表示格式合法，否则返回错误消息字符串
    """
    if not username or len(username.strip()) < 3:
        return "用户名至少需要 3 个字符"
    if not password or len(password) < 4:
        return "密码至少需要 4 个字符"
    if not re.match(r"^[a-zA-Z0-9_一-鿿]+$", username.strip()):
        return "用户名只能包含字母、数字、下划线和中文"
    return None


def register_user(
    username: str,
    password: str,
    db_path: str = DEFAULT_DB_PATH,
) -> dict | None:
    """
    注册新用户。

    参数:
        username: 用户名（3-32字符）
        password: 密码（≥4字符）
        db_path: 数据库路径

    返回:
        成功返回 {"id": int, "username": str, "created_at": str}
        用户名已存在或格式不合法返回 None
    """
    # 格式校验
    error = _validate_credentials(username, password)
    if error:
        return None

    username = username.strip()

    # 确保表存在
    init_tables(db_path)

    try:
        with get_db(db_path) as db:
            password_hash = hash_password(password)
            now = datetime.now().isoformat()
            cursor = db.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, password_hash, now),
            )
            user_id = cursor.lastrowid
            return {"id": user_id, "username": username, "created_at": now}
    except Exception:
        # 用户名已存在或其他数据库错误
        return None


def login_user(
    username: str,
    password: str,
    db_path: str = DEFAULT_DB_PATH,
) -> dict | None:
    """
    用户登录验证。

    参数:
        username: 用户名
        password: 明文密码
        db_path: 数据库路径

    返回:
        成功返回 {"id": int, "username": str, "created_at": str}
        失败返回 None
    """
    username = username.strip()

    init_tables(db_path)

    with get_db(db_path) as db:
        user = db.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if user is None:
            return None

        if not verify_password(password, user["password_hash"]):
            return None

        return {
            "id": user["id"],
            "username": user["username"],
            "created_at": user["created_at"],
        }


def get_user_by_id(
    user_id: int,
    db_path: str = DEFAULT_DB_PATH,
) -> dict | None:
    """
    根据用户 ID 查询用户信息。

    参数:
        user_id: 用户 ID
        db_path: 数据库路径

    返回:
        用户信息字典或 None
    """
    init_tables(db_path)

    with get_db(db_path) as db:
        user = db.execute(
            "SELECT id, username, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()

        if user is None:
            return None

        return {
            "id": user["id"],
            "username": user["username"],
            "created_at": user["created_at"],
        }


# ============================================================================
# 模块自测
# ============================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("测试 auth 模块")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        import os
        db_path = os.path.join(tmpdir, "test_auth.db")

        # 测试密码哈希
        print("\n[1] 测试 bcrypt 哈希...")
        hashed = hash_password("test1234")
        assert verify_password("test1234", hashed), "正确密码应通过"
        assert not verify_password("wrong", hashed), "错误密码应拒绝"
        print("    ✓ 哈希/验证通过")

        # 测试注册
        print("\n[2] 测试注册...")
        user = register_user("张三", "mypassword", db_path=db_path)
        assert user is not None, "注册应成功"
        assert user["username"] == "张三"
        assert user["id"] == 1
        print(f"    注册成功: {user}")

        # 测试重复注册
        user2 = register_user("张三", "another", db_path=db_path)
        assert user2 is None, "重复用户名应返回 None"
        print("    ✓ 重复注册拒绝")

        # 测试格式校验
        assert register_user("ab", "1234", db_path=db_path) is None, "用户名太短"
        assert register_user("abc", "12", db_path=db_path) is None, "密码太短"
        print("    ✓ 格式校验通过")

        # 测试登录
        print("\n[3] 测试登录...")
        logged = login_user("张三", "mypassword", db_path=db_path)
        assert logged is not None, "正确密码应登录成功"
        assert logged["username"] == "张三"
        print(f"    登录成功: {logged}")

        # 测试错误密码
        bad = login_user("张三", "wrongpass", db_path=db_path)
        assert bad is None, "错误密码应拒绝"
        print("    ✓ 错误密码拒绝")

        # 测试不存在的用户
        no_user = login_user("不存在的人", "test", db_path=db_path)
        assert no_user is None
        print("    ✓ 不存在用户拒绝")

        # 测试 get_user_by_id
        print("\n[4] 测试 get_user_by_id...")
        u = get_user_by_id(1, db_path=db_path)
        assert u is not None and u["username"] == "张三"
        u2 = get_user_by_id(999, db_path=db_path)
        assert u2 is None
        print("    ✓ 查找用户通过")

    print("\n" + "=" * 60)
    print("auth 模块：全部测试通过")
    print("=" * 60)
