"""
===============================================================================
长期记忆管理模块 —— 向量存储、语义检索与遗忘机制
===============================================================================
基于 ChromaDB 实现用户记忆的增删查改，支持基于时间的遗忘权重衰减。
===============================================================================
"""

import uuid
from datetime import datetime

# ============================================================================
# 记忆遗忘阈值（天）
# ============================================================================

MEMORY_FORGET_DAYS = 30
MEMORY_WEAKEN_DAYS = 14


# ============================================================================
# 记忆存储
# ============================================================================

def store_memory(memory_collection, content: str) -> str:
    """
    将一条记忆存入 ChromaDB。

    参数:
        memory_collection: ChromaDB 记忆集合实例
        content: 记忆文本内容

    返回:
        memory_id: 记忆的唯一标识
    """
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


# ============================================================================
# 记忆检索（含遗忘权重）
# ============================================================================

def retrieve_memories(memory_collection, query: str, n_results: int = 3) -> list[dict]:
    """
    检索与当前查询最相关的记忆，并应用遗忘权重。

    检索流程:
    1. 语义检索 Top N*3 候选记忆
    2. 对每条记忆应用遗忘权重（超过阈值则增加距离惩罚）
    3. 按调整后距离排序，返回 Top N
    4. 更新被选中记忆的访问记录

    参数:
        memory_collection: ChromaDB 记忆集合实例
        query: 查询文本
        n_results: 返回的记忆数量

    返回:
        [{"id": str, "content": str, "created_at": str,
          "last_accessed": str, "access_count": int, "distance": float}, ...]
    """
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
            last_accessed = datetime.fromisoformat(
                meta.get("last_accessed", meta.get("created_at", now.isoformat()))
            )
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


# ============================================================================
# 记忆列表与删除
# ============================================================================

def get_all_memories(memory_collection, limit: int = 20) -> list[dict]:
    """
    获取所有已存储的记忆（用于可视化面板）。

    参数:
        memory_collection: ChromaDB 记忆集合实例
        limit: 最大返回条数

    返回:
        [{"id": str, "content": str, "created_at": str, "access_count": int}, ...]
    """
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


def delete_memory(memory_collection, memory_id: str) -> bool:
    """
    删除单条记忆。

    参数:
        memory_collection: ChromaDB 记忆集合实例
        memory_id: 记忆唯一标识

    返回:
        True 表示删除成功
    """
    try:
        memory_collection.delete(ids=[memory_id])
        return True
    except Exception:
        return False


def clear_all_memories(memory_collection, feedback_collection) -> bool:
    """
    清空所有记忆和反馈。

    参数:
        memory_collection: ChromaDB 记忆集合实例
        feedback_collection: ChromaDB 反馈集合实例

    返回:
        True 表示清空成功
    """
    try:
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
# 模块自测
# ============================================================================

if __name__ == "__main__":
    import tempfile
    from src.chroma_client import (
        create_chroma_client,
        create_embedding_function,
        get_or_create_collections,
    )

    print("=" * 60)
    print("测试 memory 模块")
    print("=" * 60)

    # 准备测试环境
    ef = create_embedding_function()
    with tempfile.TemporaryDirectory() as tmpdir:
        client = create_chroma_client(persist_path=tmpdir)
        mem_col, fb_col = get_or_create_collections(client, ef)

        # 测试存储记忆
        print("\n[1] 存储记忆...")
        mid1 = store_memory(mem_col, "用户喜欢吃辣")
        mid2 = store_memory(mem_col, "用户不吃香菜")
        mid3 = store_memory(mem_col, "用户喜欢听摇滚乐")
        print(f"    已存储 3 条记忆: {mid1}, {mid2}, {mid3}")
        print("    ✓ 存储通过")

        # 测试获取所有记忆
        print("\n[2] 获取所有记忆...")
        all_mems = get_all_memories(mem_col, limit=10)
        print(f"    共 {len(all_mems)} 条")
        for m in all_mems:
            print(f"    - {m['content']} (id={m['id']})")
        assert len(all_mems) == 3, f"应有 3 条，实际 {len(all_mems)}"
        print("    ✓ 列表获取通过")

        # 测试语义检索
        print("\n[3] 语义检索...")
        results = retrieve_memories(mem_col, "推荐餐厅", n_results=2)
        print(f"    查询「推荐餐厅」→ 返回 {len(results)} 条:")
        for r in results:
            print(f"    - {r['content']} (距离={r['distance']:.4f})")
        assert len(results) >= 1, "应至少返回 1 条"
        print("    ✓ 检索通过")

        # 测试删除单条记忆
        print("\n[4] 删除单条记忆...")
        ok = delete_memory(mem_col, mid3)
        print(f"    删除 {mid3}: {'成功' if ok else '失败'}")
        all_mems = get_all_memories(mem_col, limit=10)
        print(f"    剩余: {len(all_mems)} 条")
        assert len(all_mems) == 2, f"应有 2 条，实际 {len(all_mems)}"
        print("    ✓ 单条删除通过")

        # 测试清空全部
        print("\n[5] 清空全部...")
        ok = clear_all_memories(mem_col, fb_col)
        all_mems = get_all_memories(mem_col, limit=10)
        print(f"    清空后: {len(all_mems)} 条")
        assert len(all_mems) == 0, f"应为 0 条，实际 {len(all_mems)}"
        print("    ✓ 清空全部通过")

    print("\n" + "=" * 60)
    print("memory 模块：全部测试通过 ✓")
    print("=" * 60)
