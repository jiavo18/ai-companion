"""
===============================================================================
反馈学习模块 —— 用户反馈的存储、检索与规则解析
===============================================================================
用户通过"反馈：xxx"指令提供风格偏好，系统自动解析为具体约束，
影响后续对话的回答风格。
===============================================================================
"""

import uuid
from datetime import datetime


# ============================================================================
# 反馈存储与检索
# ============================================================================

def store_feedback(feedback_collection, content: str) -> str:
    """
    将用户反馈存入反馈集合。

    参数:
        feedback_collection: ChromaDB 反馈集合实例
        content: 反馈内容（如"回答太长了，精简一点"）

    返回:
        feedback_id: 反馈标识
    """
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


def retrieve_feedback(feedback_collection, n_results: int = 3) -> list[str]:
    """
    检索最新的用户反馈。

    参数:
        feedback_collection: ChromaDB 反馈集合实例
        n_results: 返回的反馈数量

    返回:
        反馈内容文本列表（按时间倒序）
    """
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


# ============================================================================
# 反馈解析
# ============================================================================

def parse_feedback_to_constraints(feedbacks: list[str]) -> str:
    """
    将用户反馈解析为系统约束文本。

    提取常见模式并生成具体的约束指令：
    - "精简/啰嗦/太长" → 简洁约束
    - "详细/展开/深入" → 详细约束
    - "温柔/温和" → 语气柔和约束
    - "幽默/有趣" → 幽默约束
    - "专业/严谨" → 专业约束
    - 其他 → 保留原文作为软约束

    参数:
        feedbacks: 反馈内容列表

    返回:
        格式化后的约束文本（每行一条），无反馈时返回空字符串
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
        known_constraints = [
            "回答应尽量简洁精炼，控制在100字以内，避免冗余表达。",
            "回答应更加详细深入，提供充分的解释和背景信息。",
            "请使用更加温和、柔和的语气。",
            "请在回答中适当加入幽默感，让对话更轻松。",
            "请保持专业严谨的表达风格。",
        ]
        if not constraints or constraints[-1] not in known_constraints:
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
    print("测试 feedback 模块")
    print("=" * 60)

    # 准备测试环境
    ef = create_embedding_function()
    with tempfile.TemporaryDirectory() as tmpdir:
        client = create_chroma_client(persist_path=tmpdir)
        mem_col, fb_col = get_or_create_collections(client, ef)

        # 测试存储反馈
        print("\n[1] 存储反馈...")
        fid1 = store_feedback(fb_col, "你回答太啰嗦了，精简一点")
        fid2 = store_feedback(fb_col, "请保持更专业严谨的风格")
        print(f"    已存储 2 条反馈: {fid1}, {fid2}")
        print("    ✓ 存储通过")

        # 测试检索反馈
        print("\n[2] 检索反馈...")
        feedbacks = retrieve_feedback(fb_col, n_results=3)
        print(f"    共 {len(feedbacks)} 条:")
        for fb in feedbacks:
            print(f"    - {fb}")
        assert len(feedbacks) == 2, f"应有 2 条，实际 {len(feedbacks)}"
        print("    ✓ 检索通过")

        # 测试解析约束
        print("\n[3] 解析约束...")
        constraints = parse_feedback_to_constraints(feedbacks)
        print(f"    解析结果:\n{constraints}")
        assert "简洁精炼" in constraints, "应包含简洁约束"
        assert "专业严谨" in constraints, "应包含专业约束"
        print("    ✓ 解析通过")

    # 测试空反馈
    print("\n[4] 空反馈测试...")
    result = parse_feedback_to_constraints([])
    assert result == "", f"空反馈应返回空字符串，实际: {result}"
    print("    ✓ 空反馈通过")

    print("\n" + "=" * 60)
    print("feedback 模块：全部测试通过 ✓")
    print("=" * 60)
