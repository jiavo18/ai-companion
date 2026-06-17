"""
===============================================================================
系统提示词构建器 —— 动态组装 System Prompt
===============================================================================
根据检索到的记忆、用户反馈约束、情感指引和主动关心文本，
动态构建每次对话的系统提示词。
===============================================================================
"""


def build_system_prompt(
    memories: list[dict],
    feedback_constraints: str,
    emotion_guide: str,
    proactive_care: str | None,
    ai_name: str = "禾苗",
) -> str:
    """
    动态构建系统提示词，整合所有上下文信息。

    组成结构:
    1. 核心原则（AI 人格定义）
    2. 用户偏好约束（来自反馈学习）
    3. 关于用户的重要记忆（来自向量检索）
    4. 当前回应策略（来自情感检测）
    5. 主动关心（可选，来自触发条件检测）

    参数:
        memories: 检索到的用户记忆列表
        feedback_constraints: 反馈约束文本（格式化后）
        emotion_guide: 情感回应指引
        proactive_care: 主动关心语句（可选，通常为 None）
        ai_name: AI 的名字

    返回:
        完整的系统提示词文本
    """
    prompt_parts = [
        f"你是一位善解人意、温和体贴的AI伴侣。你的名字是「{ai_name}」。",
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
        prompt_parts.append(
            "请在回答中自然地融入这些记忆，让用户感受到你真正记得关于TA的事。"
        )
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
# 模块自测
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试 prompt_builder 模块")
    print("=" * 60)

    # 测试完整 prompt 构建
    print("\n[1] 测试完整 System Prompt 构建...")
    memories = [
        {"content": "用户喜欢吃辣", "created_at": "2026-06-15T10:30:00"},
        {"content": "用户不吃香菜", "created_at": "2026-06-14T08:20:00"},
    ]
    feedback_constraints = "- 回答应尽量简洁精炼，控制在100字以内"
    emotion_guide = "用户情绪积极。请在回应中保持热情、活泼的语气。"
    proactive_care = None  # 主动关心已移至 LLM 回复前缀

    prompt = build_system_prompt(
        memories=memories,
        feedback_constraints=feedback_constraints,
        emotion_guide=emotion_guide,
        proactive_care=proactive_care,
        ai_name="禾苗",
    )

    print(f"    Prompt 长度: {len(prompt)} 字符")
    print("    关键段落检查:")
    for keyword in ["禾苗", "核心原则", "用户偏好", "喜欢吃辣", "不吃香菜", "回应策略"]:
        assert keyword in prompt, f"缺少关键词: {keyword}"
        print(f"      ✓ 包含「{keyword}」")
    print("    ✓ 完整构建通过")

    # 测试空记忆
    print("\n[2] 测试空记忆/空反馈...")
    prompt = build_system_prompt(
        memories=[],
        feedback_constraints="",
        emotion_guide="用户情绪中性。请保持常规的友好、专业语气。",
        proactive_care=None,
    )
    assert "禾苗" in prompt
    assert "核心原则" in prompt
    assert "用户偏好" not in prompt  # 空反馈不应出现偏好段
    assert "关于用户的重要记忆" not in prompt  # 空记忆不应出现记忆段
    print("    ✓ 空数据处理通过")

    # 测试自定义 AI 名字
    print("\n[3] 测试自定义 AI 名字...")
    prompt = build_system_prompt(
        memories=[{"content": "测试", "created_at": "2026-01-01"}],
        feedback_constraints="- 测试约束",
        emotion_guide="中性",
        proactive_care=None,
        ai_name="小明助手",
    )
    assert "小明助手" in prompt
    print("    ✓ 自定义名字通过")

    print("\n" + "=" * 60)
    print("prompt_builder 模块：全部测试通过 ✓")
    print("=" * 60)
