"""
===============================================================================
情感检测模块 —— 基于关键词匹配的情感极性分析
===============================================================================
通过预置的正/负面情感关键词库，检测用户输入中的情感倾向，
并生成对应的回应语气指引。
===============================================================================
"""

# ============================================================================
# 情感关键词库
# ============================================================================

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


# ============================================================================
# 情感检测
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
        return {
            "polarity": "中性",
            "score": 0.5,
            "matched_keywords": matched_pos + matched_neg,
        }


# ============================================================================
# LLM 情感分析（比关键词更准，理解语义而非匹配字词）
# ============================================================================

LLM_EMOTION_PROMPT = """分析这句话的情感，严格只返回JSON（不要其他文字）：
"{text}"
{"polarity":"正面/负面/中性","score":0.0~1.0,"emotion":"具体情感名称"}"""


def detect_emotion_llm(text: str, llm) -> dict:
    """
    用 LLM 做情感分析，理解语义而非简单关键词匹配。

    优势：能区分"烦死了今天怎么这么开心"（正面）和"我好难过"（负面），
          关键词模式会被"烦"字误导。

    参数:
        text: 用户输入文本
        llm: ChatOpenAI 实例（共用对话 LLM，不额外调 API）

    返回:
        {
            "polarity": "正面" | "负面" | "中性",
            "score": float (0~1),
            "emotion": str (具体情感名称，如"焦虑"/"兴奋"/"失落"/"满足"),
            "mode": "llm",
        }
        失败时 fallback 到关键词模式
    """
    import re
    import json
    from langchain_core.messages import HumanMessage

    prompt = LLM_EMOTION_PROMPT.format(text=text)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # 提取 JSON（LLM 有时会在外面包说明文字）
        json_match = re.search(r'\{[^{}]*"polarity"[^{}]*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)

        data = json.loads(raw)
        polarity = data.get("polarity", "中性")

        # 标准化极性值
        if polarity not in ("正面", "负面", "中性"):
            polarity = "中性"

        score = float(data.get("score", 0.5))
        score = max(0.0, min(1.0, score))  # 夹到 0~1

        return {
            "polarity": polarity,
            "score": score,
            "emotion": data.get("emotion", ""),
            "mode": "llm",
        }
    except Exception:
        # LLM 调用失败 → fallback 到关键词模式
        keyword_result = detect_emotion(text)
        keyword_result["mode"] = "keyword"
        keyword_result["emotion"] = ""
        return keyword_result


# ============================================================================
# 回应语气指引
# ============================================================================

def get_emotion_response_guide(polarity: str) -> str:
    """
    根据情感极性返回回应语气指引。

    参数:
        polarity: "正面" | "负面" | "中性"

    返回:
        语气指引文本
    """
    if polarity == "正面":
        return "用户情绪积极。请在回应中保持热情、活泼的语气，可以适当使用感叹号来表达共鸣。"
    elif polarity == "负面":
        return "用户情绪偏向负面。请使用温和、共情、安抚的语气。先表达理解与关心，再给出建议。避免过于轻快的表达。"
    else:
        return "用户情绪中性。请保持常规的友好、专业语气。"


# ============================================================================
# 模块自测
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试 emotion 模块")
    print("=" * 60)

    # 测试正面情感
    print("\n[1] 测试正面情感检测...")
    result = detect_emotion("今天真是太开心了！你好棒！")
    print(f"    输入: 今天真是太开心了！你好棒！")
    print(f"    极性: {result['polarity']}")
    print(f"    分数: {result['score']:.2f}")
    print(f"    匹配: {result['matched_keywords']}")
    print(f"    指引: {get_emotion_response_guide(result['polarity'])}")
    assert result["polarity"] == "正面", "应为正面"
    print("    ✓ 正面检测通过")

    # 测试负面情感
    print("\n[2] 测试负面情感检测...")
    result = detect_emotion("我很难过，最近太累了")
    print(f"    输入: 我很难过，最近太累了")
    print(f"    极性: {result['polarity']}")
    print(f"    分数: {result['score']:.2f}")
    print(f"    匹配: {result['matched_keywords']}")
    print(f"    指引: {get_emotion_response_guide(result['polarity'])}")
    assert result["polarity"] == "负面", "应为负面"
    print("    ✓ 负面检测通过")

    # 测试中性情感
    print("\n[3] 测试中性情感检测...")
    result = detect_emotion("今天天气怎么样")
    print(f"    输入: 今天天气怎么样")
    print(f"    极性: {result['polarity']}")
    print(f"    分数: {result['score']:.2f}")
    assert result["polarity"] == "中性", "应为中性"
    print("    ✓ 中性检测通过")

    print("\n" + "=" * 60)
    print("emotion 模块：全部测试通过 ✓")
    print("=" * 60)
