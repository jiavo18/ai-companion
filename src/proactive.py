"""
===============================================================================
主动提问模块 —— 时间间隔 + 关键词连续触发 + 冷却机制
===============================================================================
检测两个触发条件：
  1. 超过 24 小时未对话 → 生成关怀问候
  2. 最近 3 轮用户消息连续包含压力关键词 → 生成放松建议
同一触发类型在冷却期（默认 1 小时）内不重复。
===============================================================================
"""

import os
import json
from datetime import datetime

# ============================================================================
# 常量
# ============================================================================

# —— 压力关键词列表（用于连续检测） ——
STRESS_KEYWORDS = ["累", "加班", "压力", "熬夜", "疲惫", "焦虑", "忙", "烦躁", "崩溃"]

# —— 主动关心触发关键词与对应建议 ——
CONCERN_PATTERNS = {
    "熬夜": "注意到你最近经常提到熬夜，身体是革命的本钱，记得早点休息。",
    "加班": "你最近似乎工作很忙，别忘了给自己留一些喘息的时间。",
    "累": "感觉你状态有些疲惫，要不要听听轻音乐放松一下？",
    "压力": "压力大的时候，深呼吸或者出去走走都会有帮助。",
    "焦虑": "你提到了焦虑，我想告诉你，这种感觉很正常，慢慢来。",
}

# —— 主动提问冷却时间（秒） —— 同一触发条件 1 小时内不重复
PROACTIVE_COOLDOWN_SECONDS = 3600

# —— 活跃时间文件路径 ——
DEFAULT_LAST_ACTIVE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "last_active.json"
)


# ============================================================================
# 冷却机制（接受外部 last_proactive_time 字典，解耦 Streamlit）
# ============================================================================

def _is_in_cooldown(last_proactive_time: dict, trigger_type: str,
                    cooldown_seconds: int = PROACTIVE_COOLDOWN_SECONDS) -> bool:
    """
    检查指定触发类型是否在冷却期内。

    参数:
        last_proactive_time: {"time_gap": datetime, "keyword_stress": datetime}
        trigger_type: "time_gap" 或 "keyword_stress"
        cooldown_seconds: 冷却时间（秒），默认 3600

    返回:
        True 表示冷却中，不应再次触发
    """
    last_time = last_proactive_time.get(trigger_type)
    if last_time is None:
        return False
    elapsed = (datetime.now() - last_time).total_seconds()
    return elapsed < cooldown_seconds


def _record_trigger(last_proactive_time: dict, trigger_type: str):
    """
    记录触发时间戳，启动冷却。

    参数:
        last_proactive_time: 冷却记录字典（会被原地修改）
        trigger_type: "time_gap" 或 "keyword_stress"
    """
    last_proactive_time[trigger_type] = datetime.now()


# ============================================================================
# 活跃时间管理
# ============================================================================

def get_last_active_time(file_path: str = DEFAULT_LAST_ACTIVE_FILE) -> datetime | None:
    """
    从本地文件读取上次活跃时间。

    参数:
        file_path: last_active.json 文件路径

    返回:
        上次活跃的 datetime，文件不存在则返回 None
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return datetime.fromisoformat(data["last_active"])
    except Exception:
        pass
    return None


def update_last_active_time(file_path: str = DEFAULT_LAST_ACTIVE_FILE):
    """
    更新本地活跃时间戳。

    参数:
        file_path: last_active.json 文件路径
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"last_active": datetime.now().isoformat()}, f, ensure_ascii=False)
    except Exception:
        pass


# ============================================================================
# 主动提问检测
# ============================================================================

def check_trigger_conditions(
    user_input: str,
    conversation_history: list[dict],
    last_proactive_time: dict,
    last_active_file: str = DEFAULT_LAST_ACTIVE_FILE,
    cooldown_seconds: int = PROACTIVE_COOLDOWN_SECONDS,
) -> str | None:
    """
    检测主动提问触发条件，返回关心语句或 None。

    触发条件 1 —— 时间间隔触发：
        如果用户超过 24 小时未对话，且该类型不在冷却期，
        生成一句欢迎/关心问候。

    触发条件 2 —— 关键词连续触发：
        如果用户在过去 3 轮对话中，每条消息都包含压力关键词，
        且该类型不在冷却期，生成一句放松建议。

    参数:
        user_input: 当前用户输入文本
        conversation_history: 完整对话历史（list of {"role": str, "content": str}）
        last_proactive_time: 冷却记录字典（会被原地修改）
        last_active_file: 活跃时间文件路径
        cooldown_seconds: 冷却时间（秒）

    返回:
        主动提问文本（将作为 AI 回答的第一句话），无触发则返回 None
    """
    now = datetime.now()
    care_parts = []

    # ============================================================
    # 触发条件 1：超过 24 小时未对话
    # ============================================================
    if not _is_in_cooldown(last_proactive_time, "time_gap", cooldown_seconds):
        last_active = get_last_active_time(file_path=last_active_file)
        if last_active:
            hours_since = (now - last_active).total_seconds() / 3600
            if hours_since > 24:
                days = int(hours_since / 24)
                care_parts.append(
                    f"已经 {days} 天没聊了，最近过得怎么样？有什么想和我分享的吗？"
                )
                _record_trigger(last_proactive_time, "time_gap")

    # ============================================================
    # 触发条件 2：最近 3 轮用户消息连续包含压力关键词
    # ============================================================
    if not _is_in_cooldown(last_proactive_time, "keyword_stress", cooldown_seconds):
        # 收集用户消息
        user_messages = [
            msg.get("content", "")
            for msg in conversation_history
            if msg.get("role") == "user"
        ]
        # 取最近 2 条历史 + 当前输入 = 共 3 条
        recent_user_texts = user_messages[-2:] + [user_input]

        if len(recent_user_texts) >= 3:
            # 检查每条消息是否至少包含一个压力关键词
            all_stressed = all(
                any(kw in text for kw in STRESS_KEYWORDS)
                for text in recent_user_texts[-3:]
            )
            if all_stressed:
                # 找出最后一条消息中匹配到的关键词，生成针对性建议
                last_text = recent_user_texts[-1]
                matched = [kw for kw in STRESS_KEYWORDS if kw in last_text]
                if "加班" in matched or "忙" in matched:
                    suggestion = "你最近似乎工作很忙，别忘了给自己留一些喘息的时间。"
                elif "累" in matched or "疲惫" in matched:
                    suggestion = "感觉你最近挺疲惫的，要不要听听轻音乐放松一下？"
                elif "熬夜" in matched:
                    suggestion = "注意到你最近经常熬夜，身体是革命的本钱，记得早点休息。"
                elif "压力" in matched or "焦虑" in matched or "烦躁" in matched:
                    suggestion = "你好像压力有点大，深呼吸或者出去走走都会有帮助。"
                else:
                    suggestion = "最近辛苦了，要不要给自己放个小假调整一下？"

                care_parts.append(suggestion)
                _record_trigger(last_proactive_time, "keyword_stress")

    if care_parts:
        return " ".join(care_parts)

    return None


# ============================================================================
# 模块自测
# ============================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("测试 proactive 模块")
    print("=" * 60)

    # 准备测试数据
    cooldown_tracker = {}  # 模拟 st.session_state.last_proactive_time

    # 测试冷却机制
    print("\n[1] 测试冷却机制...")
    assert not _is_in_cooldown(cooldown_tracker, "time_gap"), "初始不应冷却"
    _record_trigger(cooldown_tracker, "time_gap")
    assert _is_in_cooldown(cooldown_tracker, "time_gap"), "记录后应冷却"
    # 清除冷却
    cooldown_tracker.pop("time_gap")
    print("    ✓ 冷却机制通过")

    # 测试时间间隔触发（手动构造 last_active 文件）
    print("\n[2] 测试时间间隔触发...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = os.path.join(tmpdir, "last_active.json")
        # 写入 48 小时前
        old_time = datetime.now().replace()  # can't subtract here easily
        import time as _time
        old_iso = (datetime.now().replace(microsecond=0) - __import__('datetime').timedelta(hours=48)).isoformat()
        with open(tmp_file, "w") as f:
            json.dump({"last_active": old_iso}, f)

        result = check_trigger_conditions(
            "你好呀", [], cooldown_tracker,
            last_active_file=tmp_file
        )
        print(f"    触发结果: {result[:50] if result else 'None'}...")
        assert result is not None, "48h 未对话应触发时间间隔"
        assert "天没聊了" in result, "应包含关怀语句"
        print("    ✓ 时间间隔触发通过")

        # 验证冷却生效
        result2 = check_trigger_conditions(
            "你好", [], cooldown_tracker,
            last_active_file=tmp_file
        )
        assert result2 is None, "同一类型冷却中不应再次触发"
        print("    ✓ 冷却生效验证通过")

    # 测试关键词连续触发
    print("\n[3] 测试关键词连续触发...")
    cooldown_tracker2 = {}  # 新的冷却追踪器
    # 模拟 2 条历史 + 1 条当前 = 连续 3 条压力消息
    history = [
        {"role": "user", "content": "今天加班到很晚"},
        {"role": "assistant", "content": "辛苦了"},
        {"role": "user", "content": "是啊，累得不行"},
        {"role": "assistant", "content": "注意休息"},
    ]
    result = check_trigger_conditions(
        "压力好大，又熬夜了", history, cooldown_tracker2,
        last_active_file="/nonexistent.json"  # 不存在，跳过时间触发
    )
    print(f"    触发结果: {result[:60] if result else 'None'}...")
    assert result is not None, "连续压力关键词应触发"
    print("    ✓ 关键词连续触发通过")

    # 测试不触发的情况
    print("\n[4] 测试不触发情况...")
    cooldown_tracker3 = {}
    result = check_trigger_conditions(
        "今天天气真好", [
            {"role": "user", "content": "我想出去玩"},
            {"role": "assistant", "content": "好啊"},
        ], cooldown_tracker3,
        last_active_file="/nonexistent.json"
    )
    assert result is None, f"无触发条件应返回 None，实际: {result}"
    print("    ✓ 不触发验证通过")

    print("\n" + "=" * 60)
    print("proactive 模块：全部测试通过 ✓")
    print("=" * 60)
