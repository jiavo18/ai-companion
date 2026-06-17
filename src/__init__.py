"""
===============================================================================
AI伴侣 · src 包
===============================================================================
模块化架构：
  chroma_client  — ChromaDB 客户端与嵌入函数封装
  emotion        — 情感检测与语气指引
  memory         — 长期记忆存储、检索与遗忘机制
  feedback       — 反馈学习与风格约束解析
  proactive      — 主动提问检测与冷却机制
  prompt_builder — 动态 System Prompt 构建
  companion      — 核心编排类，整合所有模块
===============================================================================
"""

from src.companion import Companion
from src.chroma_client import (
    create_embedding_function,
    create_chroma_client,
    get_or_create_collections,
    create_llm,
)
from src.emotion import detect_emotion, get_emotion_response_guide
from src.memory import (
    store_memory,
    retrieve_memories,
    get_all_memories,
    delete_memory,
    clear_all_memories,
)
from src.feedback import store_feedback, retrieve_feedback, parse_feedback_to_constraints
from src.proactive import check_trigger_conditions, get_last_active_time, update_last_active_time
from src.prompt_builder import build_system_prompt

__all__ = [
    "Companion",
    "create_embedding_function",
    "create_chroma_client",
    "get_or_create_collections",
    "create_llm",
    "detect_emotion",
    "get_emotion_response_guide",
    "store_memory",
    "retrieve_memories",
    "get_all_memories",
    "delete_memory",
    "clear_all_memories",
    "store_feedback",
    "retrieve_feedback",
    "parse_feedback_to_constraints",
    "check_trigger_conditions",
    "get_last_active_time",
    "update_last_active_time",
    "build_system_prompt",
]
