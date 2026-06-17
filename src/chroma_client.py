"""
===============================================================================
ChromaDB 客户端封装 —— 提供嵌入函数、客户端和集合的创建与管理
===============================================================================
技术栈: ChromaDB + SentenceTransformers + DeepSeek API
===============================================================================
"""

# ============================================================================
# 关键：必须在导入 chromadb 之前设置，解决 protobuf 兼容性问题
# 参考: https://developers.google.com/protocol-buffers/docs/news/2022-05-06
# ============================================================================
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import chromadb
from chromadb.utils import embedding_functions

from langchain_openai import ChatOpenAI

# ============================================================================
# 常量
# ============================================================================

# 默认 ChromaDB 持久化路径（可由调用方覆盖）
DEFAULT_CHROMA_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db"
)

# 默认 Embedding 模型名称
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


# ============================================================================
# 嵌入函数
# ============================================================================

def create_embedding_function(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """
    创建嵌入函数。

    使用 SentenceTransformer 的轻量多语言模型，本地运行无需 API Key。
    首次运行会自动下载模型（约 120MB），后续使用缓存。

    参数:
        model_name: SentenceTransformer 模型名称

    返回:
        SentenceTransformerEmbeddingFunction 实例
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )


# ============================================================================
# ChromaDB 客户端
# ============================================================================

def create_chroma_client(persist_path: str = DEFAULT_CHROMA_DB_PATH):
    """
    创建 ChromaDB 持久化客户端。

    参数:
        persist_path: 持久化数据存储路径

    返回:
        chromadb.PersistentClient 实例
    """
    return chromadb.PersistentClient(path=persist_path)


# ============================================================================
# 集合管理
# ============================================================================

def get_or_create_collections(client, embedding_function):
    """
    获取或创建 ChromaDB 集合。

    参数:
        client: ChromaDB 客户端实例
        embedding_function: 嵌入函数实例

    返回:
        (memory_collection, feedback_collection) 元组
    """
    memory_collection = client.get_or_create_collection(
        name="user_memories",
        embedding_function=embedding_function,
        metadata={"description": "用户长期记忆存储"},
    )

    feedback_collection = client.get_or_create_collection(
        name="feedback_collection",
        embedding_function=embedding_function,
        metadata={"description": "用户反馈与风格偏好"},
    )

    return memory_collection, feedback_collection


# ============================================================================
# LLM 创建
# ============================================================================

def create_llm(api_key: str, model: str = "deepseek-chat") -> ChatOpenAI:
    """
    创建 DeepSeek ChatOpenAI 实例。

    参数:
        api_key: DeepSeek API Key
        model: 模型名称，默认 deepseek-chat

    返回:
        ChatOpenAI 实例
    """
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model=model,
        temperature=0.72,
        max_tokens=1024,
        timeout=45,
    )


# ============================================================================
# 模块自测
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试 chroma_client 模块")
    print("=" * 60)

    # 测试嵌入函数
    print("\n[1] 创建嵌入函数...")
    ef = create_embedding_function()
    print(f"    嵌入函数类型: {type(ef).__name__}")
    print("    ✓ 嵌入函数创建成功")

    # 测试 ChromaDB 客户端
    print("\n[2] 创建 ChromaDB 客户端...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        client = create_chroma_client(persist_path=tmpdir)
        print(f"    客户端类型: {type(client).__name__}")
        print("    ✓ 客户端创建成功")

        # 测试集合创建
        print("\n[3] 创建集合...")
        mem_col, fb_col = get_or_create_collections(client, ef)
        print(f"    记忆集合名称: {mem_col.name}")
        print(f"    反馈集合名称: {fb_col.name}")
        print("    ✓ 集合创建成功")

    # 测试 LLM 创建（不需要真实 API Key，只测对象构造）
    print("\n[4] 创建 LLM 实例...")
    llm = create_llm(api_key="sk-test")
    print(f"    LLM 类型: {type(llm).__name__}")
    print(f"    模型: {llm.model_name}")
    print("    ✓ LLM 实例创建成功")

    print("\n" + "=" * 60)
    print("chroma_client 模块：全部测试通过 ✓")
    print("=" * 60)
