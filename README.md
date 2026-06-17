# 禾苗 · AI 伴侣

具备**长期记忆**、**自我进化**、**情感感知**与**用户系统**的智能对话系统。

## 核心能力

| 能力 | 说明 |
|------|------|
| 长期记忆存储 | 用户说"记住：我喜欢吃辣"，AI 将记忆向量化存入 ChromaDB |
| 自动记忆提取 | 无需手动指令，AI 自动从聊天中识别并记住你的信息 |
| 智能记忆检索 | 用户提问时自动检索最相关记忆，融入个性化回答 |
| 主动关心 | 检测对话间隔、压力关键词，主动表达关心 |
| 反馈学习 | 用户反馈"太啰嗦了"，AI 自动调整后续回答风格 |
| 情感感知 | 支持关键词匹配 / LLM 语义分析双模式 |
| 用户系统 | 注册登录，每人独立记忆库，对话刷新不丢 |
| 记忆可视化 | 侧边栏展示所有已存储记忆，支持删除 |
| 记忆遗忘+去重 | 超过 30 天未访问自动降权，重复信息自动合并 |
| 情绪曲线 | Plotly 绘制的情绪波动图表 |

## 技术架构

```
Streamlit (UI) → LangChain → DeepSeek Chat (LLM)
       ↓              ↓
   SQLite          ChromaDB (向量存储)
 (用户/对话)           ↓
              Sentence-Transformers (Embedding)
```

## 本地运行

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd ai_companion
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

首次运行时会自动下载 Embedding 模型（约 120MB），请耐心等待。

### 3. 配置 API Key

在项目根目录创建 `.env` 文件：

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

API Key 获取地址：[platform.deepseek.com](https://platform.deepseek.com)

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501` 即可使用。

## Docker 部署

一条命令在任何服务器上跑起来。

### 1. 构建镜像

```bash
docker build -t ai-companion .
```

首次构建会下载 Python 基础镜像并安装依赖，约 3-5 分钟。

### 2. 启动容器

```bash
docker run -p 8501:8501 \
  -e DEEPSEEK_API_KEY=sk-your-key \
  -v ai-companion-data:/app/data \
  -v ai-companion-chroma:/app/chroma_db \
  ai-companion
```

- `-e DEEPSEEK_API_KEY=sk-your-key` — 设置 API Key
- `-v` — 持久化数据（SQLite 数据库和 ChromaDB 向量库）
- 浏览器访问 `http://localhost:8501`

### 3. 停止与重启

```bash
docker stop <container-id>
docker start <container-id>
```

## 部署到 Streamlit Cloud

### 步骤 1：推送代码到 GitHub

将项目推送到一个 GitHub 仓库（确保 `.env` 在 `.gitignore` 中，不会泄露）。

### 步骤 2：连接 Streamlit Cloud

1. 访问 [share.streamlit.io](https://share.streamlit.io)
2. 使用 GitHub 账号登录
3. 点击「New app」
4. 选择仓库、分支（main）、主文件路径（`app.py`）

### 步骤 3：设置 Secrets

在 Streamlit Cloud 的应用设置中：

1. 进入「Settings」→「Secrets」
2. 添加以下内容：

```toml
DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

3. 保存并重启应用

### 步骤 4：访问

应用会自动部署，分配一个 `.streamlit.app` 域名。

## 使用指南

### 记忆指令

```
记住：我不吃香菜
记住：我的生日是5月20日
记住：我目前在学Python
```

### 反馈指令

```
反馈：你回答太啰嗦了，以后精简一点
反馈：语气可以再温和一些
反馈：请给出更专业的建议
```

### 普通对话

直接聊天，AI 会自动：
- 检测你的情绪并适配语气（关键词 / LLM 双模式）
- 从对话中自动提取并记住你的个人信息
- 检索相关记忆融入回答
- 在合适的时机主动关心你
- 遵循你之前的反馈偏好

### 用户系统

1. 首次使用先注册账号（用户名≥3字符，密码≥4字符）
2. 登录后每个人的记忆和对话完全隔离
3. 支持多对话：侧边栏新建/切换/删除对话
4. 对话自动存 SQLite，刷新不丢

## 项目结构

```
ai_companion/
├── app.py              # Streamlit UI（登录/聊天/侧边栏）
├── Dockerfile          # Docker 部署
├── .dockerignore
├── requirements.txt    # Python 依赖
├── README.md
├── .env.example        # 环境变量示例
├── .gitignore
└── src/
    ├── __init__.py
    ├── chroma_client.py     # ChromaDB 客户端 + Embedding
    ├── emotion.py           # 情感检测（关键词 + LLM）
    ├── memory.py            # 记忆 CRUD + 遗忘权重
    ├── memory_extractor.py  # 自动记忆提取 + 去重
    ├── feedback.py          # 反馈学习 + 规则解析
    ├── proactive.py         # 主动提问（双触发 + 冷却）
    ├── prompt_builder.py    # System Prompt 组装
    ├── companion.py         # 核心编排类
    ├── auth.py              # 用户注册登录
    ├── conversation.py      # 对话持久化
    └── database.py          # SQLite 封装
```

## 环境变量

| 变量名 | 说明 | 必需 |
|--------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | 是 |

## 常见问题

**Q: 首次运行很慢？**
A: 首次运行需要下载 Sentence-Transformers 模型（约 120MB），后续使用缓存。

**Q: ChromaDB 数据存储在哪里？**
A: 默认存储在项目目录的 `chroma_db/` 文件夹中。

**Q: Streamlit Cloud 上 ChromaDB 能正常工作吗？**
A: 可以。Streamlit Cloud 提供可写的文件系统，ChromaDB 的持久化文件会保存在应用实例上。注意：应用休眠后文件可能被清理。
