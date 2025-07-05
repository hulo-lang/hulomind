# Hulo Knowledge Base

基于 RAG + MCP + LLMs 的 Hulo 编程语言知识库系统。

## 功能特性

- 📚 **多语言文档支持**: 支持英文和中文文档的检索
- 🔍 **语义搜索**: 基于向量数据库的语义搜索
- 🤖 **LLM 集成**: 支持多种 LLM 模型
- 🔌 **MCP 协议**: 提供 MCP 服务器接口
- 🚀 **高性能**: 基于 FastAPI 的高性能 API 服务

## 快速开始

### 环境要求

- Python 3.9+
- uv (推荐) 或 pip

### 安装依赖

```bash
# 使用 uv
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 启动服务

```bash
# 开发模式
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# 生产模式
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## API 文档

启动服务后访问: http://localhost:8000/docs

## MCP 集成

本系统提供 MCP 服务器，支持以下功能：

- 文档检索
- 语义搜索
- 代码示例查询
- 语法规则查询

## 项目结构

```
src/
├── main.py              # FastAPI 应用入口
├── config/              # 配置文件
├── models/              # 数据模型
├── services/            # 业务逻辑服务
├── processors/          # 文档处理器
├── vectorstore/         # 向量数据库
└── mcp/                 # MCP 服务器
```

## 开发

```bash
# 安装开发依赖
uv sync --extra dev

# 运行测试
uv run pytest

# 代码格式化
uv run black src/
uv run isort src/
``` 

# 运行完整的知识库初始化

uv run python scripts/init_knowledge_base.py

# 启动 MCP 服务器

uv run python scripts/run_mcp_server.py

# 或者启动主 API 服务
uv run uvicorn src.main:app --reload