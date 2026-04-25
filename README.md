# FusionRAG

FusionRAG 是一个基于 FastAPI 的 RAG 知识库项目，包含文档导入、向量检索、知识图谱查询、重排序、流式问答和简单 Web 页面。

## 主要功能

- 上传文档并导入知识库
- 支持 PDF 转换和图片摘要提取
- 使用 MinIO、MongoDB、Milvus 存储文件、元数据和向量数据
- 支持向量检索、RRF 融合、重排序和可选的联网搜索
- 通过统一的 FastAPI 服务提供登录、文档、会话和问答接口

## 项目结构

```text
app/
  import_process/     文档导入流程
  query_process/      检索和问答流程
  clients/            MongoDB、Milvus、MinIO、Neo4j 工具
  lm/                 大模型、Embedding 和重排序工具
  web/                简单前端页面
docker/knowledgebase/ Docker Compose 依赖服务
doc/                  示例文档
prompts/              提示词模板
```

## 环境要求

- Python 3.11+
- Docker 和 Docker Compose
- uv，或其他 Python 依赖管理工具

## 快速启动

启动本地依赖服务：

```bash
cd docker/knowledgebase
docker compose up -d
```

安装 Python 依赖：

```bash
uv sync
```

根据本地服务和模型配置创建 `.env` 文件，然后启动服务：

```bash
uv run python app/unified_service.py
```

默认访问地址：

```text
http://127.0.0.1:8000
```

## 说明

本地运行数据不会提交到 Git，包括 `.env`、`.venv`、`logs/`、`output/` 和 Docker volume 数据。
