# FusionRAG

FusionRAG is a FastAPI-based RAG project with document import, vector retrieval, knowledge graph search, reranking, chat streaming, and a simple web UI.

## Features

- Upload and import documents into the knowledge base
- Convert PDF documents and extract image summaries
- Store files and metadata with MinIO, MongoDB, and Milvus
- Retrieve answers with embedding search, RRF, reranking, and optional web search
- Provide chat, document, session, and auth APIs through a unified FastAPI service

## Project Structure

```text
app/
  import_process/     Document import workflow
  query_process/      Retrieval and chat workflow
  clients/            MongoDB, Milvus, MinIO, Neo4j helpers
  lm/                 LLM, embedding, and reranker utilities
  web/                Simple frontend pages
docker/knowledgebase/ Docker Compose dependencies
doc/                  Sample documents
prompts/              Prompt templates
```

## Requirements

- Python 3.11+
- Docker and Docker Compose
- uv, or another Python dependency manager

## Quick Start

Start the local infrastructure:

```bash
cd docker/knowledgebase
docker compose up -d
```

Install dependencies:

```bash
uv sync
```

Create a local `.env` file based on your service and model settings, then run:

```bash
uv run python app/unified_service.py
```

The API starts on `http://127.0.0.1:8000` by default.

## Notes

Local runtime data is intentionally ignored by Git, including `.env`, `.venv`, `logs/`, `output/`, and Docker volume data.
