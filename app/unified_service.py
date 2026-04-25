import asyncio
import json
import mimetypes
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.clients.milvus_utils import get_milvus_client
from app.clients.minio_utils import get_minio_client
from app.clients.mongo_auth_utils import (
    authenticate_user,
    create_access_token,
    get_current_user,
    register_user,
    require_admin,
)
from app.clients.mongo_history_utils import clear_history, get_history_mongo_tool, get_recent_messages
from app.conf.milvus_config import milvus_config
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import get_default_state
from app.query_process.agent.main_graph import query_app
from app.utils.path_util import PROJECT_ROOT
from app.utils.sse_utils import SSEEvent, create_sse_queue, get_sse_queue, push_to_session, remove_sse_queue
from app.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    add_done_task,
    add_running_task,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    update_task_status,
)


BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent / "web"


class AuthRequest(BaseModel):
    username: str
    password: str
    role: Optional[str] = "user"
    admin_code: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户问题")
    session_id: Optional[str] = Field(None, description="前端会话 ID")


def _internal_session_id(username: str, session_id: str) -> str:
    return f"{username}:{session_id}"


def _external_session_id(username: str, session_id: str) -> str:
    prefix = f"{username}:"
    return session_id[len(prefix):] if session_id.startswith(prefix) else session_id


def _sse_data(payload: dict | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _run_import_graph(task_id: str, local_dir: str, local_file_path: str, parse_mode: str) -> None:
    try:
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        init_state = get_default_state()
        init_state["task_id"] = task_id
        init_state["local_dir"] = local_dir
        init_state["local_file_path"] = local_file_path
        init_state["parse_mode"] = parse_mode

        for event in kb_import_app.stream(init_state):
            for node_name in event.keys():
                add_done_task(task_id, node_name)

        update_task_status(task_id, TASK_STATUS_COMPLETED)
    except Exception as exc:
        update_task_status(task_id, TASK_STATUS_FAILED)
        push_to_session(task_id, SSEEvent.ERROR, {"error": str(exc)})


def _run_query_graph(session_id: str, user_query: str, is_stream: bool = True) -> None:
    try:
        default_state = {
            "original_query": user_query,
            "session_id": session_id,
            "is_stream": is_stream,
        }
        query_app.invoke(default_state)
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as exc:
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        push_to_session(session_id, SSEEvent.ERROR, {"error": str(exc)})
    finally:
        push_to_session(session_id, SSEEvent.CLOSE, {})


def _supported_upload(filename: str) -> bool:
    return filename.lower().endswith((".pdf", ".md"))


def _serialize_message(raw: dict) -> dict:
    return {
        "id": str(raw.get("_id", "")),
        "type": "human" if raw.get("role") == "user" else "ai",
        "content": raw.get("text", ""),
        "timestamp": raw.get("ts"),
        "rewritten_query": raw.get("rewritten_query", ""),
        "item_names": raw.get("item_names") or [],
        "image_urls": raw.get("image_urls") or [],
    }


def create_app() -> FastAPI:
    app = FastAPI(title="FusionRAG API", description="Project01 RAG trunk with unified UI and Mongo auth")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def index():
        index_path = WEB_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="index.html not found")
        return FileResponse(index_path)

    @app.post("/auth/register")
    async def register(request: AuthRequest):
        user = register_user(request.username, request.password, request.role, request.admin_code)
        token = create_access_token(user["username"], user["role"])
        return {"access_token": token, "username": user["username"], "role": user["role"]}

    @app.post("/auth/login")
    async def login(request: AuthRequest):
        user = authenticate_user(request.username, request.password)
        if not user:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        token = create_access_token(user["username"], user["role"])
        return {"access_token": token, "username": user["username"], "role": user["role"]}

    @app.get("/auth/me")
    async def me(current_user: dict = Depends(get_current_user)):
        return current_user

    @app.post("/documents/upload")
    async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        parse_mode: str = Form("auto"),
        _: dict = Depends(require_admin),
    ):
        filename = file.filename or ""
        if not filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        if not _supported_upload(filename):
            raise HTTPException(status_code=400, detail="项目一主干 MVP 仅支持 PDF / Markdown")
        if parse_mode not in {"auto", "advanced", "fast"}:
            raise HTTPException(status_code=400, detail="parse_mode 仅支持 auto/advanced/fast")
        task_id = str(uuid.uuid4())
        add_running_task(task_id, "upload_file")

        task_local_dir = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d") / task_id
        task_local_dir.mkdir(parents=True, exist_ok=True)
        local_file_abs_path = task_local_dir / filename
        with open(local_file_abs_path, "wb") as file_buffer:
            shutil.copyfileobj(file.file, file_buffer)

        try:
            minio_client = get_minio_client()
            if minio_client:
                bucket_name = os.getenv("MINIO_BUCKET_NAME", "knowledge-base-files")
                object_name = f"{os.getenv('MINIO_PDF_DIR', 'pdf_files')}/{datetime.now().strftime('%Y%m%d')}/{filename}"
                minio_client.fput_object(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    file_path=str(local_file_abs_path),
                    content_type=file.content_type,
                )
        except Exception:
            pass

        add_done_task(task_id, "upload_file")
        background_tasks.add_task(_run_import_graph, task_id, str(task_local_dir), str(local_file_abs_path), parse_mode)
        return {
            "job_id": task_id,
            "task_id": task_id,
            "filename": filename,
            "parse_mode": parse_mode,
            "message": "文件已上传，正在后台执行知识库导入流程",
        }

    @app.get("/documents/status/{task_id}")
    async def document_status(task_id: str, _: dict = Depends(require_admin)):
        return {
            "task_id": task_id,
            "status": get_task_status(task_id),
            "done_list": get_done_task_list(task_id),
            "running_list": get_running_task_list(task_id),
        }

    @app.get("/documents")
    async def list_documents(_: dict = Depends(require_admin)):
        client = get_milvus_client()
        collection_name = milvus_config.chunks_collection
        if not client or not collection_name or not client.has_collection(collection_name):
            return {"documents": []}
        rows = client.query(
            collection_name=collection_name,
            output_fields=["file_title", "item_name"],
            limit=10000,
        )
        stats: Dict[str, Dict[str, Any]] = {}
        for row in rows or []:
            name = row.get("file_title") or "未知文档"
            item = stats.setdefault(name, {"filename": name, "item_name": row.get("item_name", ""), "chunk_count": 0})
            item["chunk_count"] += 1
            if not item.get("item_name") and row.get("item_name"):
                item["item_name"] = row.get("item_name")
        return {"documents": list(stats.values())}

    @app.delete("/documents/{filename}")
    async def delete_document(filename: str, _: dict = Depends(require_admin)):
        client = get_milvus_client()
        collection_name = milvus_config.chunks_collection
        if not client or not collection_name or not client.has_collection(collection_name):
            raise HTTPException(status_code=404, detail="向量集合不存在")
        result = client.delete(collection_name=collection_name, filter=f'file_title == "{filename}"')
        if hasattr(client, "flush"):
            try:
                client.flush(collection_name=collection_name)
            except Exception:
                pass
        return {"filename": filename, "result": result, "message": f"已尝试删除 {filename} 的向量数据"}

    @app.get("/assets/image")
    async def proxy_image(url: str):
        parsed = urlparse(url)
        raw_path = parsed.path if parsed.scheme else url
        path = unquote(raw_path).lstrip("/")
        parts = path.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise HTTPException(status_code=400, detail="无法解析图片地址")

        bucket_name, object_name = parts[0], parts[1]
        if "example.com" in url.lower() or "image_placeholder" in url.lower():
            raise HTTPException(status_code=404, detail="忽略占位图片")

        try:
            minio_client = get_minio_client()
            if not minio_client:
                raise RuntimeError("MinIO客户端不可用")
            response = minio_client.get_object(bucket_name, object_name)
            content_type = response.headers.get("content-type") or mimetypes.guess_type(object_name)[0] or "image/jpeg"

            def iterator():
                try:
                    for chunk in response.stream(32 * 1024):
                        yield chunk
                finally:
                    response.close()
                    response.release_conn()

            return StreamingResponse(iterator(), media_type=content_type)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"图片读取失败：{exc}") from exc

    @app.post("/chat/stream")
    async def chat_stream(request: ChatRequest, current_user: dict = Depends(get_current_user)):
        session_id = request.session_id or f"session_{int(datetime.now().timestamp())}"
        internal_session_id = _internal_session_id(current_user["username"], session_id)
        create_sse_queue(internal_session_id)
        update_task_status(internal_session_id, TASK_STATUS_PROCESSING, True)

        async def event_generator():
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(None, _run_query_graph, internal_session_id, request.message, True)
            stream_queue = get_sse_queue(internal_session_id)
            if stream_queue is None:
                yield _sse_data({"type": "error", "content": "SSE 队列创建失败"})
                yield _sse_data("[DONE]")
                return

            try:
                while True:
                    try:
                        msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
                    except Exception:
                        if future.done():
                            break
                        continue

                    event = msg.get("event")
                    data = msg.get("data") or {}
                    if event == SSEEvent.DELTA:
                        yield _sse_data({"type": "content", "content": data.get("delta", "")})
                    elif event == SSEEvent.PROGRESS:
                        running = data.get("running_list") or []
                        done = data.get("done_list") or []
                        label = running[-1] if running else (done[-1] if done else "处理中")
                        yield _sse_data({"type": "rag_step", "step": {"icon": "●", "label": label, "detail": data.get("status", "")}})
                    elif event == SSEEvent.FINAL:
                        yield _sse_data({"type": "final", "content": data})
                        break
                    elif event == SSEEvent.ERROR:
                        yield _sse_data({"type": "error", "content": data.get("error", "未知错误")})
                        break
                    elif event == SSEEvent.CLOSE:
                        break
                await future
            finally:
                remove_sse_queue(internal_session_id)
                yield _sse_data("[DONE]")

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    @app.get("/sessions")
    async def list_sessions(current_user: dict = Depends(get_current_user)):
        mongo_tool = get_history_mongo_tool()
        prefix = f"{current_user['username']}:"
        session_ids = mongo_tool.chat_message.distinct("session_id", {"session_id": {"$regex": f"^{prefix}"}})
        sessions = []
        for sid in session_ids:
            latest = mongo_tool.chat_message.find({"session_id": sid}).sort("ts", -1).limit(1)
            latest_item = next(iter(latest), {})
            count = mongo_tool.chat_message.count_documents({"session_id": sid})
            sessions.append(
                {
                    "session_id": _external_session_id(current_user["username"], sid),
                    "updated_at": latest_item.get("ts", 0),
                    "message_count": count,
                }
            )
        sessions.sort(key=lambda item: item.get("updated_at") or 0, reverse=True)
        return {"sessions": sessions}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str, current_user: dict = Depends(get_current_user)):
        internal = _internal_session_id(current_user["username"], session_id)
        messages = [_serialize_message(item) for item in get_recent_messages(internal, limit=100)]
        return {"messages": messages}

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str, current_user: dict = Depends(get_current_user)):
        internal = _internal_session_id(current_user["username"], session_id)
        deleted = clear_history(internal)
        return {"session_id": session_id, "deleted_count": deleted, "message": "会话已删除"}

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
