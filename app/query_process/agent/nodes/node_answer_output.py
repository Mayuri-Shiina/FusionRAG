import sys
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.utils.escape_milvus_string_utils import escape_milvus_string
import re

_IMAGE_BLOCK_MARKER = "【图片】"
MAX_CONTEXT_CHARS = 12000
PLACEHOLDER_IMAGE_PATTERN = re.compile(
    r'!\[[^\]]*]\((?:https?://)?(?:www\.)?example\.com/[^)]*\)|https?://(?:www\.)?example\.com/\S+|https?://\S*image_placeholder\S*',
    re.IGNORECASE,
)
IMAGE_BLOCK_PATTERN = re.compile(r'\n?【图片】[\s\S]*$', re.MULTILINE)
MD_IMAGE_PATTERN = re.compile(
    r'!\[[^\]]*]\((https?://.*?\.(?:png|jpg|jpeg|gif|webp|bmp|svg))\s*\)',
    re.IGNORECASE | re.DOTALL,
)
MD_IMAGE_CANDIDATE_PATTERN = re.compile(
    r'!\[([^\]]*)]\((https?://.*?\.(?:png|jpg|jpeg|gif|webp|bmp|svg))\s*\)',
    re.IGNORECASE | re.DOTALL,
)
RAW_IMAGE_URL_PATTERN = re.compile(r'https?://[^\s<>"\]]+?\.(?:png|jpg|jpeg|gif|webp|bmp|svg)', re.IGNORECASE)
EMPTY_IMAGE_HINT_PATTERN = re.compile(
    r'\n?(?:可参考|请参考|参见|见)下方(?:的)?(?:相关)?图片[。；;,.，]?',
    re.IGNORECASE,
)
QUERY_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]{2,}")
QUERY_STOP_WORDS = {
    "什么", "怎么", "如何", "请问", "一下", "这个", "那个", "哪些", "哪里", "相关",
    "图片", "文档", "手册", "产品", "平台", "系统", "进行", "需要", "可以", "是否",
}


def _clean_generated_answer(text: str) -> str:
  if not text:
    return ""
  text = IMAGE_BLOCK_PATTERN.sub("", text)
  text = PLACEHOLDER_IMAGE_PATTERN.sub("", text)
  return text.strip()


def _clean_empty_image_hint(text: str) -> str:
  if not text:
    return ""
  return EMPTY_IMAGE_HINT_PATTERN.sub("", text).strip()


def _is_valid_image_url(url: str) -> bool:
    value = (url or "").strip()
    if not value:
        return False
    lowered = value.lower()
    if "example.com" in lowered or "image_placeholder" in lowered:
        return False
    return lowered.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg'))


def _append_image_url(images, seen, url: str) -> None:
    value = (url or "").strip()
    if _is_valid_image_url(value) and value not in seen:
        seen.add(value)
        images.append(value)


def _extract_images_from_text(text: str, images, seen, label: str = "") -> None:
    if not text:
        return
    for img_url in MD_IMAGE_PATTERN.findall(text):
        logger.debug(f"{label}发现 Markdown 图片: {img_url}")
        _append_image_url(images, seen, img_url)
    for img_url in RAW_IMAGE_URL_PATTERN.findall(text):
        logger.debug(f"{label}发现图片 URL: {img_url}")
        _append_image_url(images, seen, img_url)


def _extract_image_candidates_from_text(text: str):
    candidates = []
    if not text:
        return candidates

    markdown_spans = []
    for match in MD_IMAGE_CANDIDATE_PATTERN.finditer(text):
        alt_text = match.group(1) or ""
        url = match.group(2) or ""
        start, end = match.span()
        markdown_spans.append((start, end))
        context = text[max(0, start - 220):start]
        candidates.append({
            "url": url,
            "context": f"{alt_text}\n{context}",
        })

    for match in RAW_IMAGE_URL_PATTERN.finditer(text):
        start, end = match.span()
        if any(span_start <= start and end <= span_end for span_start, span_end in markdown_spans):
            continue
        context = text[max(0, start - 220):start]
        candidates.append({
            "url": match.group(0),
            "context": context,
        })

    return candidates


def _unique_values(values):
    seen = set()
    result = []
    for value in values or []:
        value = str(value or "").strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _query_keywords(state: QueryGraphState):
    text = " ".join(
        str(value or "")
        for value in [
            state.get("rewritten_query"),
            state.get("original_query"),
        ]
    )
    keywords = []
    seen = set()
    def add_token(value: str) -> None:
        value = value.strip()
        lowered = value.lower()
        if not value or value in QUERY_STOP_WORDS or lowered in seen:
            return
        seen.add(lowered)
        keywords.append(value)

    for token in QUERY_TOKEN_PATTERN.findall(text):
        token = token.strip()
        add_token(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", token):
            for part in re.split(r"的|是|怎么|如何|什么|哪些|哪里|吗|呢|请问|一下|怎样|相关|有关", token):
                if len(part) >= 2:
                    add_token(part)
    return keywords


def _keyword_score(text: str, keywords) -> int:
    if not text or not keywords:
        return 0
    lowered_text = text.lower()
    score = 0
    for keyword in keywords:
        lowered_keyword = keyword.lower()
        if lowered_keyword in lowered_text:
            score += 2 if len(keyword) >= 4 else 1
    return score

def step_1_check_answer(state) -> bool:
  """
  阶段一：检查 state 中是否已有 answer。
  - 若已存在：按需推送流式 delta（用于 SSE），并返回 True
  - 若不存在：返回 False
  """
  answer = state.get("answer", None)
  is_stream = state.get("is_stream" )
  if answer:
    if is_stream:
      logger.info("---Step 1: 发现已有答案，执行流式推送---")
      push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
    else:
      set_task_result(state["session_id"], "answer", answer)
    return True
  else:
    return False

# 目标结构
# HAK 180 烫金机的操作面板位于机器正前方。开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
# 具体的按键位置请参考下图：
# 【图片】
# http://local-server/images/panel_view.jpg
# http://local-server/images/button_detail.jpg
def step_2_construct_prompt(state: QueryGraphState) -> str:
  """
  第一阶段：构建 Prompt
  根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组织prompt
  """
  # 1. 获取相关信息
  original_query = state.get("original_query", "")
  rewritten_query = state.get("rewritten_query", "")
  # 优先使用重写后的问题
  question = rewritten_query if rewritten_query else original_query
  history = state.get("history", [])
  item_names = state.get("item_names", [])
  reranked_docs = state.get("reranked_docs") or []

  # 2 从重排内容中，提取为资料字符串，不可超过限额
  # 优先使用结构化 reranked_docs（包含 source/chunk_id/url/score），便于约束与引用
  # ---------------------------------------------------------
  # 逻辑解释：
  # 1. 遍历重排序后的文档列表 (reranked_docs)，这些文档已经按相关性从高到低排序。
  # 2. 对每个文档提取关键信息 (text, source, chunk_id, url, title, score)。
  # 3. 构造 "元数据头 + 正文" 格式的字符串，例如：
  #    "[1] [local] [chunk_id=123] [score=0.95] [title=操作手册]
  #     这里是文档的正文内容..."
  # 4. 累加字符长度，如果超过 MAX_CONTEXT_CHARS (如 12000 字符)，则停止添加，
  #    确保 Prompt 长度在 LLM 的处理范围内，避免 Token 溢出。
  # ---------------------------------------------------------
  docs = []
  used = 0
  for i, doc in enumerate(reranked_docs, start=1):
    text = (doc.get("text") or "").strip()
    if not text:
      continue
    source = doc.get("source") or ""
    chunk_id = doc.get("chunk_id")
    url = (doc.get("url") or "").strip()
    title = (doc.get("title") or "").strip()
    score = doc.get("score")

    meta_parts = [f"[{i}]"]
    if source:
      meta_parts.append(f"[{source}]")
    if chunk_id:
      meta_parts.append(f"[chunk_id={chunk_id}]")
    if url:
      meta_parts.append(f"[url={url}]")
    if score is not None:
      # 保留四位小数
      meta_parts.append(f"[score={float(score):.4f}]")
    if title:
      meta_parts.append(f"[title={title}]")
    doc = " ".join(meta_parts) + "\n" + text
    if used + len(doc) > MAX_CONTEXT_CHARS:
      break
    docs.append(doc)
    # 计算使用长度！ + 2 两个\n\n
    used += len(doc) + 2
  context_str = "\n\n".join(docs) if docs else "无参考内容"


  # 3. 格式化 History (历史对话)
  # ---------------------------------------------------------
  # 逻辑解释：
  # 1. 遍历历史对话记录 (history)。
  # 2. 将每轮对话格式化为 "用户: ... \n 助手: ..." 的文本块。
  # 3. 同样进行长度累加判断 (used)，确保历史记录+参考文档的总长度不超过 MAX_CONTEXT_CHARS。
  #    注意：这里的 used 变量是接着上面处理文档后的长度继续累加的，
  #    意味着如果文档占用了太多 Token，历史记录可能会被截断或完全丢弃。
  # ---------------------------------------------------------
  history_str = ""
  if history:
    for msg in history:
      # 修正：MongoDB存储格式为 {"role": "user"/"assistant", "text": "..."}
      role = msg.get("role")
      text = msg.get("text")
      if role == "user" and text:
        history_str += f"用户: {text}\n"
      elif role == "assistant" and text:
        history_str += f"助手: {text}\n"
        
      used += len(history_str) + 2
      if used > MAX_CONTEXT_CHARS:
        break
  else:
    history_str = "无历史对话"

  # 4. 格式化 Item Names (提问商品)
  item_names_str = ", ".join(item_names) if item_names else "无指定商品"

  # 5. 组装 Prompt
  prompt = load_prompt("answer_out",
    context=context_str,
    history=history_str,
    item_names=item_names_str,
    question=question
  )

  logger.info(f"组装后的提示词为：{prompt}")

  return prompt


def step_3_generate_response(state: QueryGraphState, prompt: str) -> QueryGraphState:
  """
  第二阶段：生成回答
  调用llm生成答案，支持流式输出
  """
  logger.info("---Step 3: 开始生成回答 (LLM Generation)---")
  logger.debug(f"最终Prompt内容: {prompt}")
  
  # 获取 LLM 客户端
  # 注意：这里我们使用统一的 get_llm_client 获取实例
  llm = get_llm_client()

  # 判断是否需要流式输出
  # 通常 state 中会注入 stream_queue 用于 SSE 推送
  session_id = state.get("session_id")
  is_stream = state.get("is_stream")

  if is_stream:
    logger.info(f"模式: 流式输出 (Streaming), Session: {session_id}")
    final_text = ""
    try:
      # 使用 stream 方法进行流式生成
      for chunk in llm.stream(prompt):
        delta = getattr(chunk, "content", "") or ""
        if delta:
          final_text += delta
          # 将增量内容放入队列
          push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})
      
      logger.info(f"流式输出完成，总长度: {len(final_text)}")

    except Exception as e:
      logger.error(f"流式生成出错: {e}", exc_info=True)
      # 发生错误时，尝试推送到前端
      push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
      
    state["answer"] = _clean_generated_answer(final_text)
  else:
    # 非流式直接调用
    logger.info(f"模式: 非流式输出 (Blocking), Session: {session_id}")
    try:
      response = llm.invoke(prompt)
      content = response.content
      content = _clean_generated_answer(content)
      state["answer"] = content
      set_task_result(session_id, "answer", content)
      logger.info(f"生成回答完成，长度: {len(content)}")
    except Exception as e:
      logger.error(f"生成回答出错: {e}", exc_info=True)
      state["answer"] = "抱歉，生成回答时出现错误。"

  return state


def _extract_images_from_docs(docs, state: QueryGraphState | None = None, max_docs: int = 3, max_images: int = 3):
    """
    辅助方法：从文档列表中提取图片URL
    
    核心逻辑：
    1. 遍历所有相关文档（包括本地知识库切片和联网搜索结果）。
    2. 策略一：直接检查文档的 'url' 字段（常见于联网搜索结果）。
       - 验证后缀名是否为图片格式 (.jpg, .png 等)。
    3. 策略二：使用正则表达式扫描文档 'text' 正文内容（常见于本地 Markdown 文档）。
       - 匹配 Markdown 图片语法: ![alt text](image_url)。
    4. 对提取到的 URL 进行去重处理，返回唯一图片列表。
    
    :param docs: 文档列表，每个文档为字典格式
    :return: 图片 URL 字符串列表
    """
    images = []
    seen = set() # 用于去重，避免同一张图片重复出现
    if not docs:
        return []
    # ---------------------------------------------------------
    # 正则表达式解释：r'!\[.*?\]\((.*?)\)'
    # 1. !\[   -> 匹配 Markdown 图片语法的开头 "![" (注意 [ 需要转义)
    # 2. .*?   -> 非贪婪匹配图片描述文本 (Alt Text)，即 [] 中间的内容
    # 3. \]    -> 匹配描述文本的结束符 "]"
    # 4. \(    -> 匹配 URL 部分的开始符 "("
    # 5. (.*?) -> 捕获组 (Group 1)：非贪婪匹配括号内的实际 URL 内容
    # 6. \)    -> 匹配 URL 部分的结束符 ")"
    # ( ... ) （不带反斜杠）：这就是 捕获组 。
    # 它的作用是告诉程序：“虽然我匹配了整个 ![...](...) 结构，但我 只要 这括号里的内容”。
    # ---------------------------------------------------------
    logger.info(f"开始提取图片，待处理文档数: {len(docs)}")
    keywords = _query_keywords(state or {}) if state else []
    candidates = []

    for i, doc in enumerate(docs[:max_docs]):
        # 1. 优先检查 url 字段 (主要针对 Web Search 结果)
        url = (doc.get("url") or "").strip()
        if url and _is_valid_image_url(url):
            logger.debug(f"文档[{i}] 发现图片 URL (字段): {url}")
            _append_image_url(images, seen, url)

        # 2. 检查 text 字段中的 Markdown 图片 (主要针对 Local Chunk)
        text = (doc.get("text") or "").strip()
        if keywords:
            doc_context = " ".join([
                str(doc.get("title") or ""),
                str(doc.get("parent_title") or ""),
                str(doc.get("file_title") or ""),
            ])
            for candidate in _extract_image_candidates_from_text(text):
                score = _keyword_score(f"{doc_context}\n{candidate['context']}", keywords)
                if score > 0 and _is_valid_image_url(candidate["url"]):
                    candidates.append({
                        "url": candidate["url"],
                        "score": score,
                    })
        else:
            _extract_images_from_text(text, images, seen, label=f"文档[{i}] 正文")
        if len(images) >= max_images:
            break

    if keywords and candidates:
        candidates.sort(key=lambda item: item["score"], reverse=True)
        for candidate in candidates:
            _append_image_url(images, seen, candidate["url"])
            if len(images) >= max_images:
                break

    images = images[:max_images]
    logger.info(f"图片提取完成，共找到 {len(images)} 张唯一图片: {images}")
    return images


def _extract_related_images_from_milvus(state: QueryGraphState, max_images: int = 3):
  """
  当本轮重排片段没有直接命中图片时，只在同一章节内保守查找图片。
  不再按整篇文档或商品名盲取，避免封面、图标、验证码等无关图片反复出现。
  """
  docs = state.get("reranked_docs") or []
  file_titles = _unique_values([doc.get("file_title") for doc in docs])
  parent_titles = _unique_values([
      doc.get("parent_title") or doc.get("title")
      for doc in docs[:3]
  ])
  keywords = _query_keywords(state)

  filters = []
  for file_title in file_titles:
    safe_file = escape_milvus_string(file_title)
    for parent_title in parent_titles[:3]:
      safe_parent = escape_milvus_string(parent_title)
      filters.append(f'file_title == "{safe_file}" and parent_title == "{safe_parent}"')

  filters = _unique_values(filters)
  if not filters or not keywords:
    return []

  candidates = []
  seen = set()
  try:
    client = get_milvus_client()
    collection_name = milvus_config.chunks_collection
    if not client or not collection_name or not client.has_collection(collection_name):
      return []

    for filter_expr in filters:
      rows = client.query(
        collection_name=collection_name,
        filter=filter_expr,
        output_fields=["content", "title", "parent_title", "file_title", "item_name"],
        limit=120,
      )
      for row in rows or []:
        content = row.get("content") or ""
        if not content:
          continue
        row_text = " ".join([
            str(row.get("title") or ""),
            str(row.get("parent_title") or ""),
            content,
        ])
        score = _keyword_score(row_text, keywords)
        if score <= 0:
          continue

        row_images = []
        _extract_images_from_text(content, row_images, set(), label="Milvus同章节候选片段")
        for url in row_images:
          if url in seen:
            continue
          seen.add(url)
          candidates.append({"url": url, "score": score})
  except Exception as exc:
    logger.warning(f"Milvus兜底图片提取失败：{exc}")

  candidates.sort(key=lambda item: item["score"], reverse=True)
  images = [item["url"] for item in candidates[:max_images]]
  logger.info(f"Milvus同章节兜底图片提取完成，共找到 {len(images)} 张图片，关键词={keywords}")
  return images


def step_4_write_history(state: QueryGraphState, image_urls = None) -> QueryGraphState:
  """
  阶段四：把本轮答案写入 MongoDB history。
  利用 utils/mongo_history_utils.py 中的 save_chat_messages 方法。
  """
  session_id = state.get("session_id", "default")
  answer = (state.get("answer") or "").strip()
  item_names = state.get("item_names") or []

  try:
    if answer:
       save_chat_message(
        session_id=session_id,
        role="assistant",
        text=answer,
        rewritten_query="",
        item_names=item_names,
        image_urls=image_urls,
        message_id=None
      )
  except Exception as e:
    # 写历史失败不应影响主链路
    logger.error(f"写入Mongo历史记录失败: {e}")

  return state


def node_answer_output(state: QueryGraphState) -> QueryGraphState:
  """
  1 判断state 中的answer是否已经存在，如果存在直接输出answer中的答案，注意判断是否需要流式输出需要则流式输出
  2 根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组织prompt 并调用llm 生成答案
  3 阶段三：调用大模型输出答案 注意判断是否需要流式输出需要则流式输出
  4 把答案写入到mongodb的history中 利用utils/mongo_history_utils.py中的save_chat_message方法
  5 做最后一次push操作（主要是为了触发前端图片渲染)
     {
        "answer": "HAK 180 烫金机的操作面板位于...（大模型生成的纯文本）...",
        "status": "completed",
        "image_urls": [
            "http://local-server/images/panel_view.jpg",
            "http://local-server/images/button_detail.jpg"
        ]
      }
  """
  logger.info("---node_answer_output (答案生成) 节点开始处理---")
  add_running_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
  
  # 阶段一：检查answer是否存在,如果存在直接输出answer中的答案
  answer_exists = step_1_check_answer(state)
  
  # 阶段二  如果没有answer则 构建 Prompt
  if not answer_exists:
    prompt = step_2_construct_prompt(state)
    state["prompt"] = prompt

    # 阶段三：  如果没有answer则 调用大模型输出答案
    step_3_generate_response(state, prompt)

  # 提取图片URL（用于历史记录和前端展示）
  image_urls = _extract_images_from_docs(state.get("reranked_docs") or [], state=state)
  if not image_urls:
    image_urls = _extract_related_images_from_milvus(state)

  answer_image_urls = []
  answer_seen = set()
  answer_text = state.get("answer") or ""
  if answer_text:
    for url in MD_IMAGE_PATTERN.findall(answer_text):
      _append_image_url(answer_image_urls, answer_seen, url)
    for url in RAW_IMAGE_URL_PATTERN.findall(answer_text):
      _append_image_url(answer_image_urls, answer_seen, url)
  if answer_image_urls:
    merged = []
    seen = set()
    for url in image_urls + answer_image_urls:
      _append_image_url(merged, seen, url)
    image_urls = merged

  if not image_urls:
    state["answer"] = _clean_empty_image_hint(state.get("answer") or "")

  # 阶段四：把答案写入到mongodb的history中
  if state.get("answer"):
    logger.info("---写入MongoDB历史记录---")
    step_4_write_history(state, image_urls=image_urls)

  add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
  
  # 阶段五: 流式输出结束，发送 final 事件 [最后兜底，确保图片都能争取渲染和结束]
  logger.info(f"---发送 final 事件---图片为：{image_urls}")
  if state.get("is_stream"):
    push_to_session(
        state['session_id'],
        SSEEvent.FINAL,
        {
            "answer": state["answer"],
            "status": "completed",
            "image_urls": image_urls  # 发送图片URL给前端
        }
    )
  
  logger.info("---node_answer_output 节点处理结束---")
  return state


if __name__ == "__main__":
    print("\n" + "="*50)
    print(">>> 启动 node_answer_output 本地测试")
    print("="*50)
    
    # 1. 构造模拟数据
    # 模拟重排序后的文档列表 (reranked_docs)
    # 包含：本地文档（带Markdown图片）、联网结果（带URL字段）、纯文本文档
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "source": "local",
            "title": "HAK 180 烫金机操作手册_v2.pdf",
            "score": 0.95,
            "text": """
            HAK 180 烫金机的操作面板位于机器正前方。
            开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
            具体的操作面板布局请参考下图：
            ![操作面板布局图](http://local-server/images/panel_view.jpg)
            
            如果是进行局部烫金，请调节侧面的旋钮。
            ![侧面旋钮细节](http://local-server/images/knob_detail.png)
            """
        },
        {
            "chunk_id": None,
            "source": "web",
            "title": "HAK 180 常见故障排除 - 官网",
            "score": 0.88,
            "url": "http://example.com/hak180_troubleshooting.jpeg", # 这是一个直接指向图片的URL（虽然少见，但用于测试提取）
            "text": "如果机器无法加热，请检查保险丝是否熔断..."
        },
        {
            "chunk_id": "local_102",
            "source": "local",
            "title": "安全注意事项",
            "score": 0.82,
            "text": "操作时请务必佩戴隔热手套，避免高温烫伤。"
        }
    ]

    # 模拟历史记录
    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？"},
        {"role": "assistant", "text": "您好！请问您具体指的是哪一款机器？"},
        {"role": "user", "text": "HAK 180 烫金机"}
    ]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK 180 烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False, # 测试非流式
        # "is_stream": True, # 若要测试流式，需确保 SSE 环境或 mock 相关函数
        "answer": None # 初始无答案
    }

    try:
        # 运行节点
        result = node_answer_output(mock_state)
        
        print("\n" + "="*50)
        print(">>> 测试结果摘要:")
        
        # 1. 验证 Prompt 构建
        if "prompt" in result:
            print(f"[PASS] Prompt 构建成功 (长度: {len(result['prompt'])})")
            # print(f"Prompt 预览:\n{result['prompt'][:200]}...")
        else:
            print("[FAIL] Prompt 未构建")

        # 2. 验证答案生成
        answer = result.get("answer")
        if answer and len(answer) > 10:
            print(f"[PASS] 答案生成成功 (长度: {len(answer)})")
            print(f"答案预览: {answer[:50]}...")
        else:
            print(f"[WARN] 答案生成可能异常 (Content: {answer})")

        # 3. 验证图片提取
        # 我们期望提取到 3 张图片：
        # 1. http://local-server/images/panel_view.jpg (来自 local_101)
        # 2. http://local-server/images/knob_detail.png (来自 local_101)
        # 3. http://example.com/hak180_troubleshooting.jpeg (来自 web 结果的 url 字段)
        
        # 注意：这里我们没办法直接从 result state 里拿到 image_urls，因为它是作为 SSE 推送出去的，或者存库了
        # 但我们可以通过日志观察 _extract_images_from_docs 的输出
        # 如果需要验证，可以临时修改 node_answer_output 返回 image_urls
        print("\n[INFO] 请检查上方日志中是否包含 '图片提取完成' 及以下 URL:")
        print(" - http://local-server/images/panel_view.jpg")
        print(" - http://local-server/images/knob_detail.png")
        print(" - http://example.com/hak180_troubleshooting.jpeg")

        print("="*50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
