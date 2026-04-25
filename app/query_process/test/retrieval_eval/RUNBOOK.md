# 检索评测执行 Runbook

## 1. 目标

把当前题库真正转成一句可复述、可追问的评测结论，例如：

`在自建离线评测集上，增强版检索链路的 Recall@10 相比单路混合检索基线提升约 35%。`

## 2. 实验对象

建议至少保留两组：

### A. 基线组 `baseline_rewrite`

- 保留 `item_name_confirm`
- 使用 `rewritten_query + item_name filter`
- 只跑单路 `node_search_embedding`
- 不使用 HyDE
- 不做 RRF 融合

这组最适合当作简历里的“改造前”。

### B. 增强组 `final_rrf`

- 保留 `item_name_confirm`
- 使用 `rewritten_query + item_name filter`
- 并发跑 `node_search_embedding`
- 并发跑 `node_search_embedding_hyde`
- 用 `node_rrf` 做融合，取 Top-10

这组最适合当作“改造后”。

## 3. 为什么不把 rerank 算进 Recall@10

`Rerank + 动态 TopK` 的主要作用是：

- 压低噪声
- 提升最终上下文纯度
- 降低无关片段进入 Prompt 的概率

它更适合解释：

- Precision
- MRR
- NDCG
- 上下文长度下降
- 幻觉率下降

如果你把它也算进 Recall@10 的提升来源，面试官继续追问时会比较难自洽。

## 4. 实际执行步骤

### Step 1：确认题库

使用：

- [current_kb_questions.csv](./current_kb_questions.csv)
- [current_kb_gold_evidence.csv](./current_kb_gold_evidence.csv)

先确认问题和证据是否合理。

### Step 2：使用当前已导入切片或继续补齐导入

当前已经发现 3 份真实知识文档的导入产物位于：

- `output/20260420/ae4bb064-06d2-4e7b-85ab-4bb501986dad/chunks.json`
- `output/20260420/1df8dfde-855b-4716-a9e5-8e4a3d8e1294/chunks.json`
- `output/20260420/5e41ddfd-8f0a-4a7f-9482-eef09af45b4b/chunks.json`

如果你继续补更多文档，再把 `doc/` 中其余 PDF 走一遍导入链路：

- `node_pdf_to_md`
- `node_md_img`
- `node_document_split`
- `node_bge_embedding`
- `node_import_milvus`

当前版本已经把 `chunk_id` 预填成 `output_dir#chunk_index`，可以直接用于离线评测。

如果后续需要和 Milvus 实际主键完全对齐，再根据 `answer_key` 搜索对应文本，把命中的真实主键回填到 `current_kb_gold_evidence.csv`。

### Step 3：导出 chunk 级 gold

运行：

```powershell
python app/query_process/test/retrieval_eval/materialize_chunk_gold.py `
  --evidence app/query_process/test/retrieval_eval/current_kb_gold_evidence.csv `
  --output app/query_process/test/retrieval_eval/gold_chunks.from_current_kb.csv
```

### Step 4：分别跑两组检索结果

为每个 `qid` 导出两份结果：

- `baseline_rewrite.jsonl`
- `final_rrf.jsonl`

格式参考 [README.md](./README.md)。

每条至少要有：

```json
{"qid": "QKB001", "retrieved_chunk_ids": ["123", "456", "789"]}
```

### Step 5：计算指标

运行：

```powershell
python app/query_process/test/retrieval_eval/score_retrieval_eval.py `
  --questions app/query_process/test/retrieval_eval/current_kb_questions.csv `
  --gold app/query_process/test/retrieval_eval/gold_chunks.from_current_kb.csv `
  --run baseline=path/to/baseline_rewrite.jsonl `
  --run enhanced=path/to/final_rrf.jsonl `
  --compare baseline enhanced
```

## 5. 指标口径

### Recall@10

只要 Top-10 结果中命中任意一条 gold chunk，该问题就算命中。

### 相对提升

如果：

- 基线 Recall@10 = 0.52
- 增强 Recall@10 = 0.70

那么：

`(0.70 - 0.52) / 0.52 = 34.6%`

面试时应明确说：

`这里说的是相对提升约 35%，不是绝对提升 35 个百分点。`

## 6. 推荐结果展示

最小可展示表格：

| Run | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| baseline_rewrite |  |  |  |  |  |
| final_rrf |  |  |  |  |  |

如果要更强一点，再加一列：

- improved_qids 数量

也就是增强版新增命中的问题数。

## 7. 面试时的说法

推荐说法：

“我做的是 retrieval offline eval，不是直接拿最终答案做准确率。先基于现有知识库整理问题集，再给每个问题标主证据位置，最后映射到 chunk-level gold。基线是 rewrite 后的单路混合检索，增强版是在此基础上加入 HyDE 和 RRF 融合。Recall@10 的定义是 Top-10 里命中任意一条 gold chunk 就算成功，最终增强版相对基线提升约 35%。” 
