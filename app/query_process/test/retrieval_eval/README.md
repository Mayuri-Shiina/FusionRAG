# 检索离线评测说明

这套评测只衡量 `retrieval`，不直接评估最终答案生成质量。

目标是把简历中的这句话落成一套可执行流程：

`在自建离线评测集上，Top-10 召回率较单路混合检索基线提升约 35%。`

## 1. 指标定义

### Recall@K

对每个问题，先人工标注 1 到 N 个 `gold chunk_id`。

如果系统返回的 Top-K 结果中，命中了任意一个 `gold chunk_id`，则该问题记为命中。

公式：

`Recall@K = Top-K 命中的问题数 / 全部问题数`

这里的 Recall@10 更接近问答场景中的“证据是否被找回来”，适合 RAG 检索阶段。

### MRR

用于补充观察正确证据排在前面的能力，不作为简历主指标。

## 2. 标注原则

每个问题至少标注 1 个 `gold chunk_id`，建议最多 3 个。

判定相关的标准：

- 该 chunk 包含回答问题所需的核心事实
- 该 chunk 可以作为最终回答的直接证据
- 只提到主题但没有关键事实，不算 gold

如果答案依赖连续上下文，可以标：

- 1 个主证据 chunk
- 1 到 2 个补充 chunk

Recall@10 的统计口径仍然是“Top-10 命中任意一个 gold chunk 即算成功”。

## 3. 推荐测试集规模

建议先构建一个 `60 到 100` 条问题的小型离线集。

推荐分布：

- 精确问法：参数、价格、规格、步骤
- 口语化问法：省略主语、代词指代
- 同义改写：知识库术语和用户口语表达不一致
- 多轮追问：依赖最近 1 到 2 轮上下文
- 长尾问法：文档深处的限制条件、注意事项

## 4. 实验分组建议

为了让“提升 35%”更经得起追问，建议不要只保留最终版和基线版，而是做一个简化消融表。

推荐四组：

1. `baseline_raw`
   使用原始 query，单路 `node_search_embedding`

2. `baseline_rewrite`
   使用 `rewritten_query + item_name 对齐`，单路 `node_search_embedding`

3. `hyde_only`
   使用 `rewritten_query + item_name 对齐 + node_search_embedding_hyde`

4. `final_rrf`
   使用 `rewritten_query + item_name 对齐 + embedding + hyde + RRF`

如果你只想保留两组用于简历：

- 基线：`baseline_rewrite`
- 增强：`final_rrf`

这样说最稳，因为它与当前代码结构最贴近：

- 基线节点：[node_search_embedding.py](../../agent/nodes/node_search_embedding.py)
- 增强节点：[node_search_embedding_hyde.py](../../agent/nodes/node_search_embedding_hyde.py)
- 融合节点：[node_rrf.py](../../agent/nodes/node_rrf.py)

`node_rerank.py` 更适合解释排序质量和上下文降噪，不建议把它算进 Recall@10 提升来源。

## 5. 数据文件说明

### questions.csv

问题集主表，每行一个问题。

字段：

- `qid`：问题唯一编号
- `query`：用户原始问题
- `item_name`：期望命中的商品或实体，可为空
- `question_type`：问题类型，如 `precise` / `colloquial` / `multi_turn`
- `difficulty`：难度，如 `easy` / `medium` / `hard`
- `source_doc`：问题主要来源文档
- `notes`：补充说明

### gold_chunks.csv

gold 标注表，一个问题可以对应多条 gold chunk。

字段：

- `qid`：问题编号
- `chunk_id`：人工标注的 gold chunk
- `relevance`：建议固定写 `1`
- `reason`：为什么这条 chunk 是证据

### current_kb_questions.csv

基于当前仓库内真实知识库 `doc/` 及其导入产物 `output/20260420/*/chunks.json` 整理出的第一版真实问题集。

特点：

- 不再是占位模板，而是直接来自当前知识库内容
- 问法尽量贴近真实检索输入，而不是纯教科书式问答
- 包含少量 `history` / `expected_rewrite` 字段，方便后续评估 query rewrite

### current_kb_gold_evidence.csv

这是当前更适合人工标注和面试复述的一版 `evidence-level gold`。

它不强依赖现成的 `chunk_id`，而是先把问题绑定到：

- `source_file`
- `line_start`
- `line_end`
- `answer_key`

也就是先做“证据级标注”，再映射到真正的 `chunk_id`。

推荐工作流：

1. 先用 `current_kb_gold_evidence.csv` 确认问题和证据是否合理
2. 文档导入并切片后，把每条证据映射到真实 `chunk_id`
3. 用 `materialize_chunk_gold.py` 生成最终用于打分的 `gold_chunks.csv`

## 6. 从证据标注到 chunk 标注

对于当前仓库，最现实的做法不是一开始就人工写 `chunk_id`，而是分两步：

1. `evidence-level`
   先标文件位置和证据语义

2. `chunk-level`
   等文档经过导入链路切片、入库后，再把证据映射到实际 `chunk_id`

映射建议：

- 优先以 `answer_key` 中的核心句作为检索锚点
- 在切片备份 JSON 或 Milvus 中搜索对应文本
- 找到命中的 `chunk_id` 后，回填到 `current_kb_gold_evidence.csv` 的 `chunk_id` 列

完成回填后，可执行：

```powershell
python app/query_process/test/retrieval_eval/materialize_chunk_gold.py `
  --evidence app/query_process/test/retrieval_eval/current_kb_gold_evidence.csv `
  --output app/query_process/test/retrieval_eval/gold_chunks.from_current_kb.csv
```

### runs/*.jsonl

每个检索方案导出一个运行结果文件，格式为 `jsonl`。

每行一个问题，最少要包含：

```json
{"qid": "Q001", "retrieved_chunk_ids": ["101", "205", "309"]}
```

也支持更详细格式：

```json
{
  "qid": "Q001",
  "results": [
    {"chunk_id": "101", "score": 0.93},
    {"chunk_id": "205", "score": 0.84},
    {"chunk_id": "309", "score": 0.79}
  ]
}
```

## 7. 统计脚本用法

示例：

```powershell
python app/query_process/test/retrieval_eval/score_retrieval_eval.py `
  --questions app/query_process/test/retrieval_eval/questions.csv `
  --gold app/query_process/test/retrieval_eval/gold_chunks.csv `
  --run baseline=app/query_process/test/retrieval_eval/runs/baseline_predictions.sample.jsonl `
  --run enhanced=app/query_process/test/retrieval_eval/runs/enhanced_predictions.sample.jsonl `
  --compare baseline enhanced
```

脚本会输出：

- 每个 run 的 `Recall@1/3/5/10`
- `MRR`
- 若指定 `--compare`，额外输出两组之间的命中差异和相对提升

## 8. 当前知识库题库

当前已经补了一套真实题库：

- [current_kb_questions.csv](./current_kb_questions.csv)
- [current_kb_gold_evidence.csv](./current_kb_gold_evidence.csv)
- [ANNOTATION_GUIDE.md](./ANNOTATION_GUIDE.md)

当前版本直接覆盖 3 份已导入完成的真实 PDF：

- `深信服应用交付虚拟化产品 vAD技术白皮书`
- `H3C 云安全运维管理平台 安装指导(E1306)-5W100-整本手册`
- `H3C S1020V虚拟交换机技术白皮书-5W100-整本手册`

这套题库覆盖：

- 产品定义与功能
- 部署与验证
- 典型场景
- 参数与兼容性
- 工作流程与架构
- 性能优化与支持特性

## 9. 面试可复述口径

你可以这样讲：

“我做的是离线 retrieval eval，不直接看最终生成答案。先人工整理一批测试问题，每个问题标 1 到 3 个 gold chunks。只要系统返回的 Top-10 里命中任意 gold chunk，就记为 Recall@10 命中。基线是单路混合检索，增强版是 rewrite 后的 query 结合 HyDE 和 RRF 融合。最终增强版在自建评测集上的 Recall@10 相比基线提升约 35%。动态 TopK 和 rerank 主要用于降噪，不算在 recall 提升来源里。” 
