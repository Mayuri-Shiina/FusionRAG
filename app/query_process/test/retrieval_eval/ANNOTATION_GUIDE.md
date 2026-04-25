# 当前知识库评测标注指南

## 1. 这套题库的目的

这不是通用 benchmark，而是针对当前仓库真实知识库 `doc/` 以及其导入产物 `output/20260420/*/chunks.json` 整理的一套小型离线评测集。

它主要解决两个问题：

1. 简历里“Top-10 召回率提升 35%”需要一套可解释的评测方法
2. 面试官继续追问“怎么标、怎么评、改造前后差异是什么”时，需要拿得出具体样例

## 2. 为什么先做 evidence-level 标注

当前仓库里真实知识源是 `doc/` 下的 PDF，而已导入的结构化切片在 `output/20260420/*/chunks.json`。

这些 `chunks.json` 中暂时没有 Milvus 主键，因此当前版本使用：

- `output_dir#chunk_index`

作为稳定的 `chunk_id` 代理标识。

所以先做两层标注：

1. `question -> evidence`
   问题对应哪段原始知识文本

2. `evidence -> chunk_id`
   当前先映射到 `output_dir#chunk_index`；如果后续你要对接 Milvus 真正主键，再做二次映射

这样做的好处：

- 标注门槛低
- 可读性强
- 面试时能直接展示原始证据出处
- 后续迁移到真实检索评测也不需要重做题库

## 3. 推荐标注步骤

1. 先阅读 [current_kb_questions.csv](./current_kb_questions.csv)，确认问题是否像真实用户会问的话
2. 逐条检查 [current_kb_gold_evidence.csv](./current_kb_gold_evidence.csv) 中的文件位置和答案摘要
3. 如果某条证据不足以支撑问题，可以为同一个 `qid` 再补一条 evidence 行
4. 当导入链路实际切片后，在切片备份 JSON 或 Milvus 中查找对应文本
5. 把找到的 `chunk_id` 回填到 `current_kb_gold_evidence.csv`
6. 运行 `materialize_chunk_gold.py` 生成最终 `gold_chunks.csv`

## 4. 如何判断一条 evidence 算不算 gold

满足以下任一类场景时，通常可以算 gold：

- 直接给出问题的核心结论
- 直接给出规则阈值、步骤、配置、原因
- 作为最终回答的主证据，而不是仅仅提到相关话题

不建议算 gold 的情况：

- 只是同主题背景介绍，没有关键结论
- 只提到术语，没有回答问题
- 需要依赖太多外部上下文，单独看不足以作证

## 5. 一问多证据怎么处理

允许同一个 `qid` 对应多条 evidence。

推荐做法：

- 1 条主证据
- 必要时补 1 条辅助证据

最终换算为 `Recall@10` 时，Top-10 命中任意一条 gold chunk 即算该问题命中。

## 6. 题型分层建议

这套题库已经混合了几类问题：

- `architecture`：问整体设计
- `import`：问导入和切片
- `multimodal`：问图片增强和对象存储
- `rewrite`：问 query rewrite 和实体对齐
- `retrieval`：问 embedding / HyDE / RRF
- `rerank`：问重排和动态 TopK
- `multi_turn`：模拟真实对话里的省略式追问

后续若你要扩题，建议保持这个分布，不要全部变成“定义题”。

## 7. 怎么把它说成面试答案

你可以这样表述：

“我先基于现有知识库自己构了一套离线问题集，不是直接拿最终答案做判断，而是给每个问题标注主证据位置。早期先做 evidence-level 标注，等导入切片落到 Milvus 后，再映射成 chunk-level gold。最终 Recall@10 的定义是：Top-10 结果中命中任意一条 gold chunk，就算该问题召回成功。” 
