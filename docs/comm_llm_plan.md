# 通信标准证据库 LLM 对齐项目：绝对详细执行方案

## 0. 项目定义（冻结，不在执行中反复改）

### 0.1 目标（必须达成）
- 以公开通信标准（3GPP/ETSI/IETF）构建证据库。
- 在同一基座模型上完成：Base → SFT(LoRA/QLoRA) → DPO(Preference) 三版本。
- 输出可量化对比：Base vs SFT vs DPO 的核心指标。
- 输出可演示 Demo：输入“问题+证据”，输出“严格 JSON + 引用高亮 + 拒答”。
- 代码可复现：从下载文档到训练评测 Demo 的命令链条可跑通。

### 0.2 交付物清单（检查表）
- [ ] data/raw/manifest.json（文档URL+sha256；不提交PDF）
- [ ] data/corpus/evidence_corpus.jsonl（证据库，chunk级）
- [ ] data/eval/eval_set.jsonl（固定证据评测集）
- [ ] data/sft/train.jsonl + data/sft/eval.jsonl（SFT数据）
- [ ] outputs/sft_adapter/（SFT LoRA权重）
- [ ] data/dpo/train.jsonl + data/dpo/eval.jsonl（DPO偏好对）
- [ ] outputs/dpo_adapter/（DPO LoRA权重）
- [ ] reports/eval_base.json + eval_sft.json + eval_dpo.json（逐样本评测结果）
- [ ] reports/compare.md（指标表+case对比）
- [ ] src/08_demo_gradio.py（可运行Demo）
- [ ] docs/comm_llm_plan.md（本文件，作为执行准绳）

### 0.3 固定技术选择（禁止扩散）
- 模型：Qwen2.5-1.5B-Instruct（默认）；若显存充足可替换 3B，但全流程以 1.5B 为基准。
- 训练：优先 QLoRA 4bit；不可用时退回 LoRA bf16/fp16。
- 评测：不做检索评测；评测集固定提供 evidence_chunk_ids，减少变量。
- 输出格式：强制 JSON schema（见 0.4）。

### 0.4 统一输出 JSON Schema（写死）
- 所有训练/评测/Demo 输出必须满足：
```json
{
  "answer": "string",
  "citations": [
    {"doc_id": "string", "section": "string", "chunk_id": "string", "quote": "string"}
  ],
  "confidence": "low|mid|high",
  "cannot_answer_reason": "string|null"
}
```
规则：
- 可回答：cannot_answer_reason=null 且 citations 至少 1 条。
- 不可回答：cannot_answer_reason 必填；answer 允许空字符串或最小解释；citations 可为空。
- quote 必须是 evidence 原文子串（用于精确匹配与DPO打分）。

### 0.5 统一 Prompt 模板（写死）
System（固定，不随任务变）：
- 角色：通信标准问答助手。
- 强约束：只能依据给定证据回答；证据不足必须拒答；必须输出合法 JSON；不得输出多余文本。

User 输入格式（固定）：
```text
问题: {question}
证据:
[chunk_id={...} doc_id={...} section={...}]
{text}
...
```

Assistant 输出：严格 JSON（不包裹 ```）。

---

## 1. 目录结构（在仓库内落地）

在仓库根目录创建（或以现有结构对齐）以下路径：
```text
comm_project/
  data/
    raw/                     # PDF 原件下载位置(本地，gitignore)
    corpus/                  # evidence_corpus.jsonl
    sft/                     # train.jsonl / eval.jsonl
    dpo/                     # train.jsonl / eval.jsonl
    eval/                    # eval_set.jsonl
  src/
    00_download.py           # 标准文档下载脚本
    01_parse_pdf.py          # PDF ➜ 段落 JSONL
    02_build_corpus.py       # chunk 切分 + 清洗
    03_make_eval.py          # 评测集构建（固定证据）
    04_make_sft.py           # SFT 样本生成
    05_train_sft.py          # SFT LoRA/QLoRA 训练
    06_make_dpo.py           # 偏好对生成 + 规则打分
    07_train_dpo.py          # DPO 训练
    08_evaluate.py           # 评测指标与报表
    09_demo_gradio.py        # 推理服务
    utils/
      prompts.py             # prompt+schema 统一定义
      pdf_clean.py           # PDF 清洗规则
      chunking.py            # chunk 切分
      scoring.py             # JSON 校验/引用匹配/覆盖率
      io.py                  # jsonl读写
  outputs/
    sft_adapter/
    dpo_adapter/
    logs/
  reports/
    eval_base.json
    eval_sft.json
    eval_dpo.json
    compare.md
```
注意：仓库已有训练脚本（tianji/finetune/...）可复用；但 comm_project 作为“通信方向”独立目录，避免与原项目礼仪数据混杂。

---

## 2. 数据源范围（严格限制，避免爆炸）

### 2.1 首轮只选 3–6 份文档
建议清单（优先级从高到低）：
- 3GPP TS 23.501（System Architecture）
- 3GPP TS 23.502（Procedures）
- 3GPP TS 24.501（NAS Protocol）
- 可选：3GPP TS 33.501（Security Architecture）
- 可选：ETSI NFV 概述/架构文档 1 份
- 可选：IETF RFC 1 份（QUIC/TLS/HTTP3）

### 2.2 合规与提交策略（强制执行）
- 不提交 PDF 原件到 git。
- 提交下载脚本、URL 清单与 sha256 manifest。
- Demo/训练本地从 data/raw 读取；CI 不跑下载。

---

## 3. Step 1：下载与 Manifest（Day1）

### 3.1 产出
- data/raw/*.pdf（本地）
- data/raw/manifest.json（可提交）

### 3.2 具体操作
1. 在 src/00_download.py 内定义 `DOCS = [{doc_id, url, filename}]`。
2. 下载到 data/raw/filename。
3. 计算 sha256。
4. manifest.json 记录：doc_id, url, filename, sha256, downloaded_at。

### 3.3 验收标准
- manifest.json 中每个条目都能在 data/raw 找到对应文件。
- sha256 非空。

---

## 4. Step 2：PDF 解析与清洗（Day2）

### 4.1 中间产物
- data/corpus/parsed_paragraphs.jsonl（每行一个段落单元）

段落单元格式：
```json
{"doc_id":"TS 23.501","page":12,"section":"6.2.1","title":"Registration Management","text":"..."}
```

### 4.2 清洗规则（写死，避免“看感觉”）
- 去页眉页脚：重复出现且短文本（<=80 chars）的行集合，按出现频率阈值剔除。
- 去页码：仅数字或“Page x of y”。
- 合并断行：同段内换行但无句号结尾，合并为空格。
- 跳过表格：包含大量连续空格/竖线/对齐符号的行段丢弃。
- 章节识别：正则匹配 `^\d+(\.\d+)+`，作为 section。

### 4.3 抽样验收（强制）
- 从 parsed_paragraphs 随机抽 200 条：
  - section 是否合理
  - text 是否可读
  - 表格噪声占比是否 < 10%
- 若不达标：先改清洗规则，再重跑，不进入下一步。

---

## 5. Step 3：构建证据库（chunk）（Day2）

### 5.1 产出
- data/corpus/evidence_corpus.jsonl

chunk 格式：
```json
{"chunk_id":"TS23501-6.2.1-0003","doc_id":"TS 23.501","section":"6.2.1","title":"...","text":"...","keywords":["..."]}
```

### 5.2 chunk 切分规则（写死）
- 以 section 为边界聚合段落。
- 在 section 内按 token/长度切分：
  - 目标长度：600–900 tokens（或 800–1500 中文字）
  - 最小长度：200 tokens（过短与相邻 chunk 合并）
  - 最大长度：1100 tokens（超长强切）
- chunk_id 规则：`{docShort}-{section}-{index:04d}`

### 5.3 关键词抽取（可选但推荐）
- 从 title + text 中提取 top-k（简单 TF-IDF 或规则词表：AMF/SMF/UPF/RRC/NAS/PDU 等）。

### 5.4 验收标准
- evidence_corpus 数量：>= 2000 chunks（少于则增加文档或放宽切分）。
- chunk 平均长度处于目标区间。

---

## 6. Step 4：评测集构建（固定证据评测，Day3）

### 6.1 产出
- data/eval/eval_set.jsonl（200–300 条）

样本格式：
```json
{
  "id":"E0001",
  "type":"answerable",
  "question":"...",
  "evidence_chunk_ids":["...","..."],
  "must_quotes":["..."],
  "notes":"用于人工核对的备注"
}
```

### 6.2 题型配比（写死）
- answerable：140–180 条
  - 定义/作用：40%
  - 条件/触发：30%
  - 流程/步骤：30%
- unanswerable：60–120 条

### 6.3 构造方法（可复现）
- answerable：从 evidence chunk 中抽取 1–2 句作为 must_quotes，然后把句子改写成问题。
- unanswerable：将问题与证据错配（从别的 doc/section 抽取 must_quotes 的关键词生成问题，但不给相关 chunk）。

### 6.4 验收标准
- unanswerable 样本中，证据确实不包含答案（两人各抽查 30 条）。

---

## 7. Step 5：SFT 数据构建（Day3–Day4）

### 7.1 产出
- data/sft/train.jsonl（建议 5k–15k）
- data/sft/eval.jsonl（建议 300–800）

### 7.2 单条训练样本格式（与现有仓库兼容）
使用 conversation 结构：
```json
{
  "conversation": [
    {
      "system": "{SYSTEM_PROMPT}",
      "input": "问题: ...\n证据:\n[chunk_id=... doc_id=... section=...]\n...",
      "output": "{JSON}"
    }
  ]
}
```

### 7.3 SFT 样本生成规则（必须可验证）
为每个 chunk 生成若干样本：
- 定义类：从包含“refers to/is defined as”句子抽取 quote；answer 重述；citations.quote=原句子子串。
- 条件类：抽取包含 when/if/shall 的句子；问题问“何时/什么条件”；answer 使用该句子信息。
- 流程类：对包含 procedure/steps 的段落，抽取连续 2–4 句作为 quote；answer 输出“步骤列表”。
- 拒答类：
  - 从 chunkA 生成问题，但证据给 chunkB（不相关）
  - output 必须 JSON 且 cannot_answer_reason 非空

### 7.4 数据质量门槛（写死）
- JSON 可解析率 >= 99%（生成时就校验）。
- citations.quote 精确匹配率 >= 95%（quote 必须是证据子串）。
- 拒答样本占比：15%–30%。

---

## 8. Step 6：SFT 训练（Day4–Day5）

### 8.1 产出
- outputs/sft_adapter/
- outputs/logs/sft_train.log

### 8.2 训练实现（优先复用仓库脚本）
- 参考：tianji/finetune/transformers/Qwen2_5/qwen2_5_train_lora.py
- 建议新建：comm_project/src/05_train_sft.py（封装成可配置版本）

必须改动项：
- system prompt 改为通信标准任务。
- 数据读取改为 data/sft/train.jsonl。
- model_repo 设置为 qwen/Qwen2.5-1.5B-Instruct。

建议超参（可直接填）：
- max_length=2048（显存不足则 1024）
- batch_size=1–4 + gradient_accumulation=8–16
- epoch=1–2（先跑通再加）
- lr=1e-4（LoRA）

### 8.3 验收标准
- 用评测集跑一遍：schema_pass_rate 相对 base 提升明显（>= +20pp 作为合理阈值）。

---

## 9. Step 7：DPO 数据构建（Day6）

### 9.1 产出
- data/dpo/train.jsonl（3000–8000 对）
- data/dpo/eval.jsonl（300–800 对）

### 9.2 prompt 组成
- 取自 eval_set 与 sft_eval 的 prompt 结构（带证据）。

### 9.3 候选生成
- 对同一 prompt 生成两份输出 A/B：
  - temperature=0.2 与 temperature=0.8（或 top_p 不同）
  - 或 base vs sft

### 9.4 规则打分（写死）
对每个候选 output 计算 score：
- JSON 可解析 +2
- schema 字段齐全 +2
- answerable 类型：
  - citations 非空 +1
  - quote 在证据中精确匹配 +3
  - answer 与证据覆盖率>=阈值 +2
- unanswerable 类型：
  - cannot_answer_reason 非空 +2
  - citations 为空或仅给出“无相关证据”型引用 +1
- 引用不存在文本 -5
- 明显编造（覆盖率很低）-5

过滤：score(A)-score(B) >= 3 才保留；chosen=高分。

---

## 10. Step 8：DPO 训练（Day7）

### 10.1 产出
- outputs/dpo_adapter/
- outputs/logs/dpo_train.log

### 10.2 训练设置（写死）
- 基础：加载 base 模型 + sft_adapter，再进行 DPO。
- lr=1e-5
- beta=0.1
- epoch=1

### 10.3 验收标准
- citation_exact_match_rate 相对 SFT 提升 >= +10pp 或 refusal_correct_rate 提升 >= +15pp（两者至少满足一个）。

---

## 11. Step 9：评测与报表（Day5/Day7/Day8）

### 11.1 产出
- reports/eval_base.json / eval_sft.json / eval_dpo.json（逐样本）
- reports/compare.md（汇总表 + 典型case）

### 11.2 指标定义（完全确定）
- json_valid_rate：能被 json.loads 解析
- schema_pass_rate：字段齐全 + confidence 枚举合法
- citation_exact_match_rate：每条 citation.quote 都能在对应 evidence chunk 中找到子串
- grounded_answer_rate：answer 的关键短语（或n-gram）在 evidence 中覆盖率 >= 阈值
- refusal_correct_rate：unanswerable 样本中 cannot_answer_reason 非空 且 citations 为空（或无效引用）

compare.md 必须包含：
- 指标表（base/sft/dpo）
- 10 条样例对比（同一问题，三版本输出）
- 失败分类（无效JSON/乱引用/编造/误拒答）及计数

---

## 12. Step 10：Demo（Day9）

### 12.1 功能（必须实现）
- 模型选择：base / sft / dpo
- 输入框：question + evidence（允许粘贴多 chunk）
- 输出：
  - 原始 JSON
  - 解析后的字段展示
  - 引用 quote 高亮（在 evidence 中定位）
- 日志落盘：data/logs/demo_calls.jsonl

### 12.2 禁止项
- 不做在线检索。
- 不做多Agent。
- 不做复杂UI。

---

## 13. 两人协作与分工（每一步都必须双人参与）

### 13.1 固定工作法
每个步骤执行同一流程：
1) 共同设计 20–30 分钟（写下输入/输出格式与验收标准）
2) 主责实现
3) 副责在另一台机器/环境复现运行（或重新拉取代码运行）
4) 互相讲解：主责讲“为什么这样做”，副责讲“我如何验证它正确”
5) 合并代码

### 13.2 主责/副责安排（并要求两日轮换）
- Day1–2：A 主责下载与规格；B 主责解析与chunk
- Day3–4：A 主责评测与SFT生成；B 主责数据质检与脚本化
- Day4–5：B 主责训练；A 主责评测对比与错误分析
- Day6：A 主责候选生成；B 主责规则打分与过滤
- Day7：B 主责DPO训练；A 主责三版本评测与case导出
- Day8–9：B 主责Demo；A 主责推理封装与日志

### 13.3 每日强制产出
- 当日可运行命令（写在 outputs/logs/dayX_commands.txt）
- 当日产物路径列表（写在 outputs/logs/dayX_outputs.txt）

---

## 14. 10 日计划（逐日必须完成的具体事项）

### Day1
- 冻结 schema + system prompt（写入 utils/prompts.py）
- 写 00_download.py + url 清单
- 下载 3–6 份文档，生成 manifest.json

### Day2
- 完成 01_parse_pdf.py + 02_build_corpus.py
- evidence_corpus.jsonl 生成 >=2000 chunks
- 抽样 200 chunk 验收通过

### Day3
- 03_make_eval.py 生成 eval_set 200–300 条
- 04_make_sft.py 生成 SFT train>=5000、eval>=300
- 数据质量报表（JSON可解析率、拒答占比、quote匹配率）

### Day4
- 跑 SFT 首轮训练（至少完成 1 epoch 或 1000 step）
- 输出 sft_adapter
- 08_evaluate.py 先跑 base 与 sft 对比（出 reports）

### Day5
- 修正 SFT 数据或 prompt（仅允许改 prompts.py 与 make_sft 规则）
- SFT 第二轮训练（可选）
- 固定 sft_adapter_v2 作为 DPO 基础

### Day6
- 06_make_dpo.py 生成 dpo train 3000–8000 对
- 输出过滤率与均分差统计

### Day7
- 跑 DPO 训练（1 epoch）
- 输出 dpo_adapter
- 08_evaluate.py 跑 base/sft/dpo 三方对比

### Day8
- 完成 compare.md 自动生成
- 导出 10 组典型case对比（同题三版本）
- 做失败样本分类统计

### Day9
- 09_demo_gradio.py 可运行
- 引用高亮、JSON校验、日志落盘可用

### Day10
- 将“复现命令”整理到 compare.md 或单独 docs（本仓库已有文档体系，优先写入本MD末尾）
- 录屏演示脚本（不写营销，只写操作步骤与预期输出）

---

## 15. 一键命令清单（保持不变）
```bash
python comm_project/src/00_download.py
python comm_project/src/01_parse_pdf.py
python comm_project/src/02_build_corpus.py
python comm_project/src/03_make_eval.py
python comm_project/src/04_make_sft.py
python comm_project/src/05_train_sft.py
python comm_project/src/06_make_dpo.py
python comm_project/src/07_train_dpo.py
python comm_project/src/08_evaluate.py --model base
python comm_project/src/08_evaluate.py --model sft
python comm_project/src/08_evaluate.py --model dpo
python comm_project/src/09_demo_gradio.py --model dpo
```

---

## 16. 变更控制（避免偏差）
允许改动范围（只允许这三类）：
1) prompts.py 的文本与 schema 校验细节
2) make_sft/make_dpo 的规则（必须保留可验证性与引用匹配）
3) 训练超参（batch/lr/epoch）

禁止改动范围：
- 不引入在线检索与外部知识
- 不更换任务方向为泛聊天
- 不把评测改为“主观感觉”
