## 💡 Overview

Although Large Language Models (LLMs) have achieved remarkable progress in general reasoning, they struggle with tasks that require fine-grained distinctions between similar charges. These models often select plausible charges directly without discriminating among closely related alternatives. In this paper, we introduce GLARE, an agentic legal reasoning framework that enables models to actively retrieve and apply external knowledge during decision-making. Unlike static prediction, GLARE simulates comparative reasoning by dynamically expanding the decision space to include confusing candidates, then retrieving exclusionary logic from precedents and statutes to identify the correct judgment.

## 🧩 Method
We propose GLARE (*A**G**entic **L**eg**A**l **R**easoning Fram**E**work*), an agentic framework that reframes legal judgment as an active comparative reasoning process rather than passive prediction. GLARE explicitly externalizes the discrimination process through three integrated modules. 

- **Charge Expansion Module**: Expands a diverse set of confusing charges by leveraging multiple signals, such as legal structure and historical co-occurrence. This helps the model compare a wider range of candidates and avoid premature conclusions.
- **Precedents Reasoning Demonstration**: Extracts discriminative reasoning patterns from similar cases, providing concrete examples that guide the model in distinguishing subtle differences.
- **Legal Search-Augmented Reasoning**: Identifies remaining ambiguities and retrieves specific legal rules when needed to resolve competing charges. We guide the model to focus its search on differences between similar charges and details of how specific laws apply, rather than general case facts.  
  
<p align="center">
  <img src="figures/pipeline_new.jpg" width="95%" />
</p>

## 🔧 Installation
### Environment Setup
```bash
# Create conda environment
conda create -n glare python=3.9
conda activate glare

# Install requirements
cd GLARE
pip install -r requirements.txt
```
## 🏃 Quick Start

### Data Preparation

- **single-defendant scenario:**  
  [CAIL2018 dataset](https://github.com/china-ai-law-challenge/CAIL2018)

- **multi-defendant scenario:**  
  [CMDL dataset](https://github.com/littlebowlnju/CMDL)

### Model Inference

You can run different inference modes using the provided scripts. Below are examples of how to execute each mode:

1. **Direct Reasoning**
```bash
bash direct_reasoning/run.sh
```

2. **Retrieval-augmented Reasoning**
```bash
python retrieval_augmented_reasoning/retrieve.py
bash retrieval_augmented_reasoning/run.sh
```

3. **GLARE(Ours)**
```bash
# Build reasoning paths for ALL precedents, before retrieving any test cases.
python glare/offline_reasoning.py \
    --precedent_database_path ./data/precedent_database.jsonl \
    --output_path ./data/precedent_database_with_reasoning.jsonl
vllm serve "model_path" --port 19001 --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --dtype float16 --served-model-name qwq
bash glare/run.sh
```

### GLARE precedent database

`data/precedent_database.jsonl` is a placeholder path: prepare a JSONL precedent
corpus from the **training splits** of CAIL2018, CMDL, and collected China Judgments
Online cases. Exclude evaluation cases and duplicates. Each record needs a unique
`caseID`, `fact`, and gold judgments. Supported judgment schemas are unified
`outcomes[].judgment[]` (`standard_accusation` or `accusation`, plus `article`),
CAIL2018 `meta.accusation`, and MultiLJP `criminals_info.*.accusations`.
For example (synthetic schema illustration):

```json
{"caseID": "example-1", "fact": "示例案情文本", "defendants": ["张某"], "outcomes": [{"judgment": [{"standard_accusation": "盗窃", "article": ["264"]}]}]}
```

The offline script writes one enriched record per source case, retaining the
original fields and adding `reasoning_chain`. It processes the entire corpus in
batches; it does not read test cases or preselected retrieval results.

`glare/glare.py` builds a SAILER/FAISS index over this enriched corpus and retrieves
**5 precedents per input case** (`--precedent_top_k 5`; fewer only if the corpus
contains fewer than 5 cases). Retrieved facts and reasoning paths provide
`seq['reasoning_chain']`; case IDs need not be consecutive integers. The index is
rebuilt on startup and is currently kept in memory.

Charge expansion combines legal-structure neighbors with top-k historical
co-occurrences. Historical statistics can use multi-defendant, multi-charge
**training data**, such as MultiLJP or CMDL. By default, both offline generation
and online inference use `data/precedent_database.jsonl` for these statistics.
To use a separate corpus, pass `--historical_cases_path /path/to/training.jsonl`
to the offline script and set the same argument in `glare/run.sh`. Statistics are
cached in memory for the duration of the process. Required local models are
Qwen2.5-72B-Instruct (offline generator), BGE-M3 (charge expansion), SAILER_zh
(precedent retrieval), and the online model served by vLLM; adjust model paths
and GPU settings for your environment.

### Evaluation
- For single-defendant dataset:
```bash
python metrics/cal_single_defendant_article_metrics.py
python metrics/cal_single_defendant_charge_metrics.py
```

- For multi-defendant dataset:
```bash
python multi_defendant_metrics.py
```

## 📄 License

This project is released under the [MIT License](LICENSE).


