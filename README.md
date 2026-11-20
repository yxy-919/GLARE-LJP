<h1 align="center"> ✨GLARE: Agentic Reasoning for</br> Legal Judgment Prediction</a></h1>


<p align="center">
  <br>
  <a href="https://scholar.google.com/citations?user=CF_NOtoAAAAJ&hl=en" target="_blank">Xinyu Yang</a>,&nbsp;
  <a href="https://scholar.google.com/citations?user=nqp_-AsAAAAJ&hl=en" target="_blank">Chenlong Deng</a>,&nbsp;
  <a href="http://playbigdata.ruc.edu.cn/dou/" target="_blank">Zhicheng Dou</a>&nbsp;<sup>&#x2709;</sup>;
  <br>
</p>

<p align="center">
  Gaoling School of Artificial Intelligence, Renmin University of China.
  <br>
  <sup>&#x2709;</sup> Corresponding Author
</p>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

## 💡 Overview

Although Large Language Models (LLMs) have achieved remarkable progress in general reasoning, they struggle with tasks that require fine-grained distinctions between similar charges. These models often select plausible charges directly without discriminating among closely related alternatives. In this paper, we introduce GLARE, an agentic legal reasoning framework that enables models to actively retrieve and apply external knowledge during decision-making. Unlike static prediction, GLARE simulates comparative reasoning by dynamically expanding the decision space to include confusing candidates, then retrieving exclusionary logic from precedents and statutes to identify the correct judgment.

This repository provides the implementation of the core methods presented in the paper, together with the necessary supporting resources. External resources, such as the precedent database, can be constructed to suit individual research needs using public datasets or other lawfully obtained data.

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

CAIL2018 covers **202 charge categories**. We randomly sample cases uniformly
across charge categories to construct a balanced test set of **1,000 cases**,
covering **190 charge categories**.
Test-set selections may vary across studies. Here, we aim for balanced sampling
across all charge categories to assess the framework's adaptability to charges
of varying difficulty. For reproduction, you can randomly sample a test set
and evaluate all compared methods on the same sampled cases.

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
    --output_path ./data/precedent_database_with_reasoning.jsonl \
    --multiljp_file_path "$MULTILJP_FILE_PATH"
vllm serve "model_path" --port 19001 --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --dtype float16 --served-model-name qwq
bash glare/run.sh
```

### GLARE Modules and Preparation

1. **Charge Expansion Module (CEM)**

   Candidate charges are expanded from two complementary perspectives:
   **legal structure-based expansion** retrieves semantically similar charges
   within the same legal chapter and across different chapters using BGE-M3;
   **history-based expansion** retrieves charges that frequently co-occur in
   historical cases. The two expansion sets are combined to provide a broader
   set of candidates for comparative reasoning.

   Specify the co-occurrence corpus with `--multiljp_file_path`. You can prepare
   it from the training splits of multi-defendant, multi-charge datasets such as
   MultiLJP or CMDL. This argument has no default path. Set `MULTILJP_FILE_PATH`
   to your local training JSONL file before running the commands above;
   `glare/run.sh` uses the same environment variable.

2. **Precedents Reasoning Demonstration (PRD)**

   - **Prepare the precedent database.** Collect precedents from the training
     splits of CAIL2018 and CMDL, together with cases collected from China
     Judgments Online. Remove duplicates and exclude evaluation cases. Store
     the corpus at `data/precedent_database.jsonl` or specify its location with
     `--precedent_database_path`.

   - **Generate reasoning paths offline.** Run `glare/offline_reasoning.py` to
     generate a reasoning path for every precedent. Given the case facts, the
     correct judgment, and expanded candidate charges, Qwen2.5-72B-Instruct
     explains why the correct charges apply and why the alternatives should be
     excluded. The script retains the original records and adds
     `reasoning_chain`, writing the enriched corpus to
     `data/precedent_database_with_reasoning.jsonl` by default.

   - **Retrieve precedents online.** `glare/glare.py` uses SAILER and FAISS to
     retrieve five relevant precedents per input case (`--precedent_top_k 5`).
     Their facts and reasoning paths serve as demonstrations for distinguishing
     candidate charges. For repeated experiments, save the FAISS index and its
     corresponding precedent ordering after construction, then load them on
     subsequent runs to avoid rebuilding the index. The current implementation
     builds the index in memory at startup; index persistence needs to be added
     locally.

3. **Legal Search-Augmented Reasoning (LSAR)**

   During reasoning, the model generates targeted legal queries to resolve
   ambiguities between candidate charges or clarify statute-specific questions.
   The system calls the **Serper API** to search the web, fetches relevant page
   content, and summarizes useful legal knowledge before returning it to the
   model for continued reasoning. This search-and-reasoning process can repeat
   as needed, subject to `--max_search_limit` and `--max_turn`.

   Configure your Serper API key in `glare/serper.py`. Webpage fetching through
   Jina is controlled by `--use_jina`, with credentials supplied through
   `--jina_api_key`.
   With `--use_jina False` (the default), pages are fetched directly using
   `requests`: HTML text is extracted with BeautifulSoup, and PDF text with
   pdfplumber. This mode does not require Jina credentials. Set `--use_jina True`
   to use Jina instead. Direct fetching may fail on sites requiring JavaScript
   rendering or blocking automated requests; scanned PDFs require OCR, which is
   not included in this implementation.

### Evaluation

After obtaining reasoning results from any method (Direct Reasoning,
Retrieval-augmented Reasoning, or GLARE), you can use
`direct_reasoning/summarize.py` to further format the predictions before evaluation.
It supports both single-defendant and multi-defendant cases.

```bash
python direct_reasoning/summarize.py \
    --test_dataset_path "$TEST_DATASET_PATH" \
    --method_name "$METHOD_NAME" \
    --method_model "$METHOD_MODEL" \
    --model_path "$MODEL_PATH"
```

Set `METHOD_NAME` to `llm`, `rag`, or `glare`, and `METHOD_MODEL` to the model
identifier (`7b`, `qwq`, `32b`, or `72b`). The script reads
`output/{method_name}/{method_model}/predict_result` and writes formatted predictions
to `output/{method_name}/{method_model}/generated_result`. For GLARE, use `--method_name glare --method_model qwq`; its raw predictions
are saved directly to `output/glare/qwq/predict_result`.
Use `--pred_file_path` to select the corresponding `generated_result` file and
`--label_file_path` to select the test cases. All evaluation scripts print metrics
to the terminal; optionally pass `--output_file` to save them as JSON. The same
commands work for any reasoning method or model by changing these paths.
For charge evaluation, `--bge_model_path` optionally overrides the local BGE model
path (default: `./model/BAAI_bge-m3`). Label mappings and charge embeddings are
loaded from the repository's `metrics/` directory.

Set the environment variables below to your actual input and output paths;
omit `--output_file` if only terminal output is needed.

- For single-defendant dataset:
```bash
python metrics/cal_single_defendant_article_metrics.py \
    --pred_file_path "$PRED_FILE_PATH" \
    --label_file_path "$TEST_DATASET_PATH" \
    --output_file "$ARTICLE_METRICS_PATH"
python metrics/cal_single_defendant_charge_metrics.py \
    --pred_file_path "$PRED_FILE_PATH" \
    --label_file_path "$TEST_DATASET_PATH" \
    --output_file "$CHARGE_METRICS_PATH"
```

- For multi-defendant dataset:
```bash
python metrics/multi_defendant_metrics.py \
    --pred_file_path "$PRED_FILE_PATH" \
    --label_file_path "$TEST_DATASET_PATH" \
    --output_file "$MULTI_METRICS_PATH"
```

## 📄 License

This project is released under the [MIT License](LICENSE).


