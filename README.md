## 💡 Overview

Although Large Language Models (LLMs) have achieved remarkable progress in general reasoning, they struggle with tasks that require fine-grained distinctions between similar charges. These models often select plausible charges directly without discriminating among closely related alternatives. In this paper, we introduce GLARE, an agentic legal reasoning framework that enables models to actively retrieve and apply external knowledge during decision-making. Unlike static prediction, GLARE simulates comparative reasoning by dynamically expanding the decision space to include confusing candidates, then retrieving exclusionary logic from precedents and statutes to identify the correct judgment.

## 🧩 Method
We propose GLARE (*A**G**entic **L**eg**A**l **R**easoning Fram**E**work*), an agentic framework that reframes legal judgment as an active comparative reasoning process rather than passive prediction. GLARE explicitly externalizes the discrimination process through three integrated modules. 

- **Charge Expansion Module**: Expands a diverse set of confusing charges by leveraging multiple signals, such as legal structure and historical co-occurrence. This helps the model compare a wider range of candidates and avoid premature conclusions.
- **Precedents Reasoning Demonstration**: Extracts discriminative reasoning patterns from similar cases, providing concrete examples that guide the model in distinguishing subtle differences.
- **Legal Search-Augmented Reasoning**: Identifies remaining ambiguities and retrieves specific legal rules when needed to resolve competing charges. We guide the model to focus its search on differences between similar charges and details of how specific laws apply, rather than general case facts.  
  
<p align="center">
  <img src="figures/pipeline2.jpg" width="95%" />
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
python glare/offline_reasoning.py
vllm serve "model_path" --port 19001 --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --dtype float16 --served-model-name qwq
bash glare/run.sh
```

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


