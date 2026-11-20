"""Shared precedent schema and retrieval over the complete reasoning-augmented database."""
import json


def clean_charge(charge):
    charge = str(charge).replace('[', '').replace(']', '').strip()
    return charge[:-1] if charge.endswith('罪') else charge


def case_charges(case):
    """Read unified outcomes, CAIL2018 metadata, or MultiLJP criminals_info."""
    charges = []
    for outcome in case.get('outcomes', []):
        for judgment in outcome.get('judgment', []):
            charge = judgment.get('standard_accusation') or judgment.get('accusation')
            if charge:
                charges.append(charge)
    charges.extend(case.get('meta', {}).get('accusation', []))
    for criminal in case.get('criminals_info', {}).values():
        charges.extend(criminal.get('accusations', []))
    return sorted({clean_charge(charge) for charge in charges if clean_charge(charge)})


def fact_text(fact):
    return ''.join(fact) if isinstance(fact, list) else str(fact)


def load_cases(path):
    cases = []
    seen = set()
    with open(path, encoding='utf-8') as source:
        for line in source:
            if not line.strip():
                continue
            case = json.loads(line)
            case_id = case['caseID']
            if case_id in seen:
                raise ValueError(f'Duplicate caseID: {case_id}')
            seen.add(case_id)
            if not fact_text(case['fact']).strip():
                raise ValueError(f'Empty fact: {case_id}')
            cases.append(case)
    if not cases:
        raise ValueError(f'Empty precedent database: {path}')
    return cases


class PrecedentRetriever:
    def __init__(self, database_path, model_path, batch_size=64):
        import faiss
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.cases = load_cases(database_path)
        for case in self.cases:
            if not case.get('reasoning_chain', '').strip():
                raise ValueError(f'Missing reasoning_chain for case {case["caseID"]}')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        self.index = faiss.IndexFlatL2(self.model.config.hidden_size)
        for start in range(0, len(self.cases), batch_size):
            facts = [fact_text(case['fact']) for case in self.cases[start:start + batch_size]]
            self.index.add(self.encode(facts))

    def encode(self, facts):
        inputs = self.tokenizer(facts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with self.torch.no_grad():
            vectors = self.model(**inputs).last_hidden_state[:, 0, :]
        return vectors.cpu().numpy().astype('float32')

    def retrieve(self, fact, k=5):
        if k <= 0:
            raise ValueError('precedent_top_k must be positive')
        _, indices = self.index.search(self.encode([fact_text(fact)]), min(k, len(self.cases)))
        demonstrations = []
        for rank, index in enumerate(indices[0], 1):
            case = self.cases[int(index)]
            demonstrations.append(
                f'第{rank}个类案的分析（caseID={case["caseID"]}）：\n'
                f'案情描述：\n{fact_text(case["fact"])}\n{case["reasoning_chain"]}'
            )
        return '\n\n'.join(demonstrations)
