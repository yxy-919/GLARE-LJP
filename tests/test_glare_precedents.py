"""CPU-only checks for corpus coverage, retrieval alignment, and combined expansion."""
import ast
import functools
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'glare'))
from precedents import case_charges, clean_charge, PrecedentRetriever
import offline_reasoning


def expansion_functions():
    tree = ast.parse((ROOT / 'glare/expand_crimes.py').read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    from collections import defaultdict
    namespace = dict(json=json, defaultdict=defaultdict, lru_cache=functools.lru_cache,
                     case_charges=case_charges, clean_charge=clean_charge)
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'expansion', 'exec'), namespace)
    return namespace


class PrecedentTests(unittest.TestCase):
    def test_historical_schemas_and_combined_expansion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'history.jsonl'
            records = [
                {'criminals_info': {'a': {'accusations': ['盗窃罪']}, 'b': {'accusations': ['抢劫罪']}}},
                {'outcomes': [{'judgment': [{'standard_accusation': '盗窃'}, {'accusation': '诈骗'}]}]},
                {'meta': {'accusation': ['[盗窃]', '诈骗']}},
            ]
            path.write_text(''.join(json.dumps(r) + '\n' for r in records))
            ns = expansion_functions()
            self.assertEqual(ns['expand_historical_co_occurrence'](str(path), ['盗窃'], 1), ['诈骗'])
            ns['np'] = types.SimpleNamespace(load=lambda f: None)
            ns['bge_match'] = lambda *a, **kw: ['盗窃', '抢劫']
            previous = os.getcwd()
            os.chdir(ROOT)
            try:
                result = ns['expand_crimes'](['盗窃罪'], None, 1, str(path))
            finally:
                os.chdir(previous)
            self.assertEqual(set(result), {'抢劫', '诈骗'})

    def test_retrieval_uses_index_rows_and_defaults_to_five(self):
        retriever = PrecedentRetriever.__new__(PrecedentRetriever)
        retriever.cases = [dict(caseID=f'id-{i * 10}', fact=f'fact-{i}', reasoning_chain=f'chain-{i}') for i in range(7)]
        calls = []
        class Index:
            def search(self, vector, k):
                calls.append(k)
                return None, [[6, 2, 0, 4, 1][:k]]
        retriever.index = Index()
        retriever.encode = lambda facts: facts
        output = retriever.retrieve('query')
        self.assertEqual(calls, [5])
        self.assertEqual(output.count('个类案的分析'), 5)
        self.assertIn('caseID=id-60', output)
        self.assertIn('fact-6\nchain-6', output)
        self.assertNotIn('chain-3', output)

    def test_offline_generates_all_records_without_test_retrieval(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'precedents.jsonl'
            target = Path(folder) / 'enriched.jsonl'
            records = [dict(caseID=f'case-{i}', fact=f'fact-{i}', meta={'accusation': ['盗窃']}) for i in range(7)]
            source.write_text(''.join(json.dumps(r) + '\n' for r in records))
            args = types.SimpleNamespace(precedent_database_path=str(source), output_path=str(target),
                historical_cases_path=None, model_path='fake', bge_model_path='fake',
                tensor_parallel_size=1, batch_size=3, top_k_expand=3)
            class Tokenizer:
                @classmethod
                def from_pretrained(cls, path): return cls()
                def apply_chat_template(self, messages, **kw): return messages[0]['content']
            class LLM:
                def __init__(self, **kw): pass
                def generate(self, prompts, **kw):
                    return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text='reasoning:' + p)]) for p in prompts]
            modules = {
                'transformers': types.SimpleNamespace(AutoTokenizer=Tokenizer),
                'vllm': types.SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: kw),
                'sentence_transformers': types.SimpleNamespace(SentenceTransformer=lambda path: None),
                'torch': types.SimpleNamespace(bfloat16='fake'),
                'expand_crimes': types.SimpleNamespace(expand_crimes=lambda *a: ['抢劫']),
            }
            with patch.object(offline_reasoning, 'parse_args', return_value=args), patch.dict(sys.modules, modules):
                offline_reasoning.main()
            outputs = [json.loads(line) for line in target.read_text().splitlines()]
            self.assertEqual([r['caseID'] for r in outputs], [r['caseID'] for r in records])
            for record, output in zip(records, outputs):
                self.assertIn(record['fact'], output['reasoning_chain'])
                self.assertEqual(record['meta'], output['meta'])
            self.assertFalse(Path(str(target) + '.tmp').exists())


if __name__ == '__main__':
    unittest.main()
