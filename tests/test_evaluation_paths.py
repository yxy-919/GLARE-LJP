"""Check configurable evaluation I/O using synthetic cases and mocked model weights."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
try:
    import numpy as np
    import sklearn
except ImportError:
    np = None


@unittest.skipIf(np is None, 'Install numpy and scikit-learn for evaluation I/O tests')
class EvaluationPathTests(unittest.TestCase):
    def run_evaluation(self, script, labels, predictions, save=True, modules=None):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            label_file, prediction_file = folder / 'labels.jsonl', folder / 'predictions.jsonl'
            label_file.write_text(''.join(json.dumps(case) + '\n' for case in labels))
            prediction_file.write_text(''.join(json.dumps(case) + '\n' for case in predictions))
            output = folder / 'nested/metrics.json'
            argv = [script, '--pred_file_path', str(prediction_file), '--label_file_path', str(label_file)]
            if save:
                argv += ['--output_file', str(output)]
            previous = os.getcwd()
            try:
                # Work outside the repository to verify mapping/resource path resolution.
                os.chdir(folder)
                with patch.object(sys, 'argv', argv), patch.object(sys, 'path', [str(ROOT / 'metrics')] + sys.path), patch.dict(sys.modules, modules or {}), contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(ROOT / 'metrics' / script), run_name='__main__')
            finally:
                os.chdir(previous)
            if save:
                return json.loads(output.read_text())
            self.assertFalse(output.exists())

    def test_article_paths_and_optional_output(self):
        labels = [{'caseID': 1, 'meta': {'relevant_articles': [264]}}]
        predictions = [{'caseID': 1, 'predict': '【本案的相关法条是：第264条，本案的罪名是：盗窃罪。】'}]
        result = self.run_evaluation('cal_single_defendant_article_metrics.py', labels, predictions)
        self.assertEqual(result['accuracy'], 1.0)
        self.run_evaluation('cal_single_defendant_article_metrics.py', labels, predictions, save=False)

    def model_stubs(self):
        embeddings = np.load(ROOT / 'metrics/crime_encodings.npy')
        class Model:
            def __init__(self, path): pass
            def encode(self, charges): return embeddings[:1]
        return {'sentence_transformers': types.SimpleNamespace(SentenceTransformer=Model),
                'modelscope': types.SimpleNamespace(AutoTokenizer=None),
                'vllm': types.SimpleNamespace(LLM=None, SamplingParams=None),
                'tqdm': types.SimpleNamespace(tqdm=lambda value, **kw: value)}

    def test_charge_paths(self):
        charge = next(iter(json.loads((ROOT / 'metrics/c2i.json').read_text())))
        labels = [{'caseID': 1, 'meta': {'accusation': [charge]}}]
        predictions = [{'caseID': 1, 'predict': f'【本案的相关法条是：第264条，本案的罪名是：{charge}罪。】'}]
        result = self.run_evaluation('cal_single_defendant_charge_metrics.py', labels, predictions, modules=self.model_stubs())
        self.assertEqual(result['accuracy'], 1.0)

    def test_multi_paths_and_mapping_resolution(self):
        # Small numpy-backed tensor substitute exercises the existing weighted formulas.
        class Tensor:
            def __init__(self, value): self.value = np.asarray(value)
            @property
            def shape(self): return self.value.shape
            @property
            def device(self): return 'cpu'
            def size(self, dim): return self.value.shape[dim]
            def to(self, device): return self
            def float(self): return Tensor(self.value.astype(float))
            def item(self): return self.value.item()
            def __add__(self, other): return Tensor(self.value + unwrap(other))
            def __sub__(self, other): return Tensor(self.value - unwrap(other))
            def __rsub__(self, other): return Tensor(unwrap(other) - self.value)
            def __mul__(self, other): return Tensor(self.value * unwrap(other))
            def __rmul__(self, other): return Tensor(unwrap(other) * self.value)
            def __truediv__(self, other): return Tensor(self.value / unwrap(other))
            def __eq__(self, other): return Tensor(self.value == unwrap(other))
        def unwrap(value): return value.value if isinstance(value, Tensor) else value
        torch = types.SimpleNamespace(Tensor=Tensor, tensor=Tensor,
            sum=lambda value, dim=None: Tensor(np.sum(unwrap(value), axis=dim)),
            all=lambda value, dim=None: Tensor(np.all(unwrap(value), axis=dim)),
            clamp=lambda value, min: Tensor(np.maximum(unwrap(value), min)))
        charge = next(iter(json.loads((ROOT / 'metrics/c2i.json').read_text())))
        article = next(iter(json.loads((ROOT / 'metrics/a2i.json').read_text())))
        labels = [{'caseID': 1, 'outcomes': [{'judgment': [{'standard_accusation': charge, 'article': [article]}]}] * 2}]
        predictions = [{'caseID': 1, 'predict': ''.join(f'【罪犯{i}:法条:第{article}条,罪名:{charge}罪】' for i in [1, 2])}]
        modules = self.model_stubs()
        modules['torch'] = torch
        result = self.run_evaluation('multi_defendant_metrics.py', labels, predictions, modules=modules)
        self.assertEqual(result['case_level_Acc_Acc'], 1.0)
        self.assertEqual(result['case_level_Art_Acc'], 1.0)

    def test_output_cannot_overwrite_input(self):
        sys.path.insert(0, str(ROOT / 'metrics'))
        try:
            from evaluation_io import parse_evaluation_args
        finally:
            sys.path.pop(0)
        with patch.object(sys, 'argv', ['evaluate', '--pred_file_path', 'pred.jsonl', '--label_file_path', 'gold.jsonl', '--output_file', 'pred.jsonl']), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_evaluation_args('Test')


if __name__ == '__main__':
    unittest.main()
