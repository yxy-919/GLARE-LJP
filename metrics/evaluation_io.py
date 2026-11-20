"""Common command-line paths and result serialization for evaluation scripts."""
import argparse
import json
from pathlib import Path

METRICS_DIR = Path(__file__).resolve().parent


def parse_evaluation_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--pred_file_path', required=True,
                        help='JSONL predictions containing caseID and predict.')
    parser.add_argument('--label_file_path', required=True,
                        help='JSONL test cases with gold judgments.')
    parser.add_argument('--output_file', default=None,
                        help='Optional path to save metrics as JSON; stdout is always printed.')
    parser.add_argument('--bge_model_path', default='./model/BAAI_bge-m3',
                        help='Local BGE model for charge matching (unused for article-only evaluation).')
    args = parser.parse_args()
    if args.output_file:
        output = Path(args.output_file).resolve()
        if output in {Path(args.pred_file_path).resolve(), Path(args.label_file_path).resolve()}:
            parser.error('--output_file must differ from prediction and label files')
    return args


def save_metrics(result, output_file):
    if output_file:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
