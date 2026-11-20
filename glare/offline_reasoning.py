"""Generate reasoning paths for every case in the precedent database before retrieval."""
import argparse
import json
from pathlib import Path
from precedents import load_cases


def precedent_judgment(case):
    # Preserve the original per-defendant judgment formatting and charge collection.
    defendants = case["defendants"]
    all_str = ""
    all_charge_list = []
    for idx, item in enumerate(case["outcomes"]):
        defendant = defendants[idx]
        each_defendant = "罪犯" + defendant
        charge_list = []
        article_list = []
        for judgment in item['judgment']:
            charge = judgment['standard_accusation'].strip()
            articles = judgment['article']
            charge_list.append(charge)
            article_list.extend(articles)
        charge_str = ",".join(charge_list)
        all_charge_list.extend(charge_list)
        article_str = ",".join(article_list)
        each_defendant += ": 法条是: " + article_str + "；罪名是: " + charge_str + "。"
        all_str += each_defendant
    return all_str, all_charge_list


def build_prompt(case, expanded_crimes):
    fact = case["fact"].replace(" ", "")
    all_str, _ = precedent_judgment(case)
    expanded_crimes_str = ', '.join(expanded_crimes)
    # Keep the original offline reasoning prompt unchanged.
    return (
        "给定一段法律案件的案情描述以及正确的法条和罪名，并给出其他的罪名，分析选择正确罪名排除其他罪名的理由。\n"
        f"案情描述：\n{fact}\n"
        f"正确的法条和罪名：\n{all_str}\n"
        f"其他的罪名：\n{expanded_crimes_str}\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    # 类案库来源于 CAIL2018、CMDL 等数据集的训练集，以及收集的裁判文书网案件。
    # precedent_database.jsonl 是虚拟路径，需要自行准备；不得包含评测集案件。
    parser.add_argument('--precedent_database_path', default='./data/precedent_database.jsonl')
    parser.add_argument('--output_path', default='./data/precedent_database_with_reasoning.jsonl')
    parser.add_argument('--multiljp_file_path', required=True,
                        help='MultiLJP training JSONL for historical charge co-occurrence.')
    parser.add_argument('--model_path', default='./model/Qwen2.5-72B-Instruct')
    parser.add_argument('--bge_model_path', default='./model/BAAI_bge-m3')
    parser.add_argument('--tensor_parallel_size', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--top_k_expand', type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    if Path(args.output_path).resolve() == Path(args.precedent_database_path).resolve():
        raise ValueError('Output must differ from the source precedent database')
    if args.batch_size <= 0:
        raise ValueError('batch_size must be positive')
    cases = load_cases(args.precedent_database_path)
    for case in cases:
        _, charges = precedent_judgment(case)
        if not charges:
            raise ValueError(f'Missing charge labels for case {case["caseID"]}')

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from sentence_transformers import SentenceTransformer
    from expand_crimes import expand_crimes
    import torch

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, enforce_eager=True, trust_remote_code=True,
              gpu_memory_utilization=0.8, tensor_parallel_size=args.tensor_parallel_size,
              dtype=torch.bfloat16)
    bge_model = SentenceTransformer(args.bge_model_path)
    multiljp_file_path = args.multiljp_file_path
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # One output record per source case, with its original ID and labels preserved.
    # Write to a temporary file so failed generation cannot replace the complete database.
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as target:
        for start in range(0, len(cases), args.batch_size):
            batch = cases[start:start + args.batch_size]
            prompts = []
            for case in batch:
                _, charges = precedent_judgment(case)
                expanded = expand_crimes(charges, bge_model, args.top_k_expand, multiljp_file_path)
                expanded = list(set(expanded))
                prompts.append(tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': build_prompt(case, expanded)}],
                    tokenize=False, add_generation_prompt=True))
            outputs = llm.generate(prompts, sampling_params=SamplingParams(
                temperature=0.3, top_p=0.95, top_k=20, min_p=0,
                repetition_penalty=1.1, max_tokens=32768))
            if len(outputs) != len(batch):
                raise ValueError('Generation count does not match precedent count')
            for case, output in zip(batch, outputs):
                reasoning = output.outputs[0].text
                if not reasoning.strip():
                    raise ValueError(f'Empty reasoning for case {case["caseID"]}')
                record = dict(case, reasoning_chain=reasoning)
                target.write(json.dumps(record, ensure_ascii=False) + '\n')
    temporary_path.replace(output_path)
    print(f'Generated reasoning for all {len(cases)} precedents: {output_path}')


if __name__ == '__main__':
    main()
