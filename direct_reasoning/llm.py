import json
from tqdm import tqdm
import sys
from mycode.utils import loader
from modelscope import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run Direct Reasoning for various models.")

    # Dataset
    parser.add_argument(
        '--test_dataset_path',
        type=str,
        default="./test.json",
        help="Path of 1000 test data."
    )
    # Model configuration
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        choices=['7b','qwq','32b','72b'],
        help="Name of model to generate."
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the pre-trained predictive model."
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.01,
        help="Sampling temperature."
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.8,
        help="Top-p sampling parameter."
    )
    parser.add_argument(
        '--top_k_sampling',
        type=int,
        default=20,
        help="Top-k sampling parameter."
    )
    parser.add_argument(
        '--min_p',
        type=float,
        default=0,
        help="Min-p sampling parameter."
    )
    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=None,
        help="Repetition penalty."
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=32768,
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    # Extract arguments
    test_dataset_path = args.test_dataset_path
    model_name = args.model_name
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    top_k_sampling = args.top_k_sampling
    min_p = args.min_p
    repetition_penalty = args.repetition_penalty
    max_tokens = args.max_tokens

    output_dir = f'./output/llm/{model_name}' 
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(
        temperature=temperature, 
        top_p=top_p, 
        min_p=min_p, 
        top_k=top_k_sampling, 
        repetition_penalty=repetition_penalty, 
        max_tokens=max_tokens)

    llm = LLM(model=model_path,enforce_eager=True,trust_remote_code=True,gpu_memory_utilization=0.9,dtype=torch.bfloat16,tensor_parallel_size=torch.cuda.device_count())

    def generate_prompt(fact, criminal, is_multi_defendant=True):
        if is_multi_defendant:
            prediction_instruction = (
                f"为{criminal}中每个罪犯预测相关法条和罪名。\n"
                "输出格式为:【罪犯1:法条:……,罪名:……】【罪犯2:……】……"
            )
        else:
            prediction_instruction = f"得到和本案最相关的一个法条和罪名预测结果。只关注{criminal}的犯罪行为，把你的最终答案放在方括号【】中，格式：本案的相关法条是：第*条，本案的罪名是：*罪。"
        prompt = (
            "你要做的任务是相关法条预测和罪名预测。\n"
            f"本案事实：\n{fact}\n"
            f"请分析当前案件，根据你自己的知识，{prediction_instruction}"
        )
        return prompt

    def qwen_response(text_list):
        message_list = []
        for prompt in text_list:
            messages = [
                {"role": "user", "content": prompt}
            ]
            message_list.append(messages)
        texts = [tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) for messages in message_list]
        outputs = llm.generate(texts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def get_response(data):
        response_function = qwen_response
        truncated_data = []
        for i in tqdm(range(len(data)), desc="pre-process prompt"):
            case = data[i]
            fact = case["fact"].replace(" ", "") # raw fact
            is_multi_defendant = "defendants" in case
            if is_multi_defendant:
                criminal = case["defendants"]
                criminal = ",".join(criminal)
            else:
                criminal = case["meta"]["criminals"][0]
            prompt = generate_prompt(fact, criminal, is_multi_defendant)

            obj = {}
            obj["prompt"] = prompt
            obj["caseID"] = case["caseID"]
            truncated_data.append(obj)
        #truncated_data里面放着输入到llm进行预测的prompt和caseID

        print("Starting")
        i = 0
        batch = 1000
        while i < len(truncated_data):
            print(f"=== {i} / {len(data)} ===")
            prompt_list = truncated_data[i:i+batch]#[0,1][1,2][2,3]
            text_list = [t["prompt"] for t in prompt_list]
            case_list = [t["caseID"] for t in prompt_list]
            response_list = response_function(text_list)
            res_list = []
            for idx,response in enumerate(response_list):
                res = {}
                res["caseID"] = case_list[idx]
                res["predict"] = f"{response}"
                res_list.append(res)
                
            with open(os.path.join(output_dir, "predict_result"), "a+", encoding="utf-8") as f:
                for res in res_list:
                    line = json.dumps(res, ensure_ascii=False)
                    print("写入文件中的结果：")
                    print(line)
                    f.write(line + "\n")
            i += batch

    def run_llm():
        data = []
        #需要生成标签的文件
        with open(test_dataset_path, encoding="utf-8") as f:
            for line in f.readlines():
                case = json.loads(line)
                data.append(case)
        # data = data[:10]
        print("to run data count: ", len(data))
        get_response(data)
    
    run_llm()
    print("Process completed.")

if __name__ == "__main__":
    main()
