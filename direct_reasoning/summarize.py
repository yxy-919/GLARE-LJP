import json
from tqdm import tqdm
import sys
from modelscope import AutoTokenizer
import torch
from vllm import LLM, EngineArgs, SamplingParams
import torch
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize.")

    # Dataset
    parser.add_argument(
        '--test_dataset_path',
        type=str,
        default="./test.json",
        help="Path of 1000 test data."
    )

    # Model configuration
    parser.add_argument(
        '--method_model',
        type=str,
        required=True,
        choices=['7b','qwq','32b','72b'],
        help="Name of model to generate."
    )
    parser.add_argument(
        '--method_name',
        type=str,
        required=True,
        choices=['llm','rag','glare'],
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
    method_model = args.method_model
    method_name = args.method_name
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    top_k_sampling = args.top_k_sampling
    min_p = args.min_p
    repetition_penalty = args.repetition_penalty
    max_tokens = args.max_tokens

    output_dir = f'./output/{method_name}/{method_model}'
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

    #一条数据
    def generate_prompt(predict, criminals, is_multi_defendant=True):
        # if "</think>" in predict:
        #     analysis_part = predict.split("</think>")[1].strip()
        # else:
        #     analysis_part = predict 
        #return f"用<>括起来的是本案事实，用---括起来的是模型利用自己的知识对案例进行一步一步地分析得到的相关法条和罪名预测结果，请根据本案事实和预测结果，从中总结提取出最相关的一个法条和罪名,重点关注用()括起来的罪犯的犯罪行为。输出格式为：本案的相关法条是：第*条，本案的罪名是：*罪。请严格按照该输出格式生成结果。\n"+f"<"+fact+f">"+f"\n---"+predict +f"---\n"+f"("+criminals +f")\n"
        if is_multi_defendant:
            return (
                "你是一个擅长提取结论并进行格式化输出的专家。"
                f"给定一段对案件相关法条和罪名的预测内容，理解该内容，从中总结出各罪犯{criminals}的相关法条和罪名。"
                f"按照{criminals}的顺序输出，输出格式为：【罪犯1:法条是:第*条，第*条，……；罪名是:*罪，*罪，……】【罪犯2:……】"
                f"预测内容：{predict}"
            )
        else:
            return (
                "你是一个擅长提取结论并进行格式化输出的专家。"
                "给定一段对案件相关法条和罪名的预测内容，理解该内容，从中总结提取出和本案最相关的一个法条和罪名。"
                f"只关注{criminals}的犯罪行为，把你的最终答案放在方括号【】中，格式：本案的相关法条是：第*条，本案的罪名是：*罪。"
                f"预测内容：{predict}"
            )

    def qwen_response(text_list):
        message_list = []
        for prompt in text_list:
            messages = [
                {"role": "system", "content": "你是一个精通法律的专家，你的任务是相关法条预测和罪名预测。"},
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
    
    def get_response(fact_data,predict_data):
        response_function = qwen_response
        truncated_data = []
        #data里面包括多个case
        for i in tqdm(range(len(fact_data)), desc="pre-process prompt"):
            predict = predict_data[i]["predict"].replace(" ", "")
            case = fact_data[i]
            is_multi_defendant = "defendants" in case
            if is_multi_defendant:
                criminal = case["defendants"]
                criminal = ",".join(criminal)
            else:
                criminal = case["meta"]["criminals"][0]
            prompt = generate_prompt(predict, criminal, is_multi_defendant)

            obj = {}
            obj["prompt"] = prompt
            obj["caseID"] = fact_data[i]["caseID"]
            truncated_data.append(obj)
        #truncated_data里面放着输入到llm进行预测的prompt和caseID

        print("Starting")
        i = 0
        batch = 100
        while i < len(truncated_data):
            print(f"=== {i} / {len(fact_data)} ===")
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
                
            with open(os.path.join(output_dir, "generated_result"), "a+", encoding="utf-8") as f:
                for res in res_list:
                    line = json.dumps(res, ensure_ascii=False)
                    print("写入文件中的结果：")
                    print(line)
                    f.write(line + "\n")
            i += batch

    def run_llm():
        predict_data = []
        #需要生成标签的文件
        with open(os.path.join(output_dir, "predict_result"), encoding="utf-8") as f:
            for line in f.readlines():
                case = json.loads(line)
                predict_data.append(case)
        print("to run prediction data count: ", len(predict_data))
        fact_data = []
        with open(test_dataset_path,encoding="utf-8") as f:
            for line in f.readlines():
                case = json.loads(line)
                fact_data.append(case)
        print("to run fact data count: ",len(fact_data))
        get_response(fact_data,predict_data)

    print("已记载好模型！开始生成！")
    run_llm()

if __name__ == "__main__":
    main()
