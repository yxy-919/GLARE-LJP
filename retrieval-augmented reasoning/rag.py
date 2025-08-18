import json
from tqdm import tqdm
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("/home/u2024001059/PLJP")
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import argparse
import os
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Run Naive RAG for various models.")

    # Dataset
    parser.add_argument(
        '--test_dataset_path',
        type=str,
        default="./test.json",
        help="Path of 1000 test data."
    )
    parser.add_argument(
        '--precedent_database_path',
        type=str,
        default="./train.json",
        help="Path of precedent database."
    )
    # Model configuration
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        choices=['7b','qwq','32b','72b', 'ds7b','ds14b','ds32b'],
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
    precedent_database_path = args.precedent_database_path
    model_name = args.model_name
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    top_k_sampling = args.top_k_sampling
    min_p = args.min_p
    repetition_penalty = args.repetition_penalty
    max_tokens = args.max_tokens

    output_dir = f'./output/rag/{model_name}' 
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

    case_list = []
    with open(precedent_database_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            case_list.append(json.loads(line))
    
    def retrieved_label_option_fewshot(fact, criminals, in_context_content):
        
        #生成few-shot示例
        examples = in_context_content
        texts = []
        for caseid in examples:
            case = case_list[caseid-1]
            #对examples中的case提取出fact和三个label，并组合为text
            cur_fact = case["fact"].replace(" ", "")
            defendants = case["defendants"]
            # 每个罪犯
            all_str = ""
            for idx,item in enumerate(case["outcomes"]):
                defendant = defendants[idx]
                each_defendant = "罪犯" + defendant
                # 多罪处理：一个被告可能有多个 judgment
                charge_list = []
                article_list = []
                for judgment in item['judgment']:
                    charge = judgment['standard_accusation'].strip()
                    articles = judgment['article']
                    charge_list.append(charge)
                    article_list.extend(articles)
                for article in article_list:
                    article = "第" + article + "条"
                charge_str = ",".join(charge_list)
                article_str = ",".join(article_list)
                each_defendant += ": 法条是: " + article_str + "；罪名是: " + charge_str + "。"
                all_str += each_defendant
                # print(all_str)

            text = "本案的事实是：" + f"{cur_fact}\n" + all_str + "\n"
            texts.append(text)
        examples = "\n\n".join(texts)
        prompt = (
            "你要做的任务是相关法条预测和罪名预测。\n"
            f"本案事实：\n{fact}\n"
            f"与本案案情类似的5个案件以及各罪犯相应的法条和罪名：\n{examples}\n"
            f"请根据你自己的知识，并分析当前案件与类案的相似性，为{criminals}中每个罪犯预测相关法条和罪名。\n"
            "输出格式为:【罪犯1:法条:……,罪名:……】【罪犯2:……】……"
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


    def get_response(data, retrieved_precedents):
        assert len(data) == len(retrieved_precedents)
        response_function = qwen_response

        truncated_data = []
        #data里面包括多个case
        for i in tqdm(range(len(data)), desc="pre-process prompt"):
            #单独处理每个fact
            case = data[i]
            fact = case["fact"].replace(" ", "") # raw fact
            criminal = case["defendants"]
            criminal = ",".join(criminal)
            prompt = retrieved_label_option_fewshot(fact, criminal, retrieved_precedents[i])
            #prompt = truncate_prompt(prompt, tokenizer, 8192)

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
            for idx, line in enumerate(f.readlines()):
                case = json.loads(line)
                data.append(case)
        print("to run data count: ", len(data))
        # data = data[:1]
        # load precedents (全部内容)
        with open('./rag/retrieved_precedents_5.json', 'r', encoding='utf-8') as f:
            retrieved_precedents = json.load(f)
        # retrieved_precedents = retrieved_precedents[:1]
        get_response(data, retrieved_precedents)


    print("已加载好模型！开始生成！")
    run_llm()
    print("Process completed.")

if __name__ == "__main__":
    main()