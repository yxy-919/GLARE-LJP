import json
from tqdm import tqdm
import sys
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
from expand_crimes import expand_crimes
from sentence_transformers import SentenceTransformer

def main():
    # load qwq model
    tokenizer = AutoTokenizer.from_pretrained("./model/Qwen2.5-72B-Instruct")
    llm = LLM(model="./model/Qwen2.5-72B-Instruct",enforce_eager=True,trust_remote_code=True,gpu_memory_utilization=0.8,tensor_parallel_size=4,dtype = torch.bfloat16)
    # load bge model
    bge_model = SentenceTransformer("./model/BAAI_bge-m3")
    print("已加载好模型！开始生成！")
    
    # load data
    data = []
    with open("./data/CAIL/test.json", 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f.readlines()):
            case = json.loads(line)
            data.append(case)
    # data = data[:5]
    # precedents
    with open('./output/rag/retrieved_precedents_5.json', 'r', encoding='utf-8') as f:
        retrieved_precedents = json.load(f)

    # precedent database
    case_list = []
    with open('./data/CAIL/train.json', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            case_list.append(json.loads(line))
    precedent_fact = []
    prompt_list = []
    for precedents in retrieved_precedents:
        for idx,precedent in enumerate(precedents):
            # reasoning_chain += f"第{idx}个类案：\n"
            case = case_list[precedent-1]
            fact = case["fact"].replace(" ", "")
            precedent_fact.append(fact)
            defendants = case["defendants"]
            # 每个罪犯
            all_str = ""
            all_charge_list = []
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
                all_charge_list.extend(charge_list)
                article_str = ",".join(article_list)
                each_defendant += ": 法条是: " + article_str + "；罪名是: " + charge_str + "。"
                all_str += each_defendant
                # print(all_str)
            # expand the charge
            expanded_crimes = expand_crimes(all_charge_list,bge_model,3)
            expanded_crimes = list(set(expanded_crimes))
            expanded_crimes_str = ', '.join(expanded_crimes)
            prompt = (
                "给定一段法律案件的案情描述以及正确的法条和罪名，并给出其他的罪名，分析选择正确罪名排除其他罪名的理由。\n"
                f"案情描述：\n{fact}\n"
                f"正确的法条和罪名：\n{all_str}\n"
                f"其他的罪名：\n{expanded_crimes_str}\n"
            )
            prompt_list.append(prompt)
    print(len(prompt_list))
    
    prompts = [{"role":"user", "content":prompt} for prompt in prompt_list]
    prompts = [tokenizer.apply_chat_template([prompt], tokenize=False, add_generation_prompt=True) for prompt in prompts]

    # generate
    output = llm.generate(
        prompts,
        sampling_params = SamplingParams(
            temperature=0.3, 
            top_p=0.95,
            top_k=20, 
            min_p=0, 
            repetition_penalty=1.1, 
            max_tokens=32768
        )
    )
    outputs = [out.outputs[0].text for out in output]

    for i in range(len(outputs)):
        fact = precedent_fact[i]
        outputs[i] = f"案情描述：\n{fact}\n" + outputs[i] 

    reasoning_chain_list = []
    i = 1
    for output in outputs:
        if i == 1:
            reasoning_chain = ""
        reasoning_chain += f"第{i}个类案的分析：\n" + output + "\n" 
        # 提取 </think> 之后的内容
        # if "</think>" in output:
        #     analysis_part = output.split("</think>")[1].strip()
        # else:
        #     analysis_part = output  # 如果没有 </think>，保留全部内容
        # reasoning_chain += f"第{i}个类案的分析：\n" + analysis_part + "\n"       
        i += 1
        if i == 4:
            i = 1
            reasoning_chain_list.append(reasoning_chain)

    # save
    res_list = []
    for i, reasoning_chain in enumerate(reasoning_chain_list):
        res = {}
        res["caseID"] = data[i]["caseID"]
        res["reasoning_chain"] = reasoning_chain
        res_list.append(res)

    with open('./output/rag/reasoning_chain.json', "a+", encoding="utf-8") as f:
        for res in res_list:
            line = json.dumps(res, ensure_ascii=False)
            f.write(line + "\n")
    print("生成完成！")

if __name__ == "__main__":
    main()


