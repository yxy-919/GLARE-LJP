import json
import numpy as np
import faiss 
from transformers import AutoModel, AutoTokenizer
import torch

# 使用预训练的句子编码器，将文本转换为向量表示
tokenizer = AutoTokenizer.from_pretrained("./model/SAILER_zh")
model = AutoModel.from_pretrained("./model/SAILER_zh")

def load_json(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            data_list.append(json.loads(line))
    return data_list

cail_data_path = './data/CAIL/test.json'
precedent_database_path = './data/precedent_database.jsonl'

cail_data = load_json(cail_data_path)
precedent_database = load_json(precedent_database_path)
print("已加载数据！")
# 将文本转换为向量表示的函数
def encode_texts(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # 使用[CLS]向量
    return embeddings.numpy()

# 将类案数据库中的事实描述按批次转换为向量表示
def encode_precedent_cases_in_batches(precedent_database, batch_size=64):
    index = faiss.IndexFlatL2(768)  
    all_case_ids = []  

    for i in range(0, len(precedent_database), batch_size):
        batch_cases = precedent_database[i:i + batch_size]  
        case_facts = [' '.join(case['fact']) for case in batch_cases] 
        vectors = encode_texts(case_facts).astype('float32')  
        index.add(vectors)  
        
        # 保存当前批次的案例ID
        all_case_ids.extend([case['caseID'] for case in batch_cases])

    return index, all_case_ids  

def save_faiss_index(index, index_file):
    faiss.write_index(index, index_file)
    print(f"FAISS索引已保存到 {index_file}")

def load_faiss_index(index_file):
    index = faiss.read_index(index_file)
    print(f"FAISS索引已从 {index_file} 加载")
    return index

# 根据候选法条，从precedent_database中检索与每个法条最相似的类案
def find_most_similar_case_for_labels(case_fact, index, precedent_database_ids, precedent_vectors):
    selected_cases = []  

    case_vector = encode_texts([case_fact]).astype('float32')
 
    D, I = index.search(case_vector, k=5)

    for idx in I[0]:
        selected_cases.append(precedent_database_ids[idx]) 
    return selected_cases

# 主函数：对每个案件进行类案检索
def retrieve_precedents_for_cases(cail_data, index, precedent_database_ids):
    all_selected_cases = []  

    for case in cail_data:
        case_fact = ' '.join(case['fact'])  
        selected_cases = find_most_similar_case_for_labels(case_fact, index, precedent_database_ids, None)
        all_selected_cases.append(selected_cases)  

    return all_selected_cases

# print("开始检索！")
# # 执行类案检索并保存结果
# faiss_index, precedent_database_ids = encode_precedent_cases_in_batches(precedent_database)  # 按批次编码类案并获得索引
# print("索引构建完成")
# index_file = "./output/rag/index.faiss"
# save_faiss_index(faiss_index, index_file)
faiss_index = load_faiss_index("./output/rag/index.faiss")
precedent_database_ids = []
for case in precedent_database:
    precedent_database_ids.append(case['caseID'])
retrieved_precedents = retrieve_precedents_for_cases(cail_data, faiss_index, precedent_database_ids)

# 将结果保存为json文件
output_path = './output/rag/retrieved_precedents_5.json'
with open(output_path, 'w', encoding='utf-8') as outfile:
    json.dump(retrieved_precedents, outfile, ensure_ascii=False, indent=4)

print("检索完成，结果已保存至", output_path)
