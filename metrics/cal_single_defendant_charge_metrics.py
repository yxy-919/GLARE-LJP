import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
#from transformers import AutoTokenizer
from modelscope import AutoTokenizer
from vllm import LLM, SamplingParams
from mycode.utils import loader
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 加载 BGE 编码模型
model = SentenceTransformer("./model/BAAI_bge-m3")

# 读取编码结果
embeddings = np.load("./metrics/crime_encodings.npy")
with open("./metrics/c2i.json", "r", encoding="utf-8") as f:
    crime_dict = json.load(f)

# 提取罪名
crime_names = list(crime_dict.keys())

case_list = []
with open('./output/rag/32b/generated_result', encoding="utf-8") as f:
    for line in f.readlines():
        case = json.loads(line)
        case_list.append(case)

# 加载映射关系的 JSON 文件
mapping_file = "./metrics/expanded_charges_mapping.json"  # 替换为你的映射文件路径
with open(mapping_file, 'r', encoding='utf-8') as f:
    # 将文件中的每一行加载为 JSON 对象，并合并为一个字典
    mapping_dict = {}
    for line in f:
        if line.strip():  # 确保行不为空
            mapping = json.loads(line)
            mapping_dict.update(mapping)

def extract_answer(text,id):
    # 使用正则表达式匹配所有【】中的内容
    matches = re.findall(r'【(.*?)】', text)
    
    # 如果没有匹配到任何【】，返回空字符串
    if not matches:
        print(f"没有匹配到{id}")
        return ""
    
    # 优先选择包含“相关法条是”的【】
    for match in matches:
        if "相关法条是" in match and "*" not in match and "X" not in match:
            return match
    return ""


charge_list = []
case_ids = []
for case in case_list:
    charge_text = case["predict"]
    case_id = case["caseID"]
    charge_text = extract_answer(charge_text,case_id)
    if charge_text == "":
        charge_list.append("未找到")
        case_ids.append(case_id)
        print(f"没有提取到答案：{case_id}")
    #match = re.search(r"本案的罪名是：(.+?)。", charge_text)
    else:
        match = re.search(r"本案的罪名是：(.+?)(?:。|$)", charge_text)
        if match:
            # 提取罪名
            charge = match.group(1)
            # 检查罪名是否包含“和”字
            if "和" in charge:
                # 使用“和”字拆分罪名
                charges = charge.split("和")
            else:
                # 如果不包含“和”，则只将原罪名作为一个列表处理
                charges = [charge]
            
            # 处理每个罪名
            mapped_charges = []
            for ch in charges:
                #print(f"未处理前的罪名：{ch}\n")
                # 去掉结尾的“罪”字
                ch = re.sub(r"罪$", "", ch).strip()
                # new_embedding = model.encode([ch])
                # similarities = cosine_similarity(new_embedding, embeddings)
                # max_index = np.argmax(similarities)
                # mapped_charge = crime_names[max_index]
                # 查找映射关系，如果没有对应的，使用原罪名
                mapped_charge = mapping_dict.get(ch, ch)
                #if mapped_charge == ch:
                if mapped_charge not in crime_names:
                    new_embedding = model.encode([mapped_charge])
                    similarities = cosine_similarity(new_embedding, embeddings)
                    max_index = np.argmax(similarities)
                    mapped_charge = crime_names[max_index]
                #print(f"处理后的罪名：{mapped_charge}\n\n")

                mapped_charges.append(mapped_charge)
            
            # 将映射后的罪名列表和对应的案件ID添加到结果中
            charge_list.append(mapped_charges)
            case_ids.append(case_id)
        else:
            charge_list.append("未找到")
            case_ids.append(case_id)
            print(f"没有提取到罪名：{case_id}")

print("preds的长度：",len(charge_list))
labels = []
with open('./data/test.json',encoding='utf-8') as f:
    for line in f.readlines():
        case = json.loads(line)
        case_id = case['caseID']
        if case_id in case_ids:
            labels.append(case["meta"]["accusation"][0])
print("labels的长度：",len(labels))

all_labels = labels
all_preds = charge_list
#print(all_preds)
# 处理一个case有两个预测结果的情况
def process_predictions(all_preds,all_labels):
    final_preds = []
    for id,preds in enumerate(all_preds):
        if(len(preds)==1):
            final_preds.append(preds[0].strip(""))
        else:
            #final_preds.append("None")
            ls = [preds[0].strip(""),preds[1].strip("")]
            if all_labels[id] in ls:
                final_preds.append(all_labels[id])
            else:
                final_preds.append(ls[0])
    return final_preds

# all_preds = [["购买假币"],["购买假币","盗窃"]]
# all_labels = ["本案的罪名是：购买假币罪。","本案的罪名是：盗窃罪。"]
# print(process_predictions(all_preds,all_labels))
all_preds = process_predictions(all_preds,all_labels) 


# 计算并输出指标
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# output_file = './output/rag/32b/charge_metrics.txt'
# with open(output_file, 'a+', encoding='utf-8') as f:
#     f.write(f"Accuracy: {accuracy:.4f}\n")
#     f.write(f"Precision: {precision:.4f}\n")
#     f.write(f"Recall: {recall:.4f}\n")
#     f.write(f"F1 Score: {f1:.4f}\n")

# print(f"指标已保存至 {output_file}")
