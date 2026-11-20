from evaluation_io import METRICS_DIR, parse_evaluation_args, save_metrics

args = parse_evaluation_args("Evaluate multi-defendant predictions.")

import math
import torch
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 这里提供一种基于 预测结果和标签的one-hot vector，及被告人数量数组 计算case-level指标的方法

def get_weight(defendant_num):
    """
        给定案件被告人人数，计算基础权重
    Args:
        defendant_num (_type_): 案件被告人数量
    Returns:
        float: 权重值，计算方式为log2(D), D为被告人数量
    """
    if isinstance(defendant_num, torch.Tensor):
        # defendant_num为tensor类型
        return math.log(defendant_num.item(), 2)
    else:
        # defendant_num为直接的数值类型
        return math.log(defendant_num, 2)


def get_per_defendant_metrics_weight(defendant_num):
    """
        计算最终结果中每个被告人对应的权重，是get_weight(defendant_num)/defendant_num
    Args:
        defendant_num (_type_): 案件被告人数量
    Returns:
        _type_: _description_
    """
    if isinstance(defendant_num, torch.Tensor):
        assert defendant_num.item() != 0
        return get_weight(defendant_num.item()) / defendant_num.item()
    else:
        assert defendant_num != 0
        return get_weight(defendant_num) / defendant_num


def get_weighted_per_defendant_metrics(outputs, labels, weight_tensor):
    """
        通过计算per-defendant指标计算per-case指标后进行加权
    Args:
        outputs (_type_): 模型预测
        labels (_type_): 真实标签
        weight_tensor (_type_): 这里的每个weight为get_weight(defendant_num)/defendant_num
    Returns:
        acc, precision, recall, f1: 加权后的最终指标
    """
    print(outputs.shape)
    print(labels.shape)
    assert outputs.shape == labels.shape
    assert outputs.size(0) == weight_tensor.size(0)
    acc, precision, recall, f1 = 0, 0, 0, 0
    sample_num = outputs.size(0)
    # 加权平均的分母
    weight_total = torch.sum(weight_tensor).item()
    TPs = torch.sum(outputs * labels, dim=1)
    FPs = torch.sum(outputs * (1 - labels), dim=1)
    FNs = torch.sum((1 - outputs) * labels, dim=1)
    TNs = torch.sum((1 - outputs) * (1 - labels), dim=1)
    precisions = TPs / torch.clamp(TPs + FPs, min=1e-8)
    recalls = TPs / torch.clamp(TPs + FNs, min=1e-8)
    f1s = 2 * precisions * recalls / torch.clamp(precisions + recalls, min=1e-8)
    # exact match accuracy
    acc = torch.sum(weight_tensor * torch.all(outputs == labels, dim=1).float()) / weight_total
    precision = torch.sum(weight_tensor * precisions) / weight_total
    recall = torch.sum(weight_tensor * recalls) / weight_total
    f1 = torch.sum(weight_tensor * f1s) / weight_total
    return acc.item(), precision.item(), recall.item(), f1.item()


def get_case_level_metrics_by_per_defendant_metrics(outputs, labels, defendant_nums):
    """
        计算case-level metrics
    Args:
        outputs (torch.Tensor): size(batch_size, num_labels)，每个元素为0或1
        labels (torch.Tensor): size(batch_size, num_labels)，每个元素为0或1
        defendant_nums (List): len(batch_size) 每条输入对应案件的被告人数量数组
    Returns:
        三个任务的结果
    """
    # 对每个被告人计算accuracy, precision, recall, f1
    weight_tensor = torch.tensor(list(map(get_per_defendant_metrics_weight, defendant_nums))).to(outputs["charge"].device)
    charge_acc, charge_precision, charge_recall, charge_f1 = get_weighted_per_defendant_metrics(outputs["charge"], labels["charge"], weight_tensor)
    article_acc, article_precision, article_recall, article_f1 = get_weighted_per_defendant_metrics(outputs["article"], labels["article"], weight_tensor)
    # penalty_acc, penalty_precision, penalty_recall, penalty_f1 = get_weighted_per_defendant_metrics(outputs["penalty"], labels["penalty"], weight_tensor)
    result = {
        "case_level_Acc_Acc": charge_acc,
        "case_level_Acc_Precision": charge_precision,
        "case_level_Acc_Recall": charge_recall,
        "case_level_Acc_F1": charge_f1,
        "case_level_Art_Acc": article_acc,
        "case_level_Art_Precision": article_precision,
        "case_level_Art_Recall": article_recall,
        "case_level_Art_F1": article_f1,
        # "case_level_Pen_Acc": penalty_acc,
        # "case_level_Pen_Precision": penalty_precision,
        # "case_level_Pen_Recall": penalty_recall,
        # "case_level_Pen_F1": penalty_f1
    }
    print(result)
    return result


def calculate_final_score(result):
    """
        计算最终得分
    Args:
        result (dict): 通过上述方法得到的三个任务的case-level结果；最终分数为
            0.3 * charge_f1 + 0.3 * article_f1 + 0.4 * penalty_acc
    Returns:
        float: 最终得分
    """
    # 三个任务的结果
    charge_f1 = result["case_level_Acc_F1"]
    article_f1 = result["case_level_Art_F1"]
    penalty_acc = result["case_level_Pen_Acc"]
    return 0.3 * charge_f1 + 0.3 * article_f1 + 0.4 * penalty_acc

with open(METRICS_DIR / "a2i.json", "r", encoding="utf-8") as f:
    a2i = json.load(f)

with open(METRICS_DIR / "c2i.json", "r", encoding="utf-8") as f:
    c2i = json.load(f)

chinese_to_num = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100
}

def chinese_to_arabic(chinese_digits):
    result = 0
    unit = 1
    temp = 0
    for char in reversed(chinese_digits):
        if char in chinese_to_num:
            num = chinese_to_num[char]
            if num >= 10:
                unit = num  # 百、十等单位
            else:
                temp = num * unit
                result += temp
        else:
            continue
    return result

# 提取 article 向量
def extract_articles(article_text):
    nums = re.findall(r'第(\d+|[一二三四五六七八九十百零]+)条', article_text)
    # print(nums)
    nums_new = []
    for num in nums:
        if num.isdigit():
            nums_new.append(int(num))
        else:
            nums_new.append(chinese_to_arabic(num))
    # print(nums_new)
    onehot = [0] * len(a2i)
    for num in nums_new:
        if str(num) in a2i:
            onehot[a2i[str(num)]] = 1
            # print("aaa")
    return onehot
# print(extract_articles("罪犯1: 荣某某: 法条是: 第三百五十九条第一款； 罪名是: 介绍卖淫罪"))

model = SentenceTransformer(args.bge_model_path)

crime_names = list(c2i.keys())
def encode_all_crimes(bge_model,crime_names,crime_encodings_file):
    embeddings = bge_model.encode(crimes_names)
    np.save(crime_encodings_file, embeddings)

embeddings = np.load(METRICS_DIR / 'crime_encodings.npy')

def bge_match(charge,bge_model,embeddings,crime_names):
    new_embedding = bge_model.encode([charge])
    similarities = cosine_similarity(new_embedding,embeddings)
    topk_indices = np.argsort(similarities[0])[-1:][::-1]
    topk_charges = [crime_names[idx] for idx in topk_indices]
    return topk_charges[0]
# print(bge_match('虚假诉讼',model,embeddings,crime_names))

def extract_charges(charge_text):
    match = re.search(r"罪名:(.+?)】", charge_text)
    if match:
        charge = match.group(1)
        if "，" in charge:
            charges = charge.split("，")
        else:
            charges = [charge]
        # print(charges)
        charges_new = []
        for charge in charges:
            charges_new.append(bge_match(charge,model,embeddings,crime_names))
        # print(charges_new)
        onehot = [0] * len(c2i)
        for charge in charges_new:
            if charge in c2i:
                onehot[c2i[charge]] = 1
        return onehot
    else:
        print(charge_text)
        onehot = [0] * len(c2i)
        return onehot

# print(extract_charges("【罪犯2:黄某B法条是: 第三百九十三条（单位行贿罪）、第二十七条（从犯）。  \n罪名是: 单位行贿罪。】"))

def convert_labels_to_onehot(label_jsonl_path):
    charge_list = []
    article_list = []
    num_list2 = []

    with open(label_jsonl_path, 'r', encoding='utf-8') as f:
        for idx,line in enumerate(f):
            data = json.loads(line)
            outcomes = data['outcomes']
            num_list2.append(len(outcomes))

            for item in outcomes:
                charge_onehot = [0] * len(c2i)
                article_onehot = [0] * len(a2i)

                # 多罪处理：一个被告可能有多个 judgment
                for judgment in item['judgment']:
                    charge = judgment['standard_accusation'].strip()
                    articles = judgment['article']

                    if charge in c2i:
                        charge_onehot[c2i[charge]] = 1
                    for art in articles:
                        if art in a2i:
                            article_onehot[a2i[art]] = 1

                charge_list.append(charge_onehot)
                article_list.append(article_onehot)

    charge_label_tensor = torch.tensor(charge_list)
    article_label_tensor = torch.tensor(article_list)
    return charge_label_tensor, article_label_tensor, num_list2
charge_label, article_label, num_list2 = convert_labels_to_onehot(args.label_file_path)

def convert_predictions_to_onehot(pred_file_path):
    article_tensor, charge_tensor = [], []
    defendant_nums = []
    num_list1 = []

    with open(pred_file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = json.loads(line)
            predict = line["predict"]
            defendants = re.findall(r'【(.*?)】', predict, re.DOTALL)

            pred_num = len(defendants)
            gold_num = num_list2[idx]

            for defendant in defendants:
                content = '【' + defendant + '】'
                article_tensor.append(extract_articles(content))
                charge_tensor.append(extract_charges(content))

            # 如果预测少于真实标签，则补上（用最后一个被告人复制）
            if pred_num < gold_num and pred_num > 0:
                last_defendant = defendants[-1]
                content = '【' + last_defendant + '】'
                for _ in range(gold_num - pred_num):
                    article_tensor.append(extract_articles(content))
                    charge_tensor.append(extract_charges(content))
                pred_num = gold_num
            num_list1.append(pred_num)

            defendant_nums.extend([gold_num] * gold_num)

    article_tensor = torch.tensor(article_tensor)
    charge_tensor = torch.tensor(charge_tensor)
    return article_tensor, charge_tensor, defendant_nums, num_list1

# convert_predictions_to_onehot('./output/llm/32b/generated_result')


def cal_metrics(pred_file_path):
    article_pred, charge_pred, defendant_nums_list, num_list1 = convert_predictions_to_onehot(pred_file_path)
    for idx,num in enumerate(num_list1):
        if num != num_list2[idx]:
            print(idx)

    outputs = {
        "article": article_pred,
        "charge": charge_pred
    }
    labels = {
        "article": article_label,
        "charge": charge_label
    }
    result = get_case_level_metrics_by_per_defendant_metrics(outputs, labels, defendant_nums_list)

    print("charge:")
    print("Acc:",result["case_level_Acc_Acc"])
    print("Precision:",result["case_level_Acc_Precision"])
    print("Recall:",result["case_level_Acc_Recall"])
    print("F1:",result["case_level_Acc_F1"])
    print("article:")
    print("Acc:",result["case_level_Art_Acc"])
    print("Precision:",result["case_level_Art_Precision"])
    print("Recall:",result["case_level_Art_Recall"])
    print("F1:",result["case_level_Art_F1"])
    save_metrics(result, args.output_file)

cal_metrics(args.pred_file_path)
