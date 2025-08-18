import json
import sys
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import json

# 汉字数字与阿拉伯数字的映射表
chinese_to_num = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100
}

# 定义函数，将中文数字转换为阿拉伯数字
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

# 定义函数，用于提取法条中的数字（包括中文和阿拉伯数字）
def extract_article_number(article_text):
    # 查找所有匹配的“第*条”
    matches = re.findall(r'第(\d+|[一二三四五六七八九十百零]+)条', article_text)    
    if not matches:
        return None
    results = []
    for digits in matches:
        if digits.isdigit():
            results.append(int(digits))
        else:
            results.append(chinese_to_arabic(digits))
    if len(results) == 1:
        return results[0]
    return results

def extract_answer(text,id):
    # 使用正则表达式匹配所有【】中的内容
    matches = re.findall(r'【(.*?)】', text)
    
    # 如果没有匹配到任何【】，返回空字符串
    if not matches:
        #print(f"没有匹配到{id}")
        return ""
    
    # 优先选择包含“相关法条是”的【】
    for match in matches:
        if "相关法条是" in match and "*" not in match and "X" not in match:
            return match
    return ""

# print(extract_answer("本案的相关法条是：第280条，本案的罪名是：【伪造、变造、买卖】国家机关【公文、证件、印章】罪。【本案的相关法条是：第280条，本案的罪名是：[伪造、变造、买卖]国家机关[公文、证件、印章]罪。】"))

# 从文件中提取每个案例的法条数字
def extract_article_numbers_from_json(file_path):
    article_numbers = []
    case_ids = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx,line in enumerate(f):
            case = json.loads(line)
            id = case.get('caseID','')
            article_text = case.get('predict', '')
            article_text = extract_answer(article_text,id)
            if article_text == "":
                article_numbers.append(0)
                case_ids.append(id)
                print(f"没有提取到答案：{id}")
            else:
                #print(id)
                article_number = extract_article_number(article_text)
                if article_number is not None:
                    article_numbers.append(article_number)
                    case_ids.append(id)
                else:
                    article_numbers.append(0)
                    case_ids.append(id)
                    print(f"没有提取到法条：{id}")
    return article_numbers,case_ids
# print(extract_article_number("第111条，第112条"))
# print(extract_article_number("第一百一十一条"))

json_file = './output/rag/32b/generated_result'  # 替换为你的文件路径

# 提取法条数字并打印
article_numbers,case_ids = extract_article_numbers_from_json(json_file)

# 不同的task需要更改
print("生成预测结果！开始计算指标啦！")
#all_preds = []
all_preds = article_numbers
all_labels = []
all_ids = []


with open('./data/test.json', 'r',encoding="utf-8") as f:
    for idx, line in enumerate(f.readlines()):
        case = json.loads(line)
        case_id = case['caseID']
        if case_id in case_ids:
            all_labels.append(case["meta"]["relevant_articles"][0])
            all_ids.append(case["caseID"])

#all_labels = all_labels[:10000]
print("labels的长度：",len(all_labels))
#print(all_labels)
print("preds的长度：",len(all_preds))
#print(all_preds)

for idx,pred in enumerate(all_preds):
    if isinstance(pred,list):
        if all_labels[idx] in pred:
            all_preds[idx] = all_labels[idx]
        else:
            all_preds[idx] = pred[0]

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# output_file = './output/rag/32b/article_metrics.txt'
# with open(output_file, 'a+', encoding='utf-8') as f:
#     f.write(f"Accuracy: {accuracy:.4f}\n")
#     f.write(f"Precision: {precision:.4f}\n")
#     f.write(f"Recall: {recall:.4f}\n")
#     f.write(f"F1 Score: {f1:.4f}\n")

print(f"指标已保存至 {output_file}")
