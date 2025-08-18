# 1、选取1000条
import json
import random
from collections import defaultdict

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def stratified_sample(data, total_num=1000, seed=42):
    random.seed(seed)
    
    # 1. 根据第一个被告的罪名分类
    label2cases = defaultdict(list)
    for item in data:
        try:
            label = item["outcomes"][0]["judgment"][0]["accusation"]
            label2cases[label].append(item)
        except Exception as e:
            continue  # 若结构不完整则跳过

    # 2. 统计种类数
    all_labels = list(label2cases.keys())
    num_labels = len(all_labels)
    per_label_num = total_num // num_labels

    print(f"共计罪名种类数：{len(label2cases)}")
    print(f"罪名分布如下：")
    for label, cases in label2cases.items():
        print(f"- {label}: {len(cases)} 个案件")

    selected_cases = []
    remaining_cases = []

    # 3. 对每类均匀采样
    for label in all_labels:
        cases = label2cases[label]
        if len(cases) >= per_label_num:
            sampled = random.sample(cases, per_label_num)
            selected_cases.extend(sampled)
            remaining_cases.extend([case for case in cases if case not in sampled])
        else:
            # 不足的也全部选上，剩下的数量后续补
            sampled = cases
            selected_cases.extend(sampled)

    # 4. 如果总数不足，从剩余样本中补齐
    current_len = len(selected_cases)
    if current_len < total_num:
        needed = total_num - current_len
        all_remaining = [case for case in data if case not in selected_cases]
        supplement = random.sample(all_remaining, min(needed, len(all_remaining)))
        selected_cases.extend(supplement)

    # 5. 打乱顺序
    random.shuffle(selected_cases)
    return selected_cases

def save_jsonl(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 使用示例
data = load_jsonl("./test_small.json")
sampled = stratified_sample(data, total_num=1000)
save_jsonl(sampled, "test1000.json")

# # 2、法条编号处理
# import json
# import re

# def clean_article_code(article_code):
#     """
#     保留 article 中的整数部分：如 234-1 → 234, 162.2 → 162
#     """
#     return re.split(r'[-\.]', article_code)[0]

# def process_json_line(json_obj):
#     # 删除 court_view 字段
#     json_obj.pop("court_view", None)

#     # 遍历所有被告的 judgment 中的 article 字段
#     for outcome in json_obj.get("outcomes", []):
#         for judgment in outcome.get("judgment", []):
#             if "article" in judgment:
#                 cleaned_articles = [clean_article_code(a) for a in judgment["article"]]
#                 judgment["article"] = cleaned_articles

#     return json_obj

# def process_jsonl_file(input_path, output_path):
#     with open(input_path, 'r', encoding='utf-8') as fin, \
#          open(output_path, 'w', encoding='utf-8') as fout:
#         for line in fin:
#             if line.strip():  # 跳过空行
#                 data = json.loads(line)
#                 cleaned = process_json_line(data)
#                 fout.write(json.dumps(cleaned, ensure_ascii=False) + '\n')

# # ✅ 文件路径设置
# input_file = "./data/valid_old.json"              # 输入原始文件名
# output_file = "./data/valid.json"    # 输出新文件名

# # ✅ 执行处理
# process_jsonl_file(input_file, output_file)
# print(f"处理完成，已保存为：{output_file}")

# # 3、增加caseID
# # import json

# def add_case_id(input_file, output_file):
#     with open(input_file, 'r', encoding='utf-8') as fin, \
#          open(output_file, 'w', encoding='utf-8') as fout:
#         for idx, line in enumerate(fin, start=1):
#             if not line.strip():
#                 continue
#             case = json.loads(line)
#             case["caseID"] = idx
#             fout.write(json.dumps(case, ensure_ascii=False) + '\n')

# # 输入输出文件名
# input_file = "./data/CMDL/train1.json"
# output_file = "./data/CMDL/train.json"

# add_case_id(input_file, output_file)
# print(f"处理完成，已保存至 {output_file}")


# 4、article2id处理
# import json
# data = {"211": 0, "275": 1, "258": 2, "212": 3, "185": 4, "418": 5, "233": 6, "330": 7, "358": 8, "162.2": 9, "238": 10, "209": 11, "219": 12, "292": 13, "205.1": 14, "341": 15, "289": 16, "288": 17, "375": 18, "397": 19, "309": 20, "342": 21, "286": 22, "333": 23, "139": 24, "267": 25, "382": 26, "261": 27, "184": 28, "348": 29, "314": 30, "125": 31, "274": 32, "338": 33, "296": 34, "351": 35, "204": 36, "222": 37, "217": 38, "285": 39, "308": 40, "262": 41, "277": 42, "177": 43, "287.1": 44, "186": 45, "215": 46, "391": 47, "360": 48, "117": 49, "158": 50, "218": 51, "354": 52, "188": 53, "383": 54, "347": 55, "197": 56, "312": 57, "390": 58, "385": 59, "372": 60, "322": 61, "326": 62, "146": 63, "363": 64, "417": 65, "328": 66, "148": 67, "316": 68, "182": 69, "118": 70, "130": 71, "305": 72, "210.1": 73, "369": 74, "282": 75, "201": 76, "400": 77, "366": 78, "359": 79, "128": 80, "262.2": 81, "135": 82, "283": 83, "396": 84, "345": 85, "264": 86, "153": 87, "413": 88, "205": 89, "293": 90, "200": 91, "120.3": 92, "268": 93, "228": 94, "151": 95, "321": 96, "240": 97, "337": 98, "172": 99, "319": 100, "280.1": 101, "252": 102, "361": 103, "171": 104, "168": 105, "301": 106, "150": 107, "279": 108, "164": 109, "161": 110, "139.1": 111, "237": 112, "162.1": 113, "253.1": 114, "387": 115, "167": 116, "224.1": 117, "124": 118, "207": 119, "176": 120, "143": 121, "398": 122, "320": 123, "273": 124, "365": 125, "152": 126, "234.1": 127, "318": 128, "206": 129, "196": 130, "307.1": 131, "214": 132, "307": 133, "208": 134, "177.1": 135, "310": 136, "280": 137, "149": 138, "324": 139, "147": 140, "349": 141, "290": 142, "284.1": 143, "119": 144, "243": 145, "293.1": 146, "271": 147, "300": 148, "402": 149, "343": 150, "352": 151, "227": 152, "192": 153, "291": 154, "144": 155, "137": 156, "221": 157, "156": 158, "302": 159, "232": 160, "272": 161, "336": 162, "291.1": 163, "294": 164, "356": 165, "263": 166, "138": 167, "246": 168, "225": 169, "220": 170, "276": 171, "353": 172, "245": 173, "281": 174, "344": 175, "287.2": 176, "223": 177, "140": 178, "115": 179, "315": 180, "193": 181, "244": 182, "136": 183, "364": 184, "306": 185, "170": 186, "239": 187, "155": 188, "116": 189, "229": 190, "269": 191, "390.1": 192, "180": 193, "183": 194, "395": 195, "389": 196, "159": 197, "190": 198, "313": 199, "266": 200, "340": 201, "234": 202, "134": 203, "399": 204, "393": 205, "247": 206, "226": 207, "408": 208, "303": 209, "213": 210, "236": 211, "133": 212, "276.1": 213, "295": 214, "270": 215, "384": 216, "241": 217, "350": 218, "191": 219, "141": 220, "133.1": 221, "392": 222, "194": 223, "114": 224, "127": 225, "388": 226, "175.1": 227, "256": 228, "145": 229, "231": 230, "175": 231, "198": 232, "235": 233, "408.1": 234, "163": 235, "260": 236, "386": 237, "224": 238}

# # 1. 去掉 key 中包含小数点的项
# filtered = {k: v for k, v in data.items() if '.' not in k}

# # 2. 对 key 按照数字大小排序，并重新赋值
# sorted_keys = sorted(filtered.keys(), key=lambda x: int(x))
# new_dict = {k: i for i, k in enumerate(sorted_keys)}

# with open("article2id", "a+", encoding="utf-8") as f:
#     json.dump(new_dict, f, ensure_ascii=False, indent=2)


#  5、统计1000条case中所有被告罪名都相同的数目
# import json

# def find_cases_with_different_accusations(file_path):
#     different_cases = []
    
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             try:
#                 data = json.loads(line.strip())
#                 case_id = data.get('caseID')
#                 defendants = data.get('defendants', [])
#                 outcomes = data.get('outcomes', [])
                
#                 if not case_id or not defendants or not outcomes:
#                     continue
                
#                 # Get all standard accusations for defendants
#                 standard_accusations = []
#                 for outcome in outcomes:
#                     for judgment in outcome.get('judgment', []):
#                         standard_accusations.append(judgment.get('standard_accusation'))
                
#                 # Check if accusations are different
#                 if len(set(standard_accusations)) > 1:
#                     different_cases.append({
#                         'caseID': case_id,
#                         'defendants': defendants,
#                         'accusations': standard_accusations
#                     })
                    
#             except json.JSONDecodeError:
#                 continue
    
#     return different_cases

# file_path = './data/test.json' 
# different_cases = find_cases_with_different_accusations(file_path)

# print(f"Number of cases with different standard accusations: {len(different_cases)}")
# print("\nDetails of these cases:")
# for case in different_cases:
#     print(f"\nCase ID: {case['caseID']}")
#     print("Defendants:", ", ".join(case['defendants']))
#     print("Accusations:", ", ".join(case['accusations']))