import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import sys
import re
from modelscope import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import random
from collections import defaultdict
from functools import lru_cache
from precedents import case_charges, clean_charge

@lru_cache(maxsize=4)
def historical_co_occurrence(multiljp_file_path):
    """
    Build a co-occurrence dictionary of crimes based on multi-defendant cases.
    
    Args:
        multiljp_file_path (str): Path to the JSON file containing case data.
        
    Returns:
        dict: A dictionary where each key is a crime, and the value is another 
              dictionary of co-occurring crimes with their frequencies.
              Format: {crime: {co_occurring_crime1: count1, co_occurring_crime2: count2, ...}}
    """
    # 历史罪名共现关系可以利用 MultiLJP、CMDL 等多被告、多罪名数据集的训练集统计。
    # Cache statistics so each online expansion does not rescan the entire corpus.
    crime_co_occurrence = defaultdict(lambda: defaultdict(int))
    
    with open(multiljp_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            case = json.loads(line.strip())
            crimes_in_case = set(case_charges(case))

            # Update co-occurrence counts (skip single-crime cases)
            if len(crimes_in_case) >= 2:
                crimes_list = list(crimes_in_case)
                for i in range(len(crimes_list)):
                    for j in range(i + 1, len(crimes_list)):
                        crime1, crime2 = crimes_list[i], crimes_list[j]
                        crime_co_occurrence[crime1][crime2] += 1
                        crime_co_occurrence[crime2][crime1] += 1
    
    # Convert defaultdict to regular dict and sort by frequency
    result = {}
    for crime, co_crimes in crime_co_occurrence.items():
        result[crime] = dict(sorted(co_crimes.items(), key=lambda x: x[1], reverse=True))
    
    return result

def expand_historical_co_occurrence(multiljp_file_path, charges, k):
    """
    根据历史共现数据，对输入的每个罪名查找其Top-K共现罪名，合并后去重返回
    
    Args:
        multiljp_file_path (str): 案例数据文件路径
        charges (list): 需要查询的罪名列表，如 ["盗窃罪", "故意伤害罪"]
        k (int): 每个罪名要返回的共现罪名数量
    
    Returns:
        list: 去重后的所有共现罪名列表
    """
    co_occurrence_dict = historical_co_occurrence(multiljp_file_path)
    expanded_charges = set()
    
    for charge in charges:
        if charge in co_occurrence_dict:
            co_occurring = co_occurrence_dict[charge]

            top_k = list(co_occurring.keys())[:k]
            
            expanded_charges.update(top_k)
    
    return list(expanded_charges)


# 按照刑法中的章节结构进行归类
def get_structure():
    result = {}
    with open('./glare/structure_expand/laws.txt', 'r', encoding='utf-8') as f:
        for idx,line in enumerate(f):
            if line == '\n':
                continue
            chapter_pattern = re.compile(r'^第[一二三四五六七八九十]章\s+(.*)')
            section_pattern = re.compile(r'^第[一二三四五六七八九十]节\s+(.*)')
            chapter_match = chapter_pattern.match(line)
            section_match = section_pattern.match(line)
            if chapter_match:
                title = chapter_match.group(1)
                result[title] = []
            elif section_match:
                title = section_match.group(1)
                result[title] = []
            else:
                match = re.search(r'【(.*?)】',line)
                if match:
                    crime_content = match.group(1)
                    # 检查是否有分号
                    if ";" in crime_content:
                        split_crimes = [c.strip() for c in crime_content.split(";")]
                    else:
                        split_crimes = [crime_content.strip()]
                    # 检查每个罪名中是否包含两个“罪”字
                    for crime in split_crimes:
                        if crime.count("罪") != 1:
                            crimes = re.split(r'(?<=罪)、', crime)
                            result[title].extend(crimes)
                        else:
                            result[title].append(crime)
                # if "【" in  line:
                #     crimes = re.findall(r'【(.*?)】',line)
                #     result[title].extend(crimes)
                # else:
                #     pass
    with open('./glare/structure_expand/laws_structure_split','w',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    law_list = []
    for value in result.values():
        if len(value) > 0:
            law_list.append(value)
    with open('./glare/structure_expand/laws_structure_list_split','w',encoding='utf-8') as f:
        json.dump(law_list,f,ensure_ascii=False,indent=2)

def bge_match(charge,bge_model,embeddings,crime_names,topk):
    new_embedding = bge_model.encode([charge])
    similarities = cosine_similarity(new_embedding,embeddings)
    topk_indices = np.argsort(similarities[0])[-topk:][::-1]
    topk_charges = [crime_names[idx] for idx in topk_indices]
    return topk_charges

def process_cail_label():
    with open('./glare/structure_expand/c2i.json','r',encoding='utf-8') as f:
        cail_label_dict = json.load(f)
    cail_label_list = list(cail_label_dict.keys())
    cail_label_list_new = []
    for cail_label in cail_label_list:
        if '[' in cail_label:
            cail_label = cail_label.replace('[','').replace(']','')
            cail_label_list_new.append(cail_label)
        else:
            cail_label_list_new.append(cail_label)
    with open('./glare/structure_expand/cail_label','w',encoding='utf-8') as f:
        json.dump(cail_label_list_new,f,ensure_ascii=False,indent=2)
    

def select_crimes():
    with open('./glare/structure_expand/cail_label','r',encoding='utf-8') as f:
        cail_label = json.load(f)

    with open('./glare/structure_expand/laws_structure_list_split','r',encoding='utf-8') as f:
        laws_structure = json.load(f)
    laws_structure_new = []
    for chapter in laws_structure:
        new_chapter = []
        for law in chapter:
            new_law = law.replace('罪','')
            if new_law in cail_label and new_law not in new_chapter:
                new_chapter.append(new_law)
            elif new_law not in cail_label:
                print(f"没有映射到cail标签的罪名：{new_law}")
            else:
                pass
        laws_structure_new.append(new_chapter)
    with open('./glare/structure_expand/laws_structure_list_split_remove','w',encoding='utf-8') as f:
        json.dump(laws_structure_new,f,ensure_ascii=False,indent=2)

def encode_all_crimes(bge_model):
    with open('./glare/structure_expand/laws_structure_list_split_remove','r',encoding='utf-8') as f:
        laws_structure_list_split_remove = []
    crimes_names = []
    for chapter in laws_structure_list_split_remove:
        crimes_names.extend(chapter)
    embeddings = bge_model.encode(crimes_names)
    np.save('./glare/structure_expand/crime_encodings.npy', embeddings)

def are_crimes_in_same_sublist(crime1,crime2,crime_list):
    """
    判断两个罪名是否位于同一个子列表中。

    :param crime1: 第一个罪名
    :param crime2: 第二个罪名
    :param crime_list: 包含多个子列表的罪名列表
    :return: 如果两个罪名在同一个子列表中，返回 True；否则返回 False
    """
    for sublist in crime_list:
        if crime1 in sublist and crime2 in sublist:
            return True
    return False

def expand_crimes(charges, bge_model, k, multiljp_file_path="./data/MultiLJP/train.jsonl"):
    """Combine legal-structure neighbors and top-k historical co-occurring charges.

    multiljp_file_path must contain training cases only (no test cases).
    """
    charges = [clean_charge(charge) for charge in charges]
    with open('./glare/structure_expand/crime_encodings.npy','rb') as f:
        embeddings = np.load(f)
    with open('./glare/structure_expand/laws_structure_list_split_remove','r',encoding='utf-8') as f:
        laws_structure_list_split_remove = json.load(f)
    crime_names = []
    for chapter in laws_structure_list_split_remove:
        crime_names.extend(chapter)
    all_expanded_charges = []
    for charge in charges:
        topk_crimes = bge_match(charge,bge_model,embeddings,crime_names,topk=2*k+10)
        # print(topk_crimes)
        # 把topk的相似罪名划分成同一章节和不同章节
        same_chapter = []
        different_chapter = []
        expanded_crimes = []
        for crime in topk_crimes:
            if crime != charge:
                #判断crime和charge是否处于同一章节
                if are_crimes_in_same_sublist(charge,crime,laws_structure_list_split_remove):
                    same_chapter.append(crime)
                else:
                    different_chapter.append(crime)
        expanded_crimes.extend(same_chapter[:k])
        expanded_crimes.extend(different_chapter[:k])
        all_expanded_charges.extend(expanded_crimes)
        # print("同一章节罪名：",same_chapter[:k])
        # print("不同章节罪名：",different_chapter[:k])
    all_expanded_charges.extend(
        expand_historical_co_occurrence(multiljp_file_path, charges, k)
    )
    return sorted(set(all_expanded_charges))
