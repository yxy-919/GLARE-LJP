import requests
from requests.exceptions import Timeout
import json
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm
import concurrent
import re
from nltk.tokenize import sent_tokenize
import string
from typing import Optional, Tuple
import jieba


url = "https://google.serper.dev/search"

def serper_search(query):
    timeout = 20
    payload = json.dumps({
    "q": query,
    "gl": "cn",
    "hl": "zh-cn"
    })
    headers = {
    'X-API-KEY': 'API-KEY',
    'Content-Type': 'application/json'
    }
    for attempt in range(5):
        try:
            response = requests.request("POST", url, headers=headers, data=payload,timeout=timeout)
            response.raise_for_status()  # Raise an error for bad responses
            return json.loads(response.text)
        except Timeout:
            print(f"Web Search request timed out ({timeout} seconds) for query: {query}")
            if attempt == 4:
                print(f"重试5次后仍失败，查询: {query}")
                return {}
            time.sleep(0.1)
        except requests.exceptions.RequestException as e:
            print(f"Error occurred during Web Search request: {e}")
            if attempt == 4:
                print(f"重试5次后仍失败，查询: {query}")
                return {}
            time.sleep(0.1)
    return {}

def extract_relevant_info(search_results):
    useful_info = []
    for id,result in enumerate(search_results["organic"]):
        if "中华人民共和国刑法" in result["title"]:
            continue
        if "snippet" in result:
            if "没有权限" in result["snippet"] and "登录" in result["snippet"] and "注册" in result["snippet"]:
                continue
            info = {
                'id': id,
                'title': result["title"],
                'url': result["link"],
                'snippet': result["snippet"]
            }
        else:
            info = {
                'id': id,
                'title': result["title"],
                'url': result["link"],
                'snippet': ""
            }
        useful_info.append(info)
    return useful_info

def extract_text_from_url(url,use_jina=False,jina_api_key=None):
    try:
        if use_jina:
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
                # 'X-With-Links-Summary': 'true'
            }
            response = requests.get(f'https://r.jina.ai/{url}', headers=jina_headers).text
            # Remove URLs
            pattern = r"\(https?:.*?\)|\[https?:.*?\]"
            text = re.sub(pattern, "", response).replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
        return text[:8000]
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError:
        return "Error: Connection error occurred"
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def fetch_page_content(urls,max_workers,use_jina=False,jina_api_key=None):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_text_from_url,url,use_jina,jina_api_key): url for url in urls
        }
        for future in tqdm(concurrent.futures.as_completed(futures), desc="Fetching URLs", total=len(futures)):
            url = futures[future]
            try:
                result = future.result()
                results[url] = result
            except Exception as e:
                print(f"Error fetching {url}: {e}")
            time.sleep(0.2)
    return results

def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))

def split_chinese_sentences(text: str) -> list:
    """按中文句号、感叹号、问号、分号分句"""
    # 使用正则表达式分句（保留分隔符）
    sentences = re.split(r'([。！？；])', text)
    # 将分隔符重新拼接到前一句
    sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '') 
                 for i in range(0, len(sentences), 2)]
    return [s.strip() for s in sentences if s.strip()]   


def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)

def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 2500) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text (str): The full text extracted from the webpage.
        snippet (str): The snippet to match.
        context_chars (int): Number of characters to include before and after the snippet.

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful, the second element is the extracted context.
    """
    if snippet != "":
        try:
            full_text = full_text[:50000]
            
            snippet_words = set(jieba.cut_for_search(snippet))

            best_sentence = None
            best_f1 = 0.2

            sentences = split_chinese_sentences(full_text)

            for sentence in sentences:
                sentence_words = set(jieba.cut_for_search(sentence))
                f1 = f1_score(snippet_words, sentence_words)
                if f1 > best_f1:
                    best_f1 = f1
                    best_sentence = sentence

            if best_sentence:
                para_start = full_text.find(best_sentence)
                para_end = para_start + len(best_sentence)
                start_index = max(0, para_start - context_chars)
                end_index = min(len(full_text), para_end + context_chars)
                context = full_text[start_index:end_index]
                return True, context
            else:
                # If no matching sentence is found, return the first context_chars*2 characters of the full text
                return False, full_text[:context_chars * 2]
        except Exception as e:
            return False, f"Failed to extract snippet context due to {str(e)}"
    else:
        # If the snippet is empty, return the first context_chars*2 characters of the full text
        return False, full_text[:context_chars * 2]


if __name__ == "__main__":
    search_results = serper_search("破坏公用电信设施罪和扰乱无线电通讯管理秩序罪的关键区分点")
    relevant_info = extract_relevant_info(search_results)
    print(relevant_info)