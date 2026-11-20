import os
import sys
import json
from typing import Optional,List,Dict
import argparse
from modelscope import AutoTokenizer
import re
import torch
from prompts import get_instruction_add_reasoning_chain,web_search_summarize_instruction,get_search_agent_user_prompt
from expand_crimes import expand_crimes
from precedents import PrecedentRetriever
from vllm import LLM, SamplingParams
from typing import Optional
from serper import serper_search,extract_relevant_info,fetch_page_content,extract_snippet_with_context
from sentence_transformers import SentenceTransformer
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm

BEGIN_EXPAND = "<|begin_expand|>"
END_EXPAND = "<|end_expand|>"
BEGIN_EXPAND_RESULT = "<|begin_expand_result|>"
END_EXPAND_RESULT = "<|end_expand_result|>"
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
bge_model = SentenceTransformer("./model/BAAI_bge-m3")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Search RAG for various models.")

    # Dataset
    parser.add_argument(
        '--test_dataset_path',
        type=str,
        default="./test.json",
        help="Path of 1000 test data."
    )

    parser.add_argument("--precedent_database_path", default="./data/precedent_database_with_reasoning.jsonl")
    parser.add_argument("--historical_cases_path", default="./data/precedent_database.jsonl")
    parser.add_argument("--retriever_model_path", default="./model/SAILER_zh")
    parser.add_argument("--precedent_top_k", type=int, default=5)

    # Search agent configuration
    parser.add_argument(
        '--max_search_limit',
        type=int,
        default=5,
        help="Maximum number of searches per case."
    )

    parser.add_argument(
        '--max_turn',
        type=int,
        default=10,
        help="Maximum number of turns for the search agent."
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=1,
        help="Top k search results."
    )

    parser.add_argument(
        '--top_k_expand',
        type=int,
        default=5,
        help="Top k crimes to expand according to law structure."
    )

    parser.add_argument(
        '--max_doc_len',
        type=int,
        default=3000,
        help="Maximum length of each searched document."
    )
    parser.add_argument(
        '--use_jina',
        type=bool,
        default=False,
        help="Whether to use Jina API for document fetching."
    )

    parser.add_argument(
        '--jina_api_key',
        type=str,
        default='None',
        help="Your Jina API Key to Fetch URL Content."
    )

    # Model configuration
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        choices=['7b','qwq','32b','72b'],
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
    parser.add_argument(
        '--concurrent_limit',
        type=int,
        default=200,
        help="Maximum number of concurrent API calls"
    )
    return parser.parse_args()

async def generate_response(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
    top_k: int,
    min_p: float,
    model_name: str,
    retry_limit: int = 3,
) -> str:
    for attempt in range(retry_limit):
        try:
            async with semaphore:
                messages = [{"role": "user", "content": prompt}]
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=min(max_tokens, 32768),  # Reserve 1000 tokens for prompt
                    stop=[END_EXPAND, END_SEARCH_QUERY],
                    extra_body={
                        'top_k': top_k,
                        'include_stop_str_in_output': True,
                        'repetition_penalty': repetition_penalty,
                    },
                    timeout=1500,
                )
                # print('---\n', response.choices[0].message.content)
                return response.choices[0].message.content
        except Exception as e:
            if attempt == retry_limit - 1:
                print(f"Failed after {retry_limit} attempts: {e}")
                return ""
            await asyncio.sleep(1 * (attempt + 1))
    return ""

def extract_answer(output):
    pattern_info = "\n**最终获取的信息**"
    if pattern_info in output:
        extracted_text = output.split(pattern_info)[-1].replace("\n","").strip("```").strip()
        return extracted_text
    else:
        return output
         
async def web_search_summarize(
    client: AsyncOpenAI,
    search_query: str,
    document: str,
    max_tokens: int = 32768,
    temperature: float = 0.7,
    top_p: float = 0.8,
    repetition_penalty: float = 1.05,
    top_k: int = 20,
    min_p: float = 0.05,
    model_name: str = 'qwq',
    semaphore: asyncio.Semaphore = None,
) -> str:
    user_prompt = web_search_summarize_instruction(search_query, document)

    raw_output = await generate_response(
        client=client,
        prompt=user_prompt,
        semaphore=semaphore,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        min_p=min_p,
        model_name=model_name,
    )
    
    extracted_info = extract_answer(raw_output)
    return extracted_info

def extract_between(text:str, start_tag:str, end_tag:str) -> Optional[str]:
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None

async def process_single_sequence(
    seq: Dict,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    search_cache: Dict,
    url_cache: Dict,
    turn: int = 0,
) -> Dict:
    """Process a single sequence through its entire reasoning chain"""
    
    while not seq['finished'] and turn < args.max_turn:
        # Generate next step in reasoning
        text = await generate_response(
            client=client,
            prompt=seq['prompt'],
            semaphore=semaphore,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k_sampling,
            min_p=args.min_p,
            model_name=args.model_name,
        )
        seq['prompt'] += text
        seq['output'] += text
        crimes_to_expand = extract_between(text, BEGIN_EXPAND, END_EXPAND)
        search_query = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

        if crimes_to_expand:
            seq['prompt'] = seq['prompt'].replace('</think>\n','')
            seq['output'] = seq['output'].replace('</think>\n','')
            crimes_to_expand = [c.strip() for c in crimes_to_expand.split(",")]
            expanded_crimes = expand_crimes(crimes_to_expand, bge_model, args.top_k_expand, args.historical_cases_path)
            expanded_crimes = list(set(expanded_crimes))
            expanded_crimes_str = ', '.join(expanded_crimes)
            # Add precedent crimes
            expanded_crimes_str = "根据语意扩展得到的罪名：" + expanded_crimes_str +  "\n" + "类案的推理路径：\n" + seq['reasoning_chain']
            cur_fact_str = "参考类案推理路径，继续分析当前的案件：\n" + seq['item']['fact'] + '\n'
            # Append expanded crimes to the prompt
            expanded_crimes_str = json.dumps(expanded_crimes_str, ensure_ascii=False, indent=2)
            cur_fact_str = json.dumps(cur_fact_str, ensure_ascii=False, indent=2)
            append_text = f"\n{BEGIN_EXPAND_RESULT}\n{expanded_crimes_str}\n{END_EXPAND_RESULT}\n{cur_fact_str}"
            seq['prompt'] += append_text
            seq['output'] += append_text

        if search_query and seq['output'].rstrip().endswith(END_SEARCH_QUERY):
            seq['whether_search'] = True
            # Remove the </think> tag from the prompt and output
            seq['prompt'] = seq['prompt'].replace('</think>\n','')
            seq['output'] = seq['output'].replace('</think>\n','')
            if seq['search_count'] < args.max_search_limit and search_query not in seq['executed_search_queries']:
                # Execute search
                if search_query in search_cache:
                    results = search_cache[search_query]
                else:
                    try:
                        results = serper_search(search_query)
                        search_cache[search_query] = results
                    except Exception as e:
                        print(f"Error during search query '{search_query}': {e}")
                        search_cache[search_query] = {}
                        results = {}
                if not results:
                    append_text = f"\n{BEGIN_SEARCH_RESULT}\n没有搜索到结果，请换个表述重新提问。\n{END_SEARCH_RESULT}\n"
                    seq['prompt'] += append_text
                    seq['output'] += append_text
                    print("No answer for search query:", search_query)
                else:
                    relevant_info = extract_relevant_info(results)[:args.top_k]
                    urls_to_fetch = []
                    for doc_info in relevant_info:
                        url = doc_info['url']
                        if url not in url_cache:
                            urls_to_fetch.append(url)

                    if urls_to_fetch:
                        try:
                            contents = fetch_page_content(urls_to_fetch, max_workers=32,use_jina=args.use_jina, jina_api_key=args.jina_api_key)
                            for url, content in contents.items():
                                url_cache[url] = content
                        except Exception as e:
                            print(f"Error fetching URLs: {e}")
                            for url in urls_to_fetch:
                                url_cache[url] = ""

                    formatted_documents = ""
                    for i, doc_info in enumerate(relevant_info):
                        url = doc_info['url']
                        raw_context = url_cache.get(url, "")
                        #print(f"raw_context: {raw_context}")
                        doc_info['snippet'] = doc_info['snippet'].replace('<b>','').replace('</b>','')            
                        success, filtered_context = extract_snippet_with_context(raw_context, doc_info['snippet'], context_chars=args.max_doc_len)
                        if success:
                            context = filtered_context
                        else:
                            context = raw_context[:args.max_doc_len*2]

                        doc_info['context'] = context
                        formatted_documents += f"**网页{i + 1}:**\n"
                        formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"   

                    analysis = await web_search_summarize(
                        client=client,
                        search_query=search_query,
                        document=formatted_documents,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        top_k=args.top_k_sampling,
                        min_p=args.min_p,
                        model_name=args.model_name,
                        semaphore=semaphore,
                    )

                    append_text = f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n"
                    seq['prompt'] += append_text
                    seq['output'] += append_text
                    seq['search_count'] += 1
                    seq['executed_search_queries'].add(search_query)

            elif seq['search_count'] >= args.max_search_limit:
                limit_message = f"\n{BEGIN_SEARCH_RESULT}\n达到最大搜索次数。\n{END_SEARCH_RESULT}\n"
                seq['prompt'] += limit_message
                seq['output'] += limit_message

            elif search_query in seq['executed_search_queries']:
                limit_message = f"\n{BEGIN_SEARCH_RESULT}\n已搜索过这个问题。\n{END_SEARCH_RESULT}\n"
                seq['prompt'] += limit_message
                seq['output'] += limit_message
        # if not crimes_to_expand and not search_query:
        #     seq['finished'] = True
        if not crimes_to_expand and not search_query and seq['whether_search']:
            seq['finished'] = True
        elif not crimes_to_expand and not seq['whether_search']:
            new_prompt = "\n\n上述的推理过程没有调用外部的搜索工具，为了确保推理过程中使用的法律知识的准确性，请重新调用搜索工具区分你最终推理得到的罪名和类案中所涉及到的罪名，并根据搜索结果重新推理。\n\n注意：使用 <|begin_search_query|> 请求网络搜索，并以 <|end_search_query|> 结束。请严格遵守这个格式。\n\n"
            # seq['prompt'] = clean_output(seq['prompt'])
            seq['prompt'] += new_prompt
            seq['output'] += "重新调用搜索工具进行推理。"
        turn += 1

    return seq

async def main_async():
    args = parse_args()

    # Define output directory based on model
    output_dir = './output/glare/qwq' 
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------- Caching Mechanism ----------------------
    # Define cache directory and file paths
    cache_dir = './output/glare/cache'
    search_cache_path = os.path.join(cache_dir, 'search_cache.json')
    url_cache_path = os.path.join(cache_dir, 'url_cache.json')

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Load existing caches or create a empty dictionary
    if os.path.exists(search_cache_path):
        with open(search_cache_path, 'r', encoding='utf-8') as f:
            search_cache = json.load(f)
    else:
        search_cache = {}
    
    if os.path.exists(url_cache_path):
        with open(url_cache_path, 'r', encoding='utf-8') as f:
            url_cache = json.load(f)
    else:
        url_cache = {}

    # Define a function to save cache
    def save_caches():
        with open(search_cache_path, 'w', encoding='utf-8') as f:
            json.dump(search_cache, f, ensure_ascii=False, indent=2)
        with open(url_cache_path, 'w', encoding='utf-8') as f:
            json.dump(url_cache, f, ensure_ascii=False, indent=2)

    # ---------------------- Model Loading ----------------------
    client = AsyncOpenAI(
        api_key="empty",
        base_url="http://localhost:19001/v1",
    )

    # ---------------------- Data Loading ----------------------
    data = []
    with open(args.test_dataset_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f.readlines()):
            case = json.loads(line)
            data.append(case)
    # data = data[:2]
    
    # Retrieve reasoning-augmented precedents by facts, never by test labels.
    retriever = PrecedentRetriever(args.precedent_database_path, args.retriever_model_path)
    reasoning_chain_list = [retriever.retrieve(item["fact"], args.precedent_top_k) for item in data]

    # ---------------------- Prepare Input ----------------------
    input_list = []
    for item in data:
        fact = item["fact"]
        criminal = item["defendants"]
        instruction = get_instruction_add_reasoning_chain(args.max_search_limit)
        user_prompt = get_search_agent_user_prompt(fact,criminal,args.model_name)
        prompt = instruction + user_prompt
        input_list.append(prompt)

    # Initialize sequences
    active_sequences = [{
        "item": item,
        "prompt": prompt,
        "output": '',
        "finished": False,
        'executed_search_queries': set(),
        'search_count' : 0,
        'whether_search': False,
        'reasoning_chain': ""
    } for item,prompt in zip(data,input_list)]
    
    # Add reasoning chain to sequences
    for idx,seq in enumerate(active_sequences):
        reasoning_chain = reasoning_chain_list[idx]
        #print(reasoning_chain)
        seq["reasoning_chain"] = reasoning_chain

    semaphore = asyncio.Semaphore(args.concurrent_limit)

    tasks = [
        process_single_sequence(
            seq=seq,
            client=client,
            semaphore=semaphore,
            args=args,
            search_cache=search_cache,
            url_cache=url_cache,
        )
        for seq in active_sequences
    ]
    with tqdm(total=len(tasks)) as pbar:
        async def track_progress(task):
            result = await task
            pbar.update(1)
            return result
            
        tracked_tasks = [track_progress(task) for task in tasks]
        completed_sequences = await asyncio.gather(*tracked_tasks)
    
    output_list = [seq['output'] for seq in completed_sequences]

    # Collect all outputs
    res_list = []
    for i,response in enumerate(output_list):
        res = {}
        res["caseID"] = data[i]["caseID"]
        res["predict"] = f"{response}"
        res_list.append(res)
    with open(os.path.join(output_dir, "output"), "a+", encoding="utf-8") as f:
        for res in res_list:
            line = json.dumps(res, ensure_ascii=False)
            f.write(line + "\n")
    
    num = 0
    search_count = []
    for seq in active_sequences:
        search_count.append(seq['search_count'])
        num += seq['search_count']
    print(f"All search count: {num}")

    # with open(os.path.join(output_dir, "search_count"), "a+", encoding="utf-8") as f:
    #     json.dump(search_count, f, ensure_ascii=False, indent=2)

    print("Process completed.")

    # ---------------------- Update Search and URL Cache ----------------------
    print('Updating Search and URL Cache...')
    # Load existing caches or initialize empty dictionaries
    if os.path.exists(search_cache_path):
        with open(search_cache_path, 'r', encoding='utf-8') as f:
            search_cache_new = json.load(f)
    else:
        search_cache_new = {}

    if os.path.exists(url_cache_path):
        with open(url_cache_path, 'r', encoding='utf-8') as f:
            url_cache_new = json.load(f)
    else:
        url_cache_new = {}

    search_cache.update(search_cache_new)
    url_cache.update(url_cache_new)

    save_caches()

    print("Process completed.")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()