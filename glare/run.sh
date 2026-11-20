export CUDA_VISIBLE_DEVICES=2,3


python ./glare/glare.py \
    --precedent_database_path ./data/precedent_database_with_reasoning.jsonl \
    --multiljp_file_path ./data/MultiLJP/train.jsonl \
    --retriever_model_path ./model/SAILER_zh \
    --precedent_top_k 5 \
    --test_dataset_path ./data/CAIL/test.json \
    --max_search_limit 10 \
    --max_turn 10 \
    --top_k 3 \
    --top_k_expand 3 \
    --model_name qwq \
    --model_path ./model/QwQ-32B \
    --max_doc_len 3000 \
    --use_jina False \
    --jina_api_key None \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k_sampling 20 \
    --min_p 0.1 \
    --repetition_penalty 1.1 \
    --max_tokens 32768 \
    --concurrent_limit 20
