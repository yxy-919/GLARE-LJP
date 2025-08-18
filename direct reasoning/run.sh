export CUDA_VISIBLE_DEVICES=0,1
python ./direct reasoning/llm.py \
    --test_dataset_path ./data/CAIL/test.json \
    --model_name 32b \
    --model_path ./model/Qwen2.5-32B-Instruct \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k_sampling 20 \
    --min_p 0.1 \
    --repetition_penalty 1.1 \
    --max_tokens 32768

python ./direct reasoning/summarize.py \
    --test_dataset_path ./data/CAIL/test.json \
    --method_name llm \
    --method_model 32b \
    --model_path ./model/Qwen2.5-32B-Instruct \
    --temperature 0.3 \
    --top_p 0.95 \
    --top_k_sampling 20 \
    --min_p 0.1 \
    --repetition_penalty 1.1 \
    --max_tokens 1024

