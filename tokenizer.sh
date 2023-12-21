python tokenize_dataset_rows.py \
    --task train \
    --jsonl_path datas/CoT \
    --save_path data/COT_dataset \
    --max_seq_length 384 \ 
    --chatglm_path /mnt/workspace/llm/Langchain-Chatchat/model_path/chatglm2-6b \
    --version v2