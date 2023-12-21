import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import json
from tqdm import tqdm


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/mnt/workspace/llm/fine_tuneglm/datas/CoT_zh/CoT_Chinese_data.csv")
    parser.add_argument("--save_dir", type=str, default="/mnt/workspace/llm/fine_tuneglm/datas/CoT")

    args = parser.parse_args()
    datas = pd.read_csv(args.data_path).to_numpy()
    length = datas.shape[0]
    train_datas, test_datas = train_test_split(datas, train_size=0.75, shuffle=True)
    with open(args.save_dir + '/train.jsonl', 'w', encoding='utf-8') as f:
        for Instruction, input, output in tqdm(train_datas, desc="formatting.."):
            example = {'instruction':Instruction, 'input':input, 'output':output}
            f.write(json.dumps(format_example(example),ensure_ascii=False) + '\n')
    with open(args.save_dir + '/test.jsonl', 'w', encoding='utf-8') as f:
        for Instruction, input, output in tqdm(test_datas, desc="formatting.."):
            example = {'instruction':Instruction, 'input':input, 'output':output}
            f.write(json.dumps(format_example(example),ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
