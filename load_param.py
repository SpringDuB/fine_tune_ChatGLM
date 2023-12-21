# -*- coding: utf-8 -*-
import torch 
from transformers import AutoModel,AutoTokenizer
def load_state(path):
    param = torch.load(path)
    opt = torch.load('/mnt/workspace/llm/fine_tuneglm/running_outputs/checkpoint-50000/optimizer.pt')
    return param, opt
        
def load_model(model_path):
    model = AutoModel.from_pretrained(model_path,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    return model, tokenizer

def fusion_state(model_path, path):
    model, tokenizer = load_model(model_path)
    param, opt = load_state(path)
    key = 'base_model.model.transformer.encoder.layers.{}.self_attention.query_key_value.lora_{}.default.weight'
    num_layers = model.config.num_layers
    name = {}
    for n,p in model.named_parameters():
        if 'self_attention.query_key_value' in n and 'bias' not in n:
            name[n] = p
    key_list = list(name.keys())
    for i in range(num_layers):
        a_key = key.format(i, 'A')
        b_key = key.format(i, 'B')
        a = param[a_key]
        b = param[b_key]
        state = torch.matmul(b, a)
        name[key_list[i]] = name[key_list[i]] + state
    model.load_state_dict(name, strict=False)
    return model, tokenizer
    
if __name__ == '__main__':
    path = r'/mnt/workspace/llm/fine_tuneglm/running_outputs/checkpoint-50000/adapter_model.bin'
    # param, opt = load_state(path)
    model_path = r'/mnt/workspace/llm/Langchain-Chatchat/model_path/chatglm2-6b'
    # load_model(model_path)
    model, tokenizer = fusion_state(model_path, path)
    model.to('cuda')
    while True:
        query = input('用户:')
        history = None
        if query.strip().lower() == 'q':
            break
        output, history = model.chat(tokenizer,query=query, history=history)
        print('chatglm:' + output)