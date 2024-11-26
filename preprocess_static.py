import argparse
import torch
from Preprocess_Llama import Llama_Model
from Preprocess_gpt2 import gpt2_Model
from data_provider.data_loader import Dataset_Preprocess
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer#, BertConfig, \BertModel, BertTokenizer

def load_content(name):
    if 'ETT' in name:
        file = 'ETT'
    else:
        file = name
    with open('./prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoTimes Preprocess')
    parser.add_argument('--gpu', type=int, default=1, help='gpu id')
    parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model') # llama_2, GPT2, BERT
    #parser.add_argument('--llm_ckp_dir', type=str, default='/home/lzhang726/llm_model_weight/Llama-2-7b-hf-old', help='llm checkpoints dir')
    parser.add_argument('--dataset', type=str, default='ETTm1', 
                        help='dataset to preprocess, options:[ETTh1, electricity, weather, traffic]')
    parser.add_argument('--llm_layers', type=int, default=6, help='gpu id')
    args = parser.parse_args()
    print(args.dataset)
    if args.llm_model == 'GPT2':
        args.llm_ckp_dir = '/home/lzhang726/llm_model_weight/gpt2'
        model = gpt2_Model(args)
        save_dir_path = './datainfo_emb_GPT2/'
        gpt2_config = GPT2Config.from_pretrained(args.llm_ckp_dir)
        gpt2_config.num_hidden_layers = args.llm_layers
        gpt2_config.output_attentions = True
        gpt2_config.output_hidden_states = True
        llm_model = GPT2Model.from_pretrained(
                    args.llm_ckp_dir,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=gpt2_config,
                )
        tokenizer = GPT2Tokenizer.from_pretrained(
                    args.llm_ckp_dir,
                    trust_remote_code=True,
                    local_files_only=True
                )
    if args.llm_model == 'llama_2':
        args.llm_ckp_dir = '/home/lzhang726/llm_model_weight/Llama-2-7b-hf-old'
        model = Llama_Model(args)
        save_dir_path = './datainfo_llama2/'
        llama_config = LlamaConfig.from_pretrained(args.llm_ckp_dir)
        llama_config.num_hidden_layers = args.llm_layers
        llama_config.output_attentions = True
        llama_config.output_hidden_states = True
        tokenizer = LlamaTokenizer.from_pretrained(
                        # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=True
                    )
        
        llm_model = LlamaModel.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=True,
                config=llama_config,
                # load_in_4bit=True
                    )
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        pad_token = '[PAD]'
        tokenizer.add_special_tokens({'pad_token': pad_token})
        tokenizer.pad_token = pad_token
    
    assert args.dataset in ['ETTh1', 'electricity', 'weather', 'traffic','ETTm1','ETTm2','ETTh2']
    root_path = '/home/lzhang726/recent_work/LinearLLM/prompt_bank/'
    if 'ETT' in args.dataset:
        print("Encoding {}".format(args.dataset))
        data_path='ETT'
    elif args.dataset == 'electricity':
        data_path='ECL'
    elif args.dataset == 'weather':
        data_path='Weather'
    elif args.dataset == 'traffic':
        data_path='Traffic'
  
    content = load_content(data_path)
    prompt = f'Dataset description: {content}'
    prompt = tokenizer(content, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
    prompt_embeddings = llm_model.get_input_embeddings()(prompt)  # (batch, prompt_token, dim)

    print(prompt_embeddings.shape)
    torch.save(prompt_embeddings.detach().cpu(), save_dir_path + f'/{args.dataset}.pt')
