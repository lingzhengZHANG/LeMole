import argparse
import torch
from Preprocess_Llama import Llama_Model
from Preprocess_gpt2 import gpt2_Model
from data_provider.data_loader import Dataset_Preprocess
from torch.utils.data import DataLoader
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoTimes Preprocess')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model') # llama_2, GPT2, BERT
    #parser.add_argument('--llm_ckp_dir', type=str, default='/home/lzhang726/llm_model_weight/Llama-2-7b-hf-old', help='llm checkpoints dir')
    parser.add_argument('--dataset', type=str, default='electricity', 
                        help='dataset to preprocess, options:[ETTh1,ETTm1, electricity, weather, traffic]')
    args = parser.parse_args()
    
    print(args.dataset)
    
    seq_len = 672
    label_len = 0
    pred_len = 96
    if args.llm_model == 'GPT2':
        args.llm_ckp_dir = '/home/lzhang726/llm_model_weight/gpt2'
        model = gpt2_Model(args)
        save_dir_path = './timestamp_emb_GPT2_'+str(seq_len)
    if args.llm_model == 'llama_2':
        args.llm_ckp_dir = '/home/lzhang726/llm_model_weight/Llama-2-7b-hf-old'
        model = Llama_Model(args)
        save_dir_path = './timestamp_emb_llama2_'+str(seq_len)
    if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)
    
    for args.dataset in ['ETTh1', 'electricity',  'traffic','ETTm1']: #'weather',,'ETTm2','ETTh2'
      if args.dataset == 'ETTh1':
          data_set = Dataset_Preprocess(
              root_path='/home/lzhang726/datasets/prediction/ETT-small/',
              data_path='ETTh1.csv',
              size=[seq_len, label_len, pred_len])
      elif args.dataset == 'ETTh2':
          data_set = Dataset_Preprocess(
              root_path='/home/lzhang726/datasets/prediction/ETT-small/',
              data_path='ETTh2.csv',
              size=[seq_len, label_len, pred_len])
      elif args.dataset == 'ETTm1':
          data_set = Dataset_Preprocess(
              root_path='/home/lzhang726/datasets/prediction/ETT-small/',
              data_path='ETTm1.csv',
              size=[seq_len, label_len, pred_len])
      elif args.dataset == 'ETTm2':
          data_set = Dataset_Preprocess(
              root_path='/home/lzhang726/datasets/prediction/ETT-small/',
              data_path='ETTm2.csv',
              size=[seq_len, label_len, pred_len]) 
      elif args.dataset == 'electricity':
          data_set = Dataset_Preprocess(
              root_path='/home/lzhang726/datasets/prediction/electricity/',
              data_path='electricity.csv',
              size=[seq_len, label_len, pred_len])
      elif args.dataset == 'weather':
          data_set = Dataset_Preprocess(
              root_path='/home/lzhang726/datasets/prediction/weather/',
              data_path='weather.csv',
              size=[seq_len, label_len, pred_len])
      elif args.dataset == 'traffic':
          data_set = Dataset_Preprocess(
              root_path='/home/lzhang726/datasets/prediction/traffic/',
              data_path='traffic.csv',
              size=[seq_len, label_len, pred_len])
  
      data_loader = DataLoader(
          data_set,
          batch_size=128,
          shuffle=False,
      )
  
      from tqdm import tqdm
      print(len(data_set.data_stamp))
      print(data_set.tot_len)
      
      output_list = []
      for idx, data in tqdm(enumerate(data_loader)):
          output = model(data)
          output_list.append(output.detach().cpu())
      result = torch.cat(output_list, dim=0)
      print(result.shape)
      torch.save(result, save_dir_path + f'/{args.dataset}.pt')
