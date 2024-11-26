import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Invertible import RevIN
#from utils.headdropout import HeadDropout
import math
import numpy as np
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel

from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        #self.n_branch = configs.n_branch
        self.num_predictions = configs.t_dim
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        if configs.seq_len==512:
            self.interval = [512,336,192,96,96]#2**(np.ceil(math.log2(self.num_predictions))) 
        if configs.seq_len==1024:   
            self.interval = [1024,672,512,336,192,96]
        self.interval = self.interval[0:self.num_predictions]
        self.length_ratio = [(n + self.pred_len)/self.seq_len for n in self.interval]
        self.dominance_freq=[int(self.interval[i] // self.configs.base_T + 1) * self.configs.H_order + 10 for i in range(self.num_predictions)]
        self.Linear = nn.ModuleList([
            nn.Linear(self.dominance_freq[i], int(self.dominance_freq[i]*self.length_ratio[i])).to(torch.cfloat) for i in range(self.num_predictions)
        ]) #if configs.individual else nn.Linear(configs.seq_len, configs.pred_len * self.num_predictions) 
        
        self.dropout = nn.Dropout(configs.drop)
        self.rev = RevIN(configs.enc_in) if not configs.disable_rev else None
        print(configs.disable_rev)
        self.individual = configs.individual
        # self.Linear_dynamic = nn.Sequential(
        #     nn.Linear(4, self.num_predictions * self.channels),
        #     nn.ReLU(),
        #     nn.Linear(self.num_predictions * self.channels, self.num_predictions * self.channels)
        # )
        self.Linear_dynamic_pdot = nn.Sequential(
            nn.Linear(configs.d_llm, self.channels*self.pred_len),
            # nn.ReLU(),
            # nn.Linear(self.channels, self.channels)
        )
        self.Linear_dynamic_add = nn.Sequential(
            nn.Linear(configs.d_llm, self.channels*self.pred_len),
            # nn.ReLU(),
            # nn.Linear( self.channels*self.pred_len,  self.channels*self.pred_len)
        )
        self.Linear_static_pdot = nn.Sequential(
            nn.Linear(configs.d_llm, self.channels*self.pred_len),
            # nn.ReLU(),
            # nn.Linear(self.channels, self.channels)
        )
        self.Linear_dynamic_pdot_1 = nn.Linear(configs.d_llm, self.channels)
        self.Linear_dynamic_pdot_2 = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_dynamic_add_1 = nn.Linear(configs.d_llm, self.channels)
        self.Linear_dynamic_add_2 = nn.Linear(self.seq_len, self.pred_len)
        

        
        self.Linear_static_add = nn.Sequential(
            nn.Linear(configs.d_llm, self.channels*self.pred_len),
            # nn.ReLU(),
            # nn.Linear( self.channels*self.pred_len,  self.channels*self.pred_len)
        )
        
       # self.head_dropout = HeadDropout(configs.head_dropout)
        self.CNN1 = nn.Sequential(
            nn.Conv1d(self.num_predictions*self.channels,self.channels,3,1,1),
            nn.Conv1d(self.channels,self.channels,3,1,1),
            # nn.Conv1d(self.channels,self.channels,3,1,1),
        )
        if self.configs.w_o_dynamic_prompt==1 and self.configs.w_o_static_prompt==0:
            n_CNN2 = 1
        if self.configs.w_o_static_prompt==1 and self.configs.w_o_dynamic_prompt==0:
            n_CNN2 = 1  
        if self.configs.w_o_static_prompt==0 and self.configs.w_o_dynamic_prompt==0:         
            n_CNN2 = 2
        if self.configs.w_o_static_prompt==1 and self.configs.w_o_dynamic_prompt==1:         
            n_CNN2 = 1
        self.CNN2 = nn.Sequential(
            nn.Conv1d(n_CNN2*self.channels,self.channels,3,1,1),
            nn.Conv1d(self.channels,self.channels,3,1,1),
            # nn.Conv1d(self.channels,self.channels,3,1,1),
        )
        

        
    def forward(self, x,  x_mark, x_mark_enc, static_prompt_embd, return_gating_weights=False, return_seperate_head=False):
        # x: [B, L, D]
        # x_mark: [B, L, D_llm]
        #static_prompt_embd torch.Size([8, 1, 72, 768])
        # x_mark = x_mark[:,0]
        static_prompt_embd = static_prompt_embd[0,0,-1]
        # #MLP1
        # dynamic_out_add = self.Linear_dynamic_add(x_mark).reshape(-1, self.channels,self.pred_len)
        # #MLP2
        # dynamic_out_pdot = self.Linear_dynamic_pdot(x_mark).reshape(-1,  self.channels,self.pred_len)
        
        dynamic_out_add = self.Linear_dynamic_add_1(x_mark)
        dynamic_out_add = self.Linear_dynamic_add_2(dynamic_out_add.permute(0,2,1)) #.permute(0,2,1)
        
        dynamic_out_pdot = self.Linear_dynamic_pdot_1(x_mark)
        dynamic_out_pdot = self.Linear_dynamic_pdot_2(dynamic_out_pdot.permute(0,2,1)) #.permute(0,2,1)
        
        #MLP1
        static_out_add = self.Linear_static_add(static_prompt_embd).reshape(-1, self.channels,self.pred_len)
        #static_out_add = self.head_dropout(static_out_add)
        #static_out_add = nn.Softmax(dim=1)(static_out_add)
        #MLP2
        static_out_pdot = self.Linear_static_pdot(static_prompt_embd).reshape(-1,  self.channels,self.pred_len)
        #static_out_pdot = self.head_dropout(static_out_pdot)
        #static_out_pdot = nn.Softmax(dim=1)(static_out_pdot)
        if self.configs.is_revin==1:
            x = self.rev(x, 'norm') if self.rev else x
            x = self.dropout(x)
        else:
            seq_last = x[:,-1:,:].detach()
            x = x - seq_last
        
        y_shape = [x.size(0),self.pred_len,x.size(2)]
        pred = torch.zeros((y_shape[0], y_shape[1] * self.num_predictions, y_shape[2])).to(x.device)
        
        for idx, proj in enumerate(self.Linear):
            # pred[:, y_shape[1]*idx:y_shape[1]*(idx+1), :] = proj(x[:, -self.interval[idx]:, :].transpose(1, 2)).transpose(1, 2)         
            low_specx = torch.fft.rfft(x[:, -self.interval[idx]:, :], dim=1)
            low_specx[:,self.dominance_freq[idx]:]=0 # LPF
            low_specx = low_specx[:,0:self.dominance_freq[idx],:] # LPF
            low_specxy_ = proj(low_specx.permute(0,2,1)).permute(0,2,1)
            
            low_specxy = torch.zeros([low_specxy_.size(0),int((self.interval[idx]+self.pred_len)/2+1),low_specxy_.size(2)],dtype=low_specxy_.dtype).to(low_specxy_.device)
            low_specxy[:,0:low_specxy_.size(1),:]=low_specxy_ # zero padding
            low_xy=torch.fft.irfft(low_specxy, dim=1)
            pred[:, y_shape[1]*idx:y_shape[1]*(idx+1), :]=(low_xy * self.length_ratio[idx])[:, -self.pred_len:] # energy compemsation for the length change

        # print(pred.shape) torch.Size([8, 288, 321])
        pred_raw = pred.permute(0, 2, 1).reshape(-1, self.channels, self.pred_len, self.num_predictions).permute(0, 1, 3, 2)
        pred_raw = pred_raw.reshape(-1,self.num_predictions*self.channels,self.pred_len)
       
        temp = self.CNN1(pred_raw) #torch.Size([8, 321, 96])
        #dynamic
        pred_dynamic = temp * dynamic_out_pdot#.unsqueeze(-1)
        pred_dynamic += dynamic_out_add
        #static
        pred_static = temp * static_out_pdot#.unsqueeze(-1)
        pred_static += static_out_add
        
        # pred = pred.sum(dim=1).permute(0,2,1)      
        # .squeeze(2).reshape(-1, self.channels, self.pred_len).permute(0,2,1)
        
        if self.configs.w_o_dynamic_prompt==1 and self.configs.w_o_static_prompt==0:
            pred = pred_static
        if self.configs.w_o_static_prompt==1 and self.configs.w_o_dynamic_prompt==0:
            pred = pred_dynamic  
        if self.configs.w_o_static_prompt==0 and self.configs.w_o_dynamic_prompt==0:         
            pred = torch.concat([pred_dynamic,pred_static],dim=1)
        if self.configs.w_o_static_prompt==1 and self.configs.w_o_dynamic_prompt==1:         
            pred = temp
        pred = (self.CNN2(pred)+temp).permute(0,2,1) 
        if self.configs.is_revin==1:
            pred = self.rev(pred, 'denorm') if self.rev else pred
        else:    
            x = pred + seq_last
        
        return pred
    
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])