import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Invertible import RevIN
from utils.headdropout import HeadDropout



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.configs = configs
        
        self.num_predictions = configs.t_dim
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        self.Linear = nn.ModuleList([
            nn.Linear(configs.seq_len, configs.pred_len * self.num_predictions) for _ in range(configs.enc_in)
        ]) if configs.individual else nn.Linear(configs.seq_len, configs.pred_len * self.num_predictions)
    
        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len)
        )
        
        self.dropout = nn.Dropout(configs.drop)
        self.rev = RevIN(configs.enc_in) if not configs.disable_rev else None
        self.individual = configs.individual
        
        self.Linear_Temporal = nn.Sequential(
            nn.Linear(4, self.num_predictions * self.channels),
            nn.ReLU(),
            nn.Linear(self.num_predictions * self.channels, self.num_predictions * self.channels)
        )
        self.head_dropout = HeadDropout(configs.head_dropout)

    def forward(self, x, x_mark, return_gating_weights=False, return_seperate_head=False):
        # x: [B, L, D]
        x_mark_initial = x_mark[:,0]
        x = self.rev(x, 'norm') if self.rev else x
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
            
        temporal_out = self.Linear_Temporal(x_mark_initial).reshape(-1, self.num_predictions, self.channels)
        temporal_out = self.head_dropout(temporal_out)
        temporal_out = nn.Softmax(dim=1)(temporal_out)
        pred_raw = pred.permute(0, 2, 1).reshape(-1, self.channels, self.pred_len, self.num_predictions).permute(0, 3, 1, 2)
        pred = pred_raw * temporal_out.unsqueeze(-1)
        pred = pred.sum(dim=1).permute(0,2,1)
        
        pred = self.rev(pred, 'denorm') if self.rev else pred

        return pred
