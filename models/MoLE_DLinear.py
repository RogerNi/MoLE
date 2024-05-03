import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hashlib
from utils.headdropout import HeadDropout

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        
        self.num_predictions = configs.t_dim
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in
        
        
        # time feature size
        self.expected_time_features = 4 if configs.freq.lower().endswith('h') else 5


        self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len * self.num_predictions)
        self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len * self.num_predictions)
            
        input_dim = self.expected_time_features
        self.Linear_Temporal = nn.Sequential(
            nn.Linear(input_dim, self.num_predictions * self.channels),
            nn.ReLU(),
            nn.Linear(self.num_predictions * self.channels, self.num_predictions * self.channels)
        )
        
        self.head_dropout = HeadDropout(configs.head_dropout)


    def forward(self, x, x_mark, return_gating_weights=False, return_seperate_head=False):
        # x: [Batch, Input length, Channel]
        x_mark_initial = x_mark[:,0]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        
        
        temporal_out = self.Linear_Temporal(x_mark_initial).reshape(-1, self.num_predictions)
        temporal_out = self.head_dropout(temporal_out) 
        temporal_out = nn.Softmax(dim=1)(temporal_out)

        x_raw = x.reshape(-1, self.pred_len, self.num_predictions)
        x = torch.matmul(x_raw, temporal_out.unsqueeze(2)).squeeze(2).reshape(-1, self.channels, self.pred_len).permute(0,2,1)
        
        return x
