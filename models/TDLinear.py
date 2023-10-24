import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hashlib
from utils.headdropout import HeadDropout

def hash_tensor(tensor, hash_size=4):
    # Handle tensor on CPU because Python's hashlib works on CPU.
    tensor_cpu = tensor.cpu().numpy()

    batch_size = tensor_cpu.shape[0]
    mapped_tensors = []

    for i in range(batch_size):
        tensor_bytes = tensor_cpu[i].tobytes()

        # Apply the hash function
        hashed = hashlib.sha256(tensor_bytes).digest()

        # Take the first hash_size bytes of the hash
        mapped_tensor = torch.tensor(list(hashed[:hash_size]), device=tensor.device)

        mapped_tensors.append(mapped_tensor.float() / 255.0 * 2 - 1)  # Normalize to [-1, 1]

    return torch.stack(mapped_tensors)

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
        
        # ablations
        self.fixed_routing = configs.use_fixed_routing
        self.ts_routing = configs.use_time_series_routing
        self.random_routing = configs.use_random_routing
        self.random_bypass = configs.use_random_bypass_routing
        self.equal_weights_test_time = configs.use_equal_weighting_in_testing_time
        
        # time feature size
        self.expected_time_features = 4 if configs.freq.lower().endswith('h') else 5

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len * self.num_predictions))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len * self.num_predictions))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len * self.num_predictions)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len * self.num_predictions)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        
        if self.ts_routing:
            input_dim = self.seq_len * self.channels
        else:
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
        if self.ts_routing:
            x_mark_initial = x.reshape(-1, self.seq_len * self.channels)
        elif self.fixed_routing:
            x_mark_initial = hash_tensor(x, hash_size=self.expected_time_features)
        elif self.random_routing:
            x_mark_initial = torch.randn(x_mark_initial.shape, device=x_mark_initial.device) * 2 - 1
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        
        
        temporal_out = self.Linear_Temporal(x_mark_initial).reshape(-1, self.num_predictions)
        
        if self.random_bypass:
            temporal_out = torch.randn(temporal_out.shape, device=temporal_out.device)
            
        if not self.training and self.equal_weights_test_time:
            temporal_out = torch.ones(temporal_out.shape, device=temporal_out.device)
        
            
        temporal_out = self.head_dropout(temporal_out)
            
        temporal_out = nn.Softmax(dim=1)(temporal_out)

        x_raw = x.reshape(-1, self.pred_len, self.num_predictions)
        
        if not self.training and return_seperate_head:
            # get output of each head and assign to x
            x = x.reshape(-1, self.channels, self.pred_len, self.num_predictions).permute(0, 2, 1, 3)
        else:
            # print(x.shape, temporal_out.shape)
            x = torch.matmul(x_raw, temporal_out.unsqueeze(2)).squeeze(2).reshape(-1, self.channels, self.pred_len).permute(0,2,1)
        
        temporal_out = temporal_out.reshape(-1, self.channels, self.num_predictions)
        # loss = self.forward_loss(x, y) - (0 if not self.increase_weights_var else torch.mean(torch.cdist(temporal_out, temporal_out)))
        # pred_raw = x_raw.permute(0, 2, 1)
        # loss -= 0 if not self.increase_head_var else torch.mean(torch.cdist(pred_raw, pred_raw))*0.1
        
        
        # return x, loss, temporal_out # to [Batch, Output length, Channel]
        if return_gating_weights:
            return x, temporal_out
        return x
