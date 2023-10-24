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
            nn.Linear(configs.seq_len, configs.pred_len * self.num_predictions) for _ in range(configs.channel)
        ]) if configs.individual else nn.Linear(configs.seq_len, configs.pred_len * self.num_predictions)
        
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
        # x_mark_hourofday = (x_mark_initial[:,0] + 0.5) * 23
        # x_mark_hourofday_sin = torch.sin(x_mark_hourofday * (2 * torch.pi / 24)).unsqueeze(1)
        # x_mark_hourofday_cos = torch.cos(x_mark_hourofday * (2 * torch.pi / 24)).unsqueeze(1)
        # x_mark_dayofweek = (x_mark_initial[:,1] + 0.5) * 6
        # x_mark_dayofweek_sin = torch.sin(x_mark_dayofweek * (2 * torch.pi / 7)).unsqueeze(1)
        # x_mark_dayofweek_cos = torch.cos(x_mark_dayofweek * (2 * torch.pi / 7)).unsqueeze(1)
        # x_mark_dayofmonth = (x_mark_initial[:,2] + 0.5) * 29
        # x_mark_dayofmonth_sin = torch.sin(x_mark_dayofmonth * (2 * torch.pi / 30)).unsqueeze(1)
        # x_mark_dayofmonth_cos = torch.cos(x_mark_dayofmonth * (2 * torch.pi / 30)).unsqueeze(1)
        # x_mark_monthofyear = (x_mark_initial[:,3] + 0.5) * 11
        # x_mark_monthofyear_sin = torch.sin(x_mark_monthofyear * (2 * torch.pi / 12)).unsqueeze(1)
        # x_mark_monthofyear_cos = torch.cos(x_mark_monthofyear * (2 * torch.pi / 12)).unsqueeze(1)
        # x_mark_initial = torch.cat([x_mark_hourofday_sin, x_mark_hourofday_cos, x_mark_dayofweek_sin, x_mark_dayofweek_cos, x_mark_dayofmonth_sin, x_mark_dayofmonth_cos, x_mark_monthofyear_sin, x_mark_monthofyear_cos], dim=1)
        # print(x_mark_initial.shape)

        temporal_out = self.Linear_Temporal(x_mark_initial).reshape(-1, self.num_predictions, self.channels)
        temporal_out = self.head_dropout(temporal_out)
        temporal_out = nn.Softmax(dim=1)(temporal_out)
            
        x = self.rev(x, 'norm') if self.rev else x
        x = self.dropout(x)
        # x_mark_initial = self.dropout(x_mark_initial)
        if self.individual:
            y_shape = [x.size(0),x.size(1),self.pred_len]
            pred = torch.zeros((y_shape[0], y_shape[1] * self.num_predictions, y_shape[2])).to(y.device)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x[:, :, idx])
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
            
        # print(pred.shape)
        pred_raw = pred.permute(0, 2, 1).reshape(-1, self.channels, self.pred_len, self.num_predictions).permute(0, 3, 1, 2)
        pred = pred_raw * temporal_out.unsqueeze(-1)
        pred = pred.sum(dim=1).permute(0,2,1)
        # .squeeze(2).reshape(-1, self.channels, self.pred_len).permute(0,2,1)
        
        pred = self.rev(pred, 'denorm') if self.rev else pred
        
        return pred
