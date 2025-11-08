import torch
import torch.nn as nn
import torch.nn.functional as F

class IKNet(nn.Module):
    def __init__(self, input_size, num_keywords=17, embedding_dim=768,
                 hidden_size=384, num_layers=1, output_size=1, dropout=0.2):
        super().__init__()
        self.num_keywords = num_keywords
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.keyword_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_keywords)
        ])

        self.keyword_gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size * 2,
            batch_first=True
        )


        self.fuse_proj = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x_price, x_emb):
        B, K, D = x_emb.shape

        projected = [self.keyword_proj[i](x_emb[:, i, :]) for i in range(K)]  
        x_emb_seq = torch.stack(projected, dim=1)
        _, emb_feat = self.keyword_gru(x_emb_seq)    
        emb_feat = emb_feat.squeeze(0)                

        lstm_out, _ = self.lstm(x_price)
        lstm_feat = lstm_out[:, -1, :]                

        combined = torch.cat((lstm_feat, emb_feat), dim=1)  
        fused = self.fuse_proj(combined)                   

        output = self.output_layer(fused)
        return output

