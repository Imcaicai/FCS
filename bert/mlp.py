from transformers import BertModel, BertConfig
import torch
import torch.nn as nn


class mlp(nn.Module):
    def __init__(self, in_features=768, out_features=13) -> None:
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_features=in_features,out_features=1024),\
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024,out_features=512),\
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512,out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512,out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256,out_features=out_features)
        )

    def forward(self, x):
        return self.net(x)
