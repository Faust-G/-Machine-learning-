
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm


class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        self.relu  = nn.ReLU()
        self.adapt_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.title_emb = nn.Embedding(n_tokens, embedding_dim=hid_size)
        self.title_conv1 = nn.Conv1d(hid_size, concat_number_of_features//3, 2)        
        
        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        self.full_emb_conv1 = nn.Conv1d(hid_size, concat_number_of_features//3, 2)
        
        self.category_out = nn.Linear(n_cat_features, concat_number_of_features//3)

        
        self.inter_dense = nn.Linear(in_features=concat_number_of_features, out_features=hid_size*2)
        self.final_dense = nn.Linear(in_features=hid_size*2, out_features=1)

        

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        title_beg = self.title_emb(input1).permute((0, 2, 1))
        title = self.relu(self.title_conv1(title_beg))
        title = self.adapt_avg_pool(title)
        
        full_beg = self.full_emb(input2).permute((0, 2, 1))
        full = self.relu(self.full_emb_conv1(full_beg))
        full = self.adapt_avg_pool(full)       
        
        category = self.relu(self.category_out(input3))      
        
        concatenated = torch.cat(
            [
            title.view(title.size(0), -1),
            full.view(full.size(0), -1),
            category.view(category.size(0), -1)
            ],
            dim=1)
        
        out = self.relu(self.inter_dense(concatenated))
        out = self.final_dense(out)
        return out