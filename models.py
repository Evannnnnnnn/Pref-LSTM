import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score
import json
import random
import config



class BertMLPClassifier(nn.Module):
    def __init__(self, pretrained_bert=config.pretrained_bert, hidden_dim=512, dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_bert)

        # Freeze BERT
        for param in self.bert.parameters():
            param.requires_grad = False

        # MLP head with more dropout
        self.mlp_head = nn.Sequential(
            nn.Dropout(dropout_rate),  # Dropout right after BERT CLS token
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout after hidden layer too
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_ids, attention_mask, return_embedding=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]  # [CLS] token
        logits = self.mlp_head(cls_token)
        if return_embedding:
            return logits.squeeze(-1), cls_token
        return logits.squeeze(-1)
