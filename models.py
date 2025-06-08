import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score
import json
import random
import config

import torch.nn as nn
from transformers import BertModel

pretrained_model_name = "prajjwal1/bert-mini"


class BertMLPClassifier(nn.Module):
    def __init__(self, pretrained_model_name=pretrained_model_name, hidden_dim=512, dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)

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

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]  # Use [CLS] token
        logits = self.mlp_head(cls_token)
        return logits.squeeze(-1)

    
# Memory controller
class MemoryController(nn.Module):
    def __init__(self, pretrained_classifier_path, embed_dim=768, memory_dim=768):
        super(MemoryController, self).__init__()

        # Load frozen classifier from .pt
        classifier = BertMLPClassifier(pretrained_model_name=config.pretrained_bert)
        mlp_state = torch.load(pretrained_classifier_path, map_location="cpu")
        classifier.mlp_head.load_state_dict(mlp_state)
        # Freeze the classifier
        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad = False
        self.bert_classifier = classifier

        self.pref_proj = nn.Linear(embed_dim, memory_dim)
        self.W_MM = nn.Linear(memory_dim, memory_dim)
        self.W_EM = nn.Linear(memory_dim, memory_dim)
        self.bias = nn.Parameter(torch.zeros(memory_dim))
        self.init_memory = nn.Parameter(torch.zeros(memory_dim))
        self.prompt_proj = nn.Linear(memory_dim, embed_dim)


    def update_memory(self, x_embedding, prev_memory, is_preference):
        f_t = torch.sigmoid(self.W_MM(prev_memory) + self.W_EM(x_embedding) + self.bias)
        updated = f_t * prev_memory + (1 - f_t) * x_embedding
        memory_t = is_preference.unsqueeze(-1) * updated + (1 - is_preference.unsqueeze(-1)) * prev_memory
        return memory_t

    def forward(self, input_ids, attention_mask, prev_memory=None):
        B = input_ids.size(0)
        with torch.no_grad():
            logits = self.bert_classifier(input_ids, attention_mask)
            is_preference = (torch.sigmoid(logits) > 0.5).float()
            cls_token = self.bert_classifier.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

        x_embedding = self.pref_proj(cls_token)
        if prev_memory is None:
            prev_memory = self.init_memory.expand(B, -1)

        updated_memory = self.update_memory(x_embedding, prev_memory, is_preference)
        soft_prompt = self.prompt_proj(updated_memory)
        return soft_prompt, updated_memory, is_preference