import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BertMLPClassifier(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased", hidden_dim=256):
        super(BertMLPClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.mlp_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)  # Binary classification
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]  # [CLS] token representation
        logits = self.mlp_head(cls_token)
        return logits.squeeze(-1)  # Shape: (batch_size,)
