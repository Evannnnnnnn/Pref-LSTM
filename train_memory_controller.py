
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from models import MemoryController

class OSSA1Dataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        user_input = example["input"]
        target = example["target"]

        encoded_input = self.tokenizer(user_input, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        encoded_target = self.tokenizer(target, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")

        return {
            "input_ids": encoded_input["input_ids"].squeeze(0),
            "attention_mask": encoded_input["attention_mask"].squeeze(0),
            "target_ids": encoded_target["input_ids"].squeeze(0)
        }

def train_memory_controller(ossa_data, epochs=3, batch_size=8, device="cuda" if torch.cuda.is_available() else "cpu"):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    llm_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    llm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1").to(device)
    for p in llm.parameters():
        p.requires_grad = False

    dataset = OSSA1Dataset(ossa_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    controller = MemoryController(pretrained_classifier_path="/path/to/frozen_bert_classifier.pt").to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=1e-4)

    for epoch in range(epochs):
        controller.train()
        total_loss = 0
        memory_state = None

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)

            soft_prompt, memory_state, _ = controller(input_ids, attention_mask, memory_state)

            with torch.no_grad():
                target_embeds = llm.model.embed_tokens(target_ids)  # (B, T, E)

            input_embeds = torch.cat([soft_prompt.unsqueeze(1), target_embeds[:, :-1, :]], dim=1)  # prepend prompt

            output = llm(inputs_embeds=input_embeds)
            logits = output.logits[:, -target_ids.shape[1]:, :]

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=llm_tokenizer.pad_token_id)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
