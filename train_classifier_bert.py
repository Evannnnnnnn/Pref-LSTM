import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from models import BertMLPClassifier  # Import your classifier from models.py
from sklearn.metrics import accuracy_score
import json
import random

# ==== Dataset ====
pretrained_model_name = "prajjwal1/bert-medium"

class PreferenceDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=200):
        self.examples = []
        with open(file_path, "r") as f:
            for line in f:
                ex = json.loads(line)
                self.examples.append(ex)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        inputs = self.tokenizer(
            ex["agent"],
            ex["user"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(ex["label"], dtype=torch.float)
        }

# ==== Training Loop ====

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    preds, targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            preds.extend((logits > 0).int().cpu().numpy())
            targets.extend(labels.cpu().numpy())

    acc = accuracy_score(targets, preds)
    return total_loss / len(dataloader), acc

# ==== Main ====

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("🔧 Using device:", device)
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

train_dataset = PreferenceDataset("dataset/train.jsonl", tokenizer)
val_dataset = PreferenceDataset("dataset/val.jsonl", tokenizer)
test_dataset = PreferenceDataset("dataset/test.jsonl", tokenizer)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

model = BertMLPClassifier(pretrained_model_name=pretrained_model_name, dropout_rate=0.3).to(device)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=3e-5,  # Slightly higher since you're training fewer parameters
    weight_decay=0.01
)
# smooth the labeling for less overfitting
class SmoothBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        smoothed = labels * (1 - 0.1) + 0.05
        return self.bce(logits, smoothed)

loss_fn = SmoothBCEWithLogitsLoss()

best_val_loss = float("inf")
best_val_acc = 0
no_improve_count = 0
patience = 3
model_save_path = "bert_mlp_preference_best.pt"

for epoch in range(50):
    train_loss = train(model, train_loader, optimizer, loss_fn, device)
    val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

    print(f"Epoch {epoch+1}: Train loss={train_loss:.4f} | Val loss={val_loss:.4f} | Val acc={val_acc:.4f}")

    improved = False

    if val_loss + 1e-4 < best_val_loss:
        best_val_loss = val_loss
        improved = True
        print("📉 Validation loss improved.")

    if val_acc > best_val_acc + 1e-4:
        best_val_acc = val_acc
        improved = True
        print("📈 Validation accuracy improved.")

    if improved:
        no_improve_count = 0
        torch.save(model.state_dict(), model_save_path)
        torch.save(model.mlp_head.state_dict(), "mlp_head_only.pt")
        print("💾 Model saved.")
    else:
        no_improve_count += 1
        print(f"⚠️ No improvement. Patience counter: {no_improve_count}/{patience}")
        if no_improve_count >= patience:
            print("⏹️ Early stopping triggered (no loss or accuracy improvement).")
            break


# Load best model before testing
model.load_state_dict(torch.load(model_save_path))


test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
print(f"\n✅ Test loss={test_loss:.4f} | Test accuracy={test_acc:.4f}")
