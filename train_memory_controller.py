import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from models import MemoryController
import config
import json
from prepare_lstm_dataset import extract_user_turns_with_context

# ===== Dataset Definition =====
class OSSA1Dataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=256):
        self.examples = []
        with open(file_path, "r") as f:
            for line in f:
                ex = json.loads(line)
                self.examples.append(ex)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        print(ex)
        agent = ex["agent"]
        user = ex["user"]
        speaker = ex["speaker"]

        encoded_agent = self.tokenizer(agent, padding="max_length", truncation=True,
                                       max_length=self.max_len, return_tensors="pt")
        encoded_user = self.tokenizer(user, padding="max_length", truncation=True,
                                      max_length=self.max_len, return_tensors="pt")

        return {
            "input_ids": encoded_user["input_ids"].squeeze(0),
            "attention_mask": encoded_user["attention_mask"].squeeze(0),
            "target_ids": encoded_user["input_ids"].squeeze(0),
            "prev_agent": encoded_agent["input_ids"].squeeze(0),
            "prev_agent_mask": encoded_agent["attention_mask"].squeeze(0),
            "speaker": torch.tensor(1 if ex["speaker"] == "user" else 0, dtype=torch.long)
        }


# ===== Train and Eval Loops =====
def train(controller, llm, tokenizer, dataloader, optimizer, device):
    controller.train()
    total_loss = 0
    memory_state = None

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_ids = batch["target_ids"].to(device)
        prev_agent_ids = batch["prev_agent"].to(device)
        prev_agent_mask = batch["prev_agent_mask"].to(device)
        speakers = batch["speaker"]

        with torch.no_grad():
            token_embeds = llm.model.embed_tokens(input_ids)

        input_embeds = []

        for i in range(input_ids.size(0)):
            if speakers[i].item() == 1:
                soft_prompt, memory_state, _ = controller(
                    prev_agent_ids[i].unsqueeze(0),
                    input_ids[i].unsqueeze(0),
                    memory_state
                )
                emb = torch.cat([soft_prompt.unsqueeze(1), token_embeds[i:i+1, :-1, :]], dim=1)
            else:
                emb = token_embeds[i:i+1, :-1, :]
            input_embeds.append(emb)

        input_embeds = torch.cat(input_embeds, dim=0)
        targets = target_ids[:, :input_embeds.size(1)].contiguous()

        output = llm(inputs_embeds=input_embeds)
        logits = output.logits[:, -targets.size(1):, :]

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=tokenizer.pad_token_id
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(controller, llm, tokenizer, dataloader, device):
    controller.eval()
    total_loss = 0
    memory_state = None
    correct, total = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            prev_agent_ids = batch["prev_agent"].to(device)
            prev_agent_mask = batch["prev_agent_mask"].to(device)
            speakers = batch["speaker"]

            print(speakers)

            token_embeds = llm.model.embed_tokens(input_ids)
            input_embeds = []

            for i in range(input_ids.size(0)):
                if speakers[i].item() == 1:
                    soft_prompt, memory_state, _ = controller(
                        prev_agent_ids[i].unsqueeze(0),
                        input_ids[i].unsqueeze(0),
                        memory_state
                    )
                    emb = torch.cat([soft_prompt.unsqueeze(1), token_embeds[i:i+1, :-1, :]], dim=1)
                else:
                    emb = token_embeds[i:i+1, :-1, :]
                input_embeds.append(emb)

            input_embeds = torch.cat(input_embeds, dim=0)
            targets = target_ids[:, :input_embeds.size(1)].contiguous()

            output = llm(inputs_embeds=input_embeds)
            logits = output.logits[:, -targets.size(1):, :]

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            total_loss += loss.item()

            pred_ids = logits.argmax(dim=-1)
            correct += (pred_ids == targets).masked_fill(targets == tokenizer.pad_token_id, False).sum().item()
            total += (targets != tokenizer.pad_token_id).sum().item()

    return total_loss / len(dataloader), correct / total if total > 0 else 0


# ===== Main =====
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_bert)
    llm = AutoModelForCausalLM.from_pretrained(config.pretrained_llm).to(device)
    for p in llm.parameters():
        p.requires_grad = False

    train_dataset = OSSA1Dataset("dataset/train_lstm.jsonl", tokenizer)
    val_dataset = OSSA1Dataset("dataset/val_lstm.jsonl", tokenizer)
    test_dataset = OSSA1Dataset("dataset/test_lstm.jsonl", tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    test_loader = DataLoader(test_dataset, batch_size=4)

    llm_embed_dim = llm.model.embed_tokens.embedding_dim
    controller = MemoryController(pretrained_classifier_path="mlp_head_only.pt", output_embed_dim=llm_embed_dim).to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=1e-4)

    best_val_loss = float("inf")
    no_improve_count = 0
    patience = 3
    model_save_path = "best_memory_model.pt"

    for epoch in range(20):
        train_loss = train(controller, llm, tokenizer, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(controller, llm, tokenizer, val_loader, device)

        print(f"Epoch {epoch+1}: Train loss={train_loss:.4f} | Val loss={val_loss:.4f} | Val acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            no_improve_count = 0
            torch.save(controller.state_dict(), model_save_path)
            print("\U0001F4BE Model saved.")
        else:
            no_improve_count += 1
            print(f"\u26A0\uFE0F No improvement. Patience: {no_improve_count}/{patience}")
            if no_improve_count >= patience:
                print("\u23F9\uFE0F Early stopping.")
                break

    controller.load_state_dict(torch.load(model_save_path))
    test_loss, test_acc = evaluate(controller, llm, tokenizer, test_loader, device)
    print(f"\n✅ Final test loss={test_loss:.4f} | test acc={test_acc:.4f}")

if __name__ == "__main__":
    main()
