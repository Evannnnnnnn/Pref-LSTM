import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from models import BertMLPClassifier
import config
import json
from tqdm import tqdm

# ====== Setup ======
device = torch.device("mps" if torch.mps.is_available() else "cpu")

# Load frozen classifier and tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained(config.pretrained_bert)
classifier = BertMLPClassifier(pretrained_bert=config.pretrained_bert).to(device)
classifier.load_state_dict(torch.load("mlp_head_only.pt", map_location=device), strict=False)
classifier.eval()
for p in classifier.parameters():
    p.requires_grad = False

# Load Mistral LLM + tokenizer
llm = AutoModelForCausalLM.from_pretrained(config.pretrained_llm, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
llm_tokenizer = AutoTokenizer.from_pretrained(config.pretrained_llm)
llm_tokenizer.pad_token = llm_tokenizer.eos_token

# LSTM memory encoder
embedding_dim = classifier.bert.config.hidden_size
hidden_dim = 512
lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True).to(device)
optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# Load datasets
with open("train_lstm.jsonl", "r") as f:
    train_data = [json.loads(line) for line in f]
with open("val_lstm.jsonl", "r") as f:
    val_data = [json.loads(line) for line in f]

# ====== Precompute classifier embeddings (once) ======
def precompute(data):
    precomputed = []
    thresh = 0.5
    for ex in tqdm(data):
        conv = ex["conversation"]
        input_ids_list, attn_masks_list = [], []
        for agent, user in conv:
            enc = bert_tokenizer(agent, user, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            input_ids_list.append(enc["input_ids"])
            attn_masks_list.append(enc["attention_mask"])

        input_ids = torch.cat(input_ids_list, dim=0).to(device)
        attn_mask = torch.cat(attn_masks_list, dim=0).to(device)

        with torch.no_grad():
            logits, cls_embeds = classifier(input_ids, attn_mask, return_embedding=True)
        mask = (torch.sigmoid(logits) > thresh)
        if mask.sum() < 1:
            continue

        keep_indices = torch.nonzero(mask).squeeze(-1).tolist()
        pref_embeds = cls_embeds[mask].detach().cpu()

        t = keep_indices[-1] if len(keep_indices) > 0 else 0
        history = "\n".join([f"<|system|>\nAgent: {a}\nUser: {u}" for a, u in conv[:t+1]])
        next_agent = conv[t][0] if t < len(conv) else ""
        prompt = history + f"\nAgent: {next_agent}\nUser:"
        target = conv[t+1][1] if t+1 < len(conv) else ""

        if target:
            precomputed.append({"pref_embeds": pref_embeds, "prompt": prompt, "target": target})
    return precomputed

train_precomputed = precompute(train_data)
val_precomputed = precompute(val_data)

# ====== Training ======
epochs = 3
batch_size = 2

for epoch in range(epochs):
    total_loss = 0.0
    for i in range(0, len(train_precomputed), batch_size):
        batch = train_precomputed[i:i+batch_size]

        embeds_batch = [torch.stack(item["pref_embeds"]) for item in batch]
        lengths = [x.shape[0] for x in embeds_batch]
        padded = pad_sequence(embeds_batch, batch_first=True).to(device)

        packed = pack_padded_sequence(padded, lengths=lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = lstm(packed)
        soft_prompts = h_n[-1]

        embeds_input, labels_input = [], []
        for sp, item in zip(soft_prompts, batch):
            inp = llm_tokenizer(item["prompt"], return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
            tgt = llm_tokenizer(item["target"], return_tensors="pt").input_ids.to(device)[0]
            tok_embeds = llm.model.embed_tokens(inp.input_ids)
            sp_token = sp.unsqueeze(0).unsqueeze(1)
            full = torch.cat([sp_token, tok_embeds], dim=1)
            embeds_input.append(full)
            labels_input.append(torch.cat([torch.tensor([llm_tokenizer.pad_token_id], device=device), tgt], dim=0).unsqueeze(0))

        embeds_input = torch.cat(embeds_input, dim=0)
        labels_input = torch.cat(labels_input, dim=0)

        outputs = llm(inputs_embeds=embeds_input, labels=labels_input)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / (len(train_precomputed) / batch_size)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

    # ====== Validation ======
    val_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(val_precomputed), batch_size):
            batch = val_precomputed[i:i+batch_size]
            embeds_batch = [torch.stack(item["pref_embeds"]) for item in batch]
            lengths = [x.shape[0] for x in embeds_batch]
            padded = pad_sequence(embeds_batch, batch_first=True).to(device)

            packed = pack_padded_sequence(padded, lengths=lengths, batch_first=True, enforce_sorted=False)
            _, (h_n, _) = lstm(packed)
            soft_prompts = h_n[-1]

            embeds_input, labels_input = [], []
            for sp, item in zip(soft_prompts, batch):
                inp = llm_tokenizer(item["prompt"], return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
                tgt = llm_tokenizer(item["target"], return_tensors="pt").input_ids.to(device)[0]
                tok_embeds = llm.model.embed_tokens(inp.input_ids)
                sp_token = sp.unsqueeze(0).unsqueeze(1)
                full = torch.cat([sp_token, tok_embeds], dim=1)
                embeds_input.append(full)
                labels_input.append(torch.cat([torch.tensor([llm_tokenizer.pad_token_id], device=device), tgt], dim=0).unsqueeze(0))

            embeds_input = torch.cat(embeds_input, dim=0)
            labels_input = torch.cat(labels_input, dim=0)

            outputs = llm(inputs_embeds=embeds_input, labels=labels_input)
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / (len(val_precomputed) / batch_size)
    print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")

# Save trained LSTM
torch.save(lstm.state_dict(), "trained_lstm.pt")
