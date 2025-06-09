"""Evaluate the trained LSTM‑memory controller on the PrefEval test split.
Assumes:
*   `memory_controller.pt`   – contains {"lstm", "proj"} keys
*   `mlp_head_only.pt`      – frozen preference classifier head
*   `config.py`             – defines .pretrained_bert and .pretrained_llm
*   No further training; everything runs in eval mode only.
"""
import os, torch, json
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from models import BertMLPClassifier
from datasets import load_dataset
from tqdm import tqdm
import config

# ── Runtime / model hyper‑params ─────────────────────────────────
DEV   = torch.device("mps" if torch.mps.is_available() else "cpu")
DTYPE = torch.float16 if DEV.type == "cuda" else torch.float32
THRESH = 0.5
HIDDEN_DIM = 512
MAX_LEN_TOK = 256

# ── Load frozen preference classifier ───────────────────────────
bert_tok = AutoTokenizer.from_pretrained(config.pretrained_bert)
clf = BertMLPClassifier(pretrained_bert=config.pretrained_bert).to(DEV)
clf.load_state_dict(torch.load("mlp_head_only.pt", map_location=DEV), strict=False)
clf.eval(); [p.requires_grad_(False) for p in clf.parameters()]
CLS_DIM = clf.bert.config.hidden_size

# ── Load frozen LLM (no gradients) ──────────────────────────────
llm = AutoModelForCausalLM.from_pretrained(config.pretrained_llm, torch_dtype=DTYPE, use_cache=False).to(DEV)
llm.gradient_checkpointing_enable(); llm.eval(); llm.requires_grad_(False)
llm_tok = AutoTokenizer.from_pretrained(config.pretrained_llm)
llm_tok.pad_token = llm_tok.eos_token
EMB_DIM = llm.config.hidden_size

# ── Load trained memory controller weights ─────────────────────
ckpt = torch.load("trained_lstm.pt", map_location=DEV)

lstm = nn.LSTM(CLS_DIM, HIDDEN_DIM, batch_first=True).to(DEV)
proj = nn.Linear(HIDDEN_DIM, EMB_DIM).to(DEV)

lstm.load_state_dict(ckpt["lstm"], strict=True)
proj.load_state_dict(ckpt["proj"], strict=True)

lstm.eval(); proj.eval(); [p.requires_grad_(False) for p in lstm.parameters()+list(proj.parameters())]

# ── Helper for a single soft‑prompted example ───────────────────
def build_example(memory, prompt_text, target_text):
    enc = llm_tok(prompt_text, return_tensors="pt", truncation=True,
                  padding="max_length", max_length=MAX_LEN_TOK).to(DEV)
    tok_e = llm.model.embed_tokens(enc.input_ids).to(DTYPE)
    sp_tok = proj(memory).unsqueeze(0).unsqueeze(1).to(DTYPE)
    full   = torch.cat([sp_tok, tok_e], 1)  # [1, L+1, D]

    tgt_ids = llm_tok(target_text, return_tensors="pt").input_ids.to(DEV)[0]
    labels  = torch.cat([torch.tensor([llm_tok.pad_token_id], device=DEV), tgt_ids])
    # pad labels right to match full seq_len
    if labels.size(0) < full.size(1):
        pad = torch.full((full.size(1) - labels.size(0),), llm_tok.pad_token_id, device=DEV)
        labels = torch.cat([labels, pad])
    else:
        labels = labels[: full.size(1)]
    return full, labels.unsqueeze(0)

# ── PrefEval evaluation loop ───────────────────────────────────
def evaluate_prefeval():
    ds = load_dataset("prefeval", split="test")
    total_loss, n_conv = 0.0, 0
    for ex in tqdm(ds, desc="PrefEval"):
        turns = [(d["assistant"], d["user"]) for d in ex["dialog"]]
        ids, masks = [], []
        for a, u in turns:
            enc = bert_tok(a, u, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            ids.append(enc.input_ids); masks.append(enc.attention_mask)
        ids   = torch.cat(ids, 0).to(DEV)
        masks = torch.cat(masks, 0).to(DEV)
        with torch.no_grad():
            logits, cls = clf(ids, masks, return_embedding=True)
        pref_mask = (torch.sigmoid(logits) > THRESH).cpu()

        h = torch.zeros(1, 1, HIDDEN_DIM, device=DEV); c = torch.zeros_like(h)
        embs, labs = [], []
        for t, (a_txt, u_txt) in enumerate(turns):
            if pref_mask[t]:
                _, (h, c) = lstm(cls[t].unsqueeze(0).unsqueeze(0), (h, c))
            prompt = f"<|system|>\nAgent: {a_txt}\nUser:"
            emb, lab = build_example(h.squeeze(0).squeeze(0), prompt, u_txt)
            embs.append(emb); labs.append(lab)
        embeds = pad_sequence([e.squeeze(0) for e in embs], batch_first=True, padding_value=0.0).to(DTYPE)
        labels = pad_sequence([l.squeeze(0) for l in labs], batch_first=True, padding_value=llm_tok.pad_token_id)
        with torch.no_grad():
            loss = llm(inputs_embeds=embeds, labels=labels).loss
        total_loss += loss.item(); n_conv += 1
    ppl = torch.exp(torch.tensor(total_loss / n_conv))
    print(f"PrefEval loss: {total_loss / n_conv:.4f} | perplexity: {ppl:.2f}")

if __name__ == "__main__":
    evaluate_prefeval()
