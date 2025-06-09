import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer
from models import BertMLPClassifier
import config, json
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────
DEV  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEV.type == "cuda" else torch.float32
HIDDEN_DIM   = 512              # LSTM hidden
MAX_LEN_TOK  = 256              # truncate long prompts
LR           = 1e-4
THRESH       = 0.5
CLIP         = 1.0
EPOCHS       = 20

# Limit how many conversations we actually load (None = no limit)
MAX_TRAIN_CONVS = None   # set to None or adjust as needed
MAX_VAL_CONVS   = None
MAX_TEST_CONVS  = None


# ── Frozen classifier  (BERT‑MLP) ────────────────────────
bert_tok = AutoTokenizer.from_pretrained(config.pretrained_bert)
clf = BertMLPClassifier(pretrained_bert=config.pretrained_bert).to(DEV)
clf.load_state_dict(torch.load("mlp_head_only.pt", map_location=DEV), strict=False)
clf.eval();  [p.requires_grad_(False) for p in clf.parameters()]
CLS_DIM = clf.bert.config.hidden_size

# ── Frozen LLM  ───────────────────────────────────────────
llm = AutoModelForCausalLM.from_pretrained(config.pretrained_llm, torch_dtype=DTYPE, use_cache=False).to(DEV)
llm.gradient_checkpointing_enable(); llm.requires_grad_(False); llm.eval()
llm_tok = AutoTokenizer.from_pretrained(config.pretrained_llm)
llm_tok.pad_token = llm_tok.eos_token
EMB_DIM = llm.config.hidden_size

# ── Trainable memory controller ───────────────────────────
lstm        = nn.LSTM(CLS_DIM, HIDDEN_DIM, batch_first=True).to(DEV)
proj        = nn.Linear(HIDDEN_DIM, EMB_DIM).to(DEV)

# ── Optional: resume from checkpoint ─────────────────────
import os
ckpt_path = "trained_lstm.pt"
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location=DEV)
    if "lstm" in state and "proj" in state:  # new format
        lstm.load_state_dict(state["lstm"], strict=True)
        proj.load_state_dict(state["proj"], strict=True)
        print(f"✅ Loaded checkpoint (new format) from {ckpt_path}")
    else:  # legacy: flat LSTM-only dict
        try:
            lstm.load_state_dict(state, strict=True)
            print(f"⚠️  Loaded legacy checkpoint from {ckpt_path} (LSTM only); projection re‑initialised")
        except Exception as e:
            print("❌ Failed to load checkpoint:", e)

optimiser = torch.optim.Adam(list(lstm.parameters())+list(proj.parameters()), lr=LR)

# ── Helper : classify every turn once  ────────────────────

def preprocess(split, limit=None):
    path = f"dataset/{split}_lstm.jsonl"
    data = []
    with open(path) as f:
        for line in tqdm(f, desc=f"load {split}"):
            if limit is not None and len(data) >= limit:
                break  # stop early to limit dataset size
            ex = json.loads(line)
            turns = [(t["assistant"], t["user"]) for t in ex["turns"]]
            # Build tensors for the whole conversation
            ids, masks = [], []
            for a,u in turns:
                enc = bert_tok(a, u, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
                ids.append(enc.input_ids)
                masks.append(enc.attention_mask)
            ids   = torch.cat(ids ,0).to(DEV)
            masks = torch.cat(masks,0).to(DEV)
            with torch.no_grad():
                logits, cls = clf(ids, masks, return_embedding=True)   # [T]
            pref_mask = (torch.sigmoid(logits) > THRESH).cpu()        # bool [T]
            data.append({
                "cls":  cls.cpu(),            # [T, CLS_DIM]
                "mask": pref_mask,           # [T]
                "turns": turns               # list of tuples
            })
    return data

train_data = preprocess("train", limit=MAX_TRAIN_CONVS)
val_data   = preprocess("val",   limit=MAX_VAL_CONVS)
test_data   = preprocess("test",   limit=MAX_VAL_CONVS)
print("✅ datasets ready →", len(train_data), "train convs /", len(val_data), "val convs")

# ── Build soft‑prompt example from one turn ─────────────────

def build_example(memory, prompt_text, target_text):
    """Return (embeds, labels) with **identical seq_len**.
    Adds right‑padding to labels when the text prompt was padded/truncated.
    """
    # 1) prompt → token embeddings
    enc   = llm_tok(prompt_text, return_tensors="pt", truncation=True,
                    padding="max_length", max_length=MAX_LEN_TOK).to(DEV)
    tok_e = llm.model.embed_tokens(enc.input_ids).to(DTYPE)           # [1, L, D]  (L = MAX_LEN_TOK)

    # 2) soft prompt token from memory
    sp_tok = proj(memory).unsqueeze(0).unsqueeze(1).to(DTYPE)         # [1, 1, D]
    full   = torch.cat([sp_tok, tok_e], dim=1)                        # [1, L+1, D]
    seq_len = full.size(1)

    # 3) build labels (pad for sp_tok at pos 0)
    tgt_ids = llm_tok(target_text, return_tensors="pt").input_ids.to(DEV)[0]  # [T_tgt]
    labels  = torch.cat(
        [torch.tensor([llm_tok.pad_token_id], device=DEV), tgt_ids],
        dim=0
    )  # [1 + T_tgt]

    # 4) right‑pad labels to seq_len so logits & labels match
    if labels.size(0) < seq_len:
        pad_len = seq_len - labels.size(0)
        pad = torch.full((pad_len,), llm_tok.pad_token_id, device=DEV, dtype=torch.long)
        labels = torch.cat([labels, pad], 0)
    else:
        # safety: truncate extra labels (shouldn’t happen)
        labels = labels[:seq_len]

    return full, labels.unsqueeze(0)                                  # keep 3‑D / 2‑D

# ── Training / validation routine  ─────────────────────────

def run_epoch(data, train=True):
    mode = "train" if train else "eval"
    if train: lstm.train(); proj.train()
    else:     lstm.eval();  proj.eval()
    total, steps = 0, 0
    for conv in data:
        cls_seq  = conv["cls"].to(DEV)        # [T, CLS_DIM]
        mask_seq = conv["mask"]               # bool tensor [T]
        turns    = conv["turns"]
        T        = cls_seq.size(0)
        h = torch.zeros(1,1,HIDDEN_DIM, device=DEV)
        c = torch.zeros_like(h)
        examples_emb, examples_lab = [], []
        for t in range(T):
            if mask_seq[t]:                    # update memory only on preference turn
                _,(h,c) = lstm(cls_seq[t].unsqueeze(0).unsqueeze(0), (h,c))
            # build LM example for turn‑t (assistant prompt ends, user reply is target)
            a_text, u_text = turns[t]
            prompt = f"<|system|>\nAgent: {a_text}\nUser:"
            emb, lab = build_example(h.squeeze(0).squeeze(0), prompt, u_text)
            examples_emb.append(emb); examples_lab.append(lab)
        # batch all turns of this conversation (could be long → pad)
        embeds = pad_sequence([e.squeeze(0) for e in examples_emb], batch_first=True, padding_value=0.0).to(DTYPE)
        labels = pad_sequence([l.squeeze(0) for l in examples_lab], batch_first=True, padding_value=llm_tok.pad_token_id)
        # forward
        outs = llm(inputs_embeds=embeds, labels=labels)
        loss = outs.loss
        if torch.isnan(loss) or torch.isinf(loss): continue
        if train:
            optimiser.zero_grad(); loss.backward(); clip_grad_norm_(list(lstm.parameters())+list(proj.parameters()), CLIP); optimiser.step()
        total += loss.item(); steps += 1
    return total/steps if steps else float('nan')

for ep in range(1, EPOCHS+1):
    tr = run_epoch(train_data, train=True)
    vl = run_epoch(val_data,   train=False)
    print(f"Epoch {ep}: train {tr:.4f} | val {vl:.4f}")

te = run_epoch(test_data, train=False)
print(f"test {te:.4f}")

torch.save({"lstm": lstm.state_dict(), "proj": proj.state_dict()}, "trained_lstm.pt")
