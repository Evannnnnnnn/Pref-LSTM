#!/usr/bin/env python
# evaluate_explicit_pref.py  – Use first N turns for memory, test on turn N

"""
Example usage
-------------
$ python evaluate_explicit_pref.py \
        --folder explicit_preference \
        --pref_turns 10 \
        --threshold 0.5 \
        --max_new 64 \
        --model_gpt gpt-4o-mini
"""

import os, sys, time, json, glob, argparse
from pathlib import Path
from typing import List, Tuple
import torch, openai, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── local config / models ---------------------------------------------------
import config                              # has .pretrained_bert / .pretrained_llm
from models import BertMLPClassifier       # must exist!

# ── CLI ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--folder", default="explicit_preference")
parser.add_argument("--pref_turns", type=int, default=10,
                    help="How many turns to feed into LSTM (N)")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--max_new", type=int, default=64)
parser.add_argument("--model_gpt", default="gpt-4o-mini")
args = parser.parse_args()

# ── API key -----------------------------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    sys.exit("❌  Please set OPENAI_API_KEY")

# ── Device ------------------------------------------------------------------
DEV = (torch.device("cuda") if torch.cuda.is_available() else
       torch.device("mps")  if torch.backends.mps.is_available() else
       torch.device("cpu"))
DTYPE = torch.float16 if DEV.type == "cuda" else torch.float32
print(f"🔧 Device: {DEV} ({DTYPE})")

# ── BERT‑MLP preference classifier (frozen) ---------------------------------
bert_tok = AutoTokenizer.from_pretrained(config.pretrained_bert)
clf = BertMLPClassifier(pretrained_bert=config.pretrained_bert).to(DEV)
clf.load_state_dict(torch.load("mlp_head_only.pt", map_location=DEV), strict=False)
clf.eval(); [p.requires_grad_(False) for p in clf.parameters()]
CLS_DIM = clf.bert.config.hidden_size

# ── Mistral‑7B (full precision for maximum compatibility) -------------------
llm_tok = AutoTokenizer.from_pretrained(config.pretrained_llm)
llm_tok.pad_token = llm_tok.eos_token
print("⚙️  Loading Mistral …")
llm = (AutoModelForCausalLM
       .from_pretrained(config.pretrained_llm,
                        torch_dtype=DTYPE, use_cache=True)
       .to(DEV).eval())
llm.requires_grad_(False)
EMB_DIM = llm.config.hidden_size
print("✅ Mistral ready.")

# ── LSTM memory controller (frozen) -----------------------------------------
lstm = nn.LSTM(CLS_DIM, 512, batch_first=True).to(DEV)
proj = nn.Linear(512, EMB_DIM).to(DEV)
ckpt = torch.load("trained_lstm.pt", map_location=DEV)
lstm.load_state_dict(ckpt["lstm"]); proj.load_state_dict(ckpt["proj"])
lstm.eval(); proj.eval()
for p in (*lstm.parameters(), *proj.parameters()):
    p.requires_grad_(False)

# ── helper: build prompt -----------------------------------------------------
def build_inputs(mem: torch.Tensor, question: str):
    """Return (inputs_embeds, attn_mask) for llm.generate()."""
    prompt = ("<|system|>\nYou are a helpful assistant who respects all learned user preferences.\n"
              f"<|user|>\n{question}\n<|assistant|> ")   # trailing space ⇒ model must respond
    enc = llm_tok(prompt, return_tensors="pt").to(DEV)
    tok_emb = llm.model.embed_tokens(enc.input_ids).to(DTYPE)   # (1, L, D)
    soft_tok = proj(mem).unsqueeze(0).unsqueeze(1)              # (1, 1, D)
    inputs   = torch.cat([soft_tok, tok_emb], 1)                # (1, L+1, D)
    mask     = torch.ones(inputs.size()[:2], dtype=torch.long, device=DEV)
    return inputs, mask

# ── helper: GPT‑4o grader ----------------------------------------------------
def grade(preferences: str, explanation: str, q: str, ans: str) -> bool:
    prompt = (f"User preference(s): {preferences}\n"
              f"Question (to assistant): {q}\n"
              f"Assistant answer: {ans}\n"
              f"Explanation of correct behaviour: {explanation}\n\n"
              "Does the assistant answer respect ALL preferences "
              "and align with the explanation? Reply only 'yes' or 'no'.")
    resp = openai.chat.completions.create(
        model=args.model_gpt,
        messages=[{"role":"system","content":"You are a strict evaluator. Reply only 'yes' or 'no'."},
                  {"role":"user","content":prompt}],
        temperature=0, timeout=10
    )
    return resp.choices[0].message.content.strip().lower().startswith("y")

# ── evaluation loop ---------------------------------------------------------
files = sorted(glob.glob(str(Path(args.folder) / "*.json")))
if not files:
    sys.exit(f"❌  No JSON found in {args.folder}")

total=correct=pref_hits=0
t_all=time.time()

for fp in tqdm(files, desc="evaluate"):
    conv = json.load(open(fp))
    if not isinstance(conv, list) or len(conv) < args.pref_turns+1:
        print(f"⚠️  {Path(fp).name}: needs at least {args.pref_turns+1} turns"); continue

    # first N turns → memory
    memory_turns = conv[:args.pref_turns]
    test_turn    = conv[args.pref_turns]   # turn N+1
    question     = test_turn["question"]
    explanation  = test_turn["explanation"]

    # build memory
    h = torch.zeros(1,1,512, device=DEV); c=torch.zeros_like(h)
    collected_prefs=[]
    for t in memory_turns:
        pref_text = t["preference"]
        enc = bert_tok("", pref_text, return_tensors="pt",
                       padding="max_length", truncation=True, max_length=128).to(DEV)
        with torch.no_grad():
            logit, emb = clf(enc.input_ids, enc.attention_mask, return_embedding=True)
        if torch.sigmoid(logit).item() >= args.threshold:
            pref_hits+=1
            collected_prefs.append(pref_text)
            _, (h,c)=lstm(emb.unsqueeze(1), (h,c))

    # ask Mistral
    inputs,mask = build_inputs(h.squeeze(0).squeeze(0), question)
    t0=time.time()
    with torch.no_grad():
        out = llm.generate(inputs_embeds=inputs, attention_mask=mask,
                           max_new_tokens=args.max_new, min_new_tokens=5,
                           do_sample=False)
    gen_sec=time.time()-t0
    answer = llm_tok.decode(out[0, inputs.size(1):], skip_special_tokens=True).strip()

    # grade
    ok = grade(" ".join(collected_prefs), explanation, question, answer)
    correct+=int(ok); total+=1

    print(f"\n{Path(fp).stem}: "
          f"{'✅' if ok else '❌'} | gen {gen_sec:4.1f}s\nQ: {question[:60]}…"
          f"\nA: {answer[:80]}{'…' if len(answer)>80 else ''}\n")

elapsed=time.time()-t_all
print("\n────────  Summary ────────")
print(f"Conversations eval'd : {total}")
print(f"Preference turns fed : {args.pref_turns} (hits {pref_hits})")
print(f"Accuracy             : {correct/total:.2%}" if total else "n/a")
print(f"Avg time / conv      : {elapsed/total:.2f}s" if total else "")
