import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from models import BertMLPClassifier
from torch import nn
from config import pretrained_bert, pretrained_llm

# === Constants ===
HIDDEN_DIM = 512
MAX_LEN_TOK = 256
THRESH = 0.5
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEV = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# === Load Models ===
bert_tok = AutoTokenizer.from_pretrained(pretrained_bert)
clf = BertMLPClassifier(pretrained_bert=pretrained_bert).to(DEV)
clf.load_state_dict(torch.load("mlp_head_only.pt", map_location=DEV), strict=False)
clf.eval(); [p.requires_grad_(False) for p in clf.parameters()]
CLS_DIM = clf.bert.config.hidden_size

llm_tok = AutoTokenizer.from_pretrained(pretrained_llm)
llm = AutoModelForCausalLM.from_pretrained(pretrained_llm, torch_dtype=DTYPE, use_cache=False).to(DEV)
llm_tok.pad_token = llm_tok.eos_token
llm.eval(); llm.requires_grad_(False)

lstm = nn.LSTM(CLS_DIM, HIDDEN_DIM, batch_first=True).to(DEV)
proj = nn.Linear(HIDDEN_DIM, llm.config.hidden_size).to(DEV)
ckpt = torch.load("trained_lstm.pt", map_location=DEV)
lstm.load_state_dict(ckpt["lstm"])
proj.load_state_dict(ckpt["proj"])

# === Load Data ===
with open("education_learning_styles.json", "r") as f:
    data = json.load(f)

# === Build LSTM Memory from N Preference Turns ===
def build_memory(preference_examples):
    h = torch.zeros(1, 1, HIDDEN_DIM, device=DEV)
    c = torch.zeros_like(h)

    for ex in preference_examples:
        pref_text = ex["preference"]
        enc = bert_tok("Agent:", pref_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(DEV)
        with torch.no_grad():
            logits, cls = clf(enc.input_ids, enc.attention_mask, return_embedding=True)
        is_pref = torch.sigmoid(logits) > THRESH
        if is_pref:
            _, (h, c) = lstm(cls.unsqueeze(1), (h, c))
    return h.squeeze(0)

# === Generate Response for Final Query with Memory Soft Prompt ===
def generate_response_with_memory(memory, query):
    prompt = f"<|system|>\nUser: {query}\nAssistant:"
    enc = llm_tok(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN_TOK).to(DEV)
    tok_e = llm.model.embed_tokens(enc.input_ids).to(DTYPE)
    soft_token = proj(memory).unsqueeze(1).to(DTYPE)  # from [1, D] → [1, 1, D]
    # Just pass a known-good token like BOS (beginning-of-sequence)

    soft_prompt = torch.cat([soft_token, tok_e], dim=1)

    # Pad attention mask to match the new sequence length (L+1)
    attention_mask = torch.cat([
        torch.ones((1, 1), dtype=torch.long, device=DEV),  # for soft token
        enc.attention_mask
    ], dim=1)

    gen_ids = llm.generate(
        inputs_embeds=soft_prompt,
        attention_mask=attention_mask,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        pad_token_id=llm_tok.pad_token_id,
        eos_token_id=llm_tok.eos_token_id
    )

    return llm_tok.decode(gen_ids[0], skip_special_tokens=True).split("Assistant:")[-1].strip()

# === Main Run ===
N = 5  # number of preference turns to feed into LSTM memory
memory = build_memory(data[:N])
query = data[N]["question"]
print("💬 Final question:", query)
response = generate_response_with_memory(memory, query)
print("🤖 Response:", response)
