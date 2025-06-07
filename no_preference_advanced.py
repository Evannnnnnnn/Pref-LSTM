# negative_turn_generator.py
"""
Generate a HUGE, extremely diverse set of **non-preference** user turns and save
them to a JSONL file.  

Key features
------------
• 5 token–length buckets covering 1-150 tokens → XS/S/M/L/XL  
• Multiple prompt templates (questions, quotes, jokes, gibberish, etc.)  
• Topic pool is huge & user-extendable.  
• FAISS + Sentence-Transformers filter near-duplicates (cosine > 0.85).  
• Simple lexical guard rejects obvious preference phrases.  
• Quota per bucket (e.g. 2 000) so the dataset ends balanced.

Usage
-----
$ python negative_turn_generator.py  # writes dataset/negative_user.jsonl

Dependencies
------------
> pip install openai tiktoken sentence-transformers faiss-cpu numpy tqdm

Fill in your OPENAI_API_KEY as an env-var or directly.
"""

import os, json, random, time, re, sys, math
from pathlib import Path
from bisect import bisect_right

import numpy as np
import faiss                    # FAISS-CPU
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import tiktoken
import openai

# ----------------------------- CONFIG ------------------------------------ #
OUTPUT_PATH = Path("dataset/negative_user.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Token-length buckets 1-150 → XS, S, M, L, XL
BUCKET_EDGES = [1, 6, 16, 41, 101, 151]              # last edge is exclusive
BUCKET_NAMES = ["XS", "S", "M", "L", "XL"]
BUCKET_QUOTA  = 10                                  # examples per bucket

PREFERENCE_PHRASES = re.compile(r"\b(i\s+like|i\s+love|prefer|would\s+rather|i\s+enjoy)\b", re.I)

# Topic pool (extend anytime)
TOPICS = [
    "hobbies", "movies", "food", "travel", "video games", "music", "books", "pets",
    "space", "quantum physics", "sports trivia", "random history", "riddles",
    "coding", "weather", "mythology", "art", "memes", "poetry", "cryptids",
    "invented languages", "time travel paradoxes", "cheese varieties",
    "types of clouds", "arcane facts", "cats vs dogs", "coffee brewing",
]

# ————— Prompt templates (MANY styles) ———————————————— #
TEMPLATES = [
    # --- Questions -------------------------------------------------------
    """You are a user asking a *weird* question about {topic}. No preferences.
    Output exactly one line:  
    User: <your question>""",

    # --- Short factual statement ----------------------------------------
    """Produce a *one-sentence* factoid related to {topic}. No personal opinion.
    Return:  
    User: <statement>""",

    # --- Tiny interjection / reaction -----------------------------------
    """Generate a super-short reaction (1-3 words) someone might blurt out when {topic} is mentioned.
    Must not show liking/disliking.  
    Format: User: ...""",

    # --- Quote -----------------------------------------------------------
    """Quote a famous or fictional person about {topic}. No first-person pronouns.
    Return exactly:  
    User: "<quote>"""  ,

    # --- Command / instruction ------------------------------------------
    """Write a quirky imperative sentence *telling* someone to do something involving {topic} (no preference).
    Output:  
    User: <command>""",

    # --- Joke ------------------------------------------------------------
    """Write a one-line pun or dad-joke about {topic} that reveals no personal taste.  
    Begin with User:""",

    # --- Gibberish -------------------------------------------------------
    """Produce playful nonsense (≤6 tokens) that loosely references {topic}. No real preference.  
    Format: User: ...""",

    # --- Riddle ----------------------------------------------------------
    """Craft a short riddle concerning {topic}. NO clues about liking, just the riddle.  
    Output line starts with User:""",

    # --- Philosophical musing -------------------------------------------
    """Write a contemplative statement (≤25 words) about the nature of {topic}. Avoid first-person preference.  
    Prefix with User:""",
]

# ------------------------- Helpers -------------------------------------- #
encoder   = tiktoken.encoding_for_model("gpt-3.5-turbo")
embedder  = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.IndexFlatIP(384)

def bucket_name(tok_count:int):
    if tok_count >= BUCKET_EDGES[-1]:
        return None
    idx = bisect_right(BUCKET_EDGES, tok_count) - 1
    return BUCKET_NAMES[idx]

def embed(text:str):
    return embedder.encode([text])[0]

def too_similar(vec:np.ndarray, threshold:float=0.85):
    if faiss_index.ntotal == 0:
        return False
    D,_ = faiss_index.search(vec[np.newaxis],1)
    return D[0,0] > threshold

def save_jsonl(obj:dict, path:Path=OUTPUT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as f:
        json.dump(obj, f)
        f.write("\n")

# ------------------------- OpenAI setup --------------------------------- #
client = openai.OpenAI()
MODEL="gpt-4.1-nano"   # cheaper / faster; swap for gpt-4o if desired

# ------------------------- Main loop ------------------------------------ #
length_targets = {name: BUCKET_QUOTA for name in BUCKET_NAMES}
missing = lambda: sum(length_targets.values())

progress = tqdm(total=missing(), desc="Generating", colour="#00ff88")
random.seed(42)

while missing() > 0:
    topic = random.choice(TOPICS)
    template = random.choice(TEMPLATES)

    # Sample creativity & length cap
    temperature = random.uniform(0.7,1.4)
    max_tokens  = random.choice([20,40,60,80])

    prompt = template.format(topic=topic)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        text = response.choices[0].message.content.strip()
        if not text.lower().startswith("user:"):
            continue
        user_line = text[len("User:"):].strip()

        # Preference guard
        if PREFERENCE_PHRASES.search(user_line):
            continue

        tok_len = len(encoder.encode(user_line))
        bucket  = bucket_name(tok_len)
        if bucket is None or length_targets[bucket] == 0:
            continue

        # Similarity guard
        vec = embed(user_line)
        if too_similar(vec):
            continue

        # Passed all checks — store
        save_jsonl({"agent": "", "user": user_line, "label": 0})
        faiss_index.add(vec[np.newaxis])
        length_targets[bucket] -= 1
        progress.update(1)
        progress.set_postfix({b: q for b,q in length_targets.items()})

    except Exception as e:
        tqdm.write(f"⚠️  Misc error: {e}")
        time.sleep(1)

progress.close()
print("✅ Generation complete!  File:", OUTPUT_PATH)
