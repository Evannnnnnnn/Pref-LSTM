# no_preference_advanced.py
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
$ python no_preference_advanced.py  # writes dataset/non_preference.jsonl

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
import time

# ----------------------------- CONFIG ------------------------------------ #
OUTPUT_PATH = Path("dataset/non_preference.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Token-length buckets 1-150 → XS, S, M, L, XL
BUCKET_EDGES = [1, 6, 16, 41, 101, 151]              # last edge is exclusive
BUCKET_NAMES = ["XS", "S", "M", "L", "XL"]
BUCKET_QUOTA  = 10                                  # examples per bucket

PREFERENCE_PHRASES = re.compile(r"\b(i\s+like|i\s+love|prefer|would\s+rather|i\s+enjoy)\b", re.I)

# Topic pool (extend anytime)
TOPICS = [
    # Original topics
    "hobbies", "movies", "food", "travel", "video games", "music", "books", "pets",
    "space", "quantum physics", "sports trivia", "random history", "riddles",
    "coding", "weather", "mythology", "art", "memes", "poetry", "cryptids",
    "invented languages", "time travel paradoxes", "cheese varieties",
    "types of clouds", "arcane facts", "cats vs dogs", "coffee brewing",

    # Daily Life & Behavior
    "sleep habits", "morning routines", "productivity tools", "procrastination", "fashion styles",
    "cleaning habits", "favorite smells", "texting etiquette", "shopping preferences", "social media",

    # Food & Drink (niche & global)
    "street food", "spicy foods", "vegetarian dishes", "bubble tea", "comfort foods",
    "breakfast cereals", "sushi types", "food delivery apps", "regional snacks", "picnic foods",

    # Tech & Digital Culture
    "keyboard layouts", "smartphone brands", "streaming platforms", "voice assistants", "VR experiences",
    "email vs messaging", "Linux distros", "browser extensions", "AI-generated art", "digital calendars",

    # Arts & Entertainment
    "musical instruments", "theater styles", "animation styles", "fan fiction", "underrated TV shows",
    "audiobooks", "movie remakes", "vinyl records", "YouTube channels", "comedy formats",

    # Nerdy, Abstract, or Meta
    "moral dilemmas", "simulation theory", "the trolley problem", "paradoxes", "aliens",
    "ethical AIs", "ancient civilizations", "algorithm bias", "personality types", "learning styles",

    # Experiences & Vibes
    "solo travel", "rainy days", "beach vs mountains", "airplane seats", "holiday traditions",
    "childhood nostalgia", "first impressions", "hometown pride", "seasonal moods", "long walks"
]

# ————— Prompt templates (MANY styles) ———————————————— #
# TEMPLATES = [
#     # 1 ─── Weird question ──────────────────────────────────────────────
#     """Write two lines about {topic}.
#     Line 1 — Agent: a brief, neutral remark that mentions {topic}.
#     Line 2 — User: a truly odd question about {topic}, 30‑60 words, no personal preference.
#     The result should contain exactly two lines, one starting with "Agent:" and the other with "User:".""",

#     # 2 ─── Short factual statement ─────────────────────────────────────
#     """Compose two lines on {topic}.
#     Line 1 — Agent: a one‑sentence factual statement about {topic}.
#     Line 2 — User: either (a) ≤3 words *or* (b) 15‑30 words of reflection—no opinions.
#     The result should contain exactly two lines, one starting with "Agent:" and the other with "User:".""",

#     # 3 ─── Tiny interjection / reaction ────────────────────────────────
#     """Produce a two‑line micro‑snippet.
#     Line 1 — Agent: mentions {topic} in ≤6 words.
#     Line 2 — User: a 1‑5‑word neutral reaction.
#     The result should contain exactly two lines, one starting with "Agent:" and the other with "User:".""",

#     # 4 ─── Quote ────────────────────────────────────────────────────────
#     """Generate a two‑line quotation exchange on {topic}.
#     Line 1 — Agent: quote a famous or fictional person about {topic}.
#     Line 2 — User: a thoughtful follow‑up comment (20‑40 words), neutral in tone.
#     The result should contain exactly two lines, one starting with "Agent:" and the other with "User:".""",

#     # 5 ─── Command / instruction ───────────────────────────────────────
#     """Create a two‑line snippet involving {topic}.
#     Line 1 — Agent: an imaginative imperative telling someone to do something with {topic}.
#     Line 2 — User: brief (2‑8 words) or elaborated (12‑25 words) acceptance.
#     Exactly two lines beginning with "Agent:" and "User:".""",

#     # 6 ─── Joke ─────────────────────────────────────────────────────────
#     """Craft a two‑line dad‑joke exchange on {topic}.
#     Line 1 — Agent: a single‑line pun or joke about {topic}.
#     Line 2 — User: either ≤3‑word groan/laugh or a 15‑35‑word humorous rant.
#     Must be two lines starting with "Agent:" and "User:".""",

#     # 7 ─── Gibberish ────────────────────────────────────────────────────
#     """Generate playful nonsense in dialogue form that loosely references {topic}.
#     Line 1 — Agent: ≤6 nonsensical tokens including {topic}.
#     Line 2 — User: ≤6 nonsensical tokens in reply.
#     Output exactly two lines prefixed "Agent:" and "User:".""",

#     # 8 ─── Riddle ───────────────────────────────────────────────────────
#     """Write a two‑line riddle concerning {topic}.
#     Line 1 — Agent: a short riddle question about {topic}.
#     Line 2 — User: “What am I?”
#     Output exactly two lines, one "Agent:" and one "User:".""",

#     # 9 ─── Philosophical musing ────────────────────────────────────────
#     """Compose two lines reflecting on {topic}.
#     Line 1 — Agent: a contemplative statement about {topic} (≤25 words).
#     Line 2 — User: a deeper meditation (20‑45 words) that stays preference‑free.
#     Exactly two lines starting with "Agent:" and "User:".""",

#     # 10 ─── Apology exchange ───────────────────────────────────────────
#     """Draft two lines involving {topic}.
#     Line 1 — Agent: a concise apology that references {topic}.
#     Line 2 — User: a forgiving or neutral response, 20‑35 words, no preferences.
#     Return exactly two lines prefixed "Agent:" and "User:".""",

#     # 11 ─── Clarification request ──────────────────────────────────────
#     """Produce a two‑line clarification dialogue on {topic}.
#     Line 1 — Agent: politely asks for clarification about a detail of {topic}.
#     Line 2 — User: provides a detailed clarification (25‑50 words) without stating preferences.
#     Two lines only: "Agent:" then "User:".""",

#     # 12 ─── Hypothetical scenario ──────────────────────────────────────
#     """Construct two lines around a hypothetical {topic} scenario.
#     Line 1 — Agent: presents a brief “Imagine if…” situation about {topic}.
#     Line 2 — User: explores consequences in 30‑55 words, neutral stance.
#     Exactly two lines labelled "Agent:" and "User:".""",

#     # 13 ─── Definition exchange ────────────────────────────────────────
#     """Write two lines defining {topic}.
#     Line 1 — Agent: asks “Can you define {topic}?”
#     Line 2 — User: delivers a clear, dictionary‑style definition (20‑35 words) with no personal views.
#     Must be two lines, "Agent:" then "User:".""",

#     # 14 ─── Brainstorm burst ───────────────────────────────────────────
#     """Generate a two‑line brainstorming snippet about {topic}.
#     Line 1 — Agent: suggests starting an idea session on {topic}.
#     Line 2 — User: fires off a rapid list of 4‑6 comma‑separated ideas in ≤35 words, no preferences.
#     Output exactly two lines starting with "Agent:" and "User:".""",

#     # 15 ─── Troubleshooting Q&A ────────────────────────────────────────
#     """Create a two‑line troubleshooting exchange.
#     Line 1 — Agent: reports a simple issue related to {topic}.
#     Line 2 — User: offers step‑by‑step guidance (25‑45 words) without personal opinion.
#     Return exactly two lines prefixed "Agent:" and "User:".""",

#     # 16 ─── Analogy reflection ─────────────────────────────────────────
#     """Compose two lines that use analogy.
#     Line 1 — Agent: presents an analogy comparing {topic} to something else.
#     Line 2 — User: reflects on the analogy in 20‑35 words, no personal preference.
#     Exactly two lines: "Agent:" first, then "User:".""",

#     # 17 ─── Historical tidbit ──────────────────────────────────────────
#     """Produce a two‑line historical exchange about {topic}.
#     Line 1 — Agent: states a little‑known historical fact (≤25 words) about {topic}.
#     Line 2 — User: asks an intrigued follow‑up question (15‑30 words) without expressing preference.
#     Two lines only, labelled "Agent:" and "User:".""",

#     # 18 ─── Future prediction ──────────────────────────────────────────
#     """Draft a two‑line future‑looking dialogue on {topic}.
#     Line 1 — Agent: makes a bold prediction about {topic} in ≤25 words.
#     Line 2 — User: responds with curiosity or skepticism (25‑45 words) yet stays neutral.
#     Output exactly two lines, first "Agent:", then "User:".""",
# ]


# TEMPLATES = [
#     # 1 ─── Strong adverb, vague human response — everyday ambiguity
#     """Write two lines about {topic}.
# Agent: Say something mundane or factual about {topic}.
# User: Casually use a strong word like "always", "never", "tend to", "usually", or "rarely"—but in a way that reveals no personal stance or consistent behavior. Don't give the example tho, choose something else.
# Output exactly two lines of complete sentences, first "Agent:", then "User:""",

#     # 2 ─── Deflection or avoidance — "I've never really decided..." flavor
#     """Write two lines about {topic}.
# Agent: Ask a casual question or make a remark about {topic}.
# User: Reply vaguely using a line like "I've never really decided on that" or "I usually let others figure it out"—clearly dodging any opinion or decision. Don't give the example tho, choose something else.
# Output exactly two lines of complete sentences, first "Agent:", then "User:""",

#     # 3 ─── Philosophical or paradoxical musing — sounds deep, says nothing
#     """Write two lines about {topic}.
# Agent: Introduce {topic} with a curious statement.
# User: Say something abstract or paradoxical using words like "never" or "always" (e.g., "I never know if I always forget") without revealing preference. Don't give the example tho, choose something else.
# Output exactly two lines of complete sentences, first "Agent:", then "User:""",

#     # 4 ─── Repetitive or deflective rambling — sounds human, stays neutral
#     """Write two lines about {topic}.
# Agent: Mention {topic} in a short statement.
# User: Give a meandering reply using a strong word but without a clear opinion—something like, "I always hear about that, but I’ve never really followed it closely. Don't use the example though, choose something else."
# Output exactly two lines of complete sentences, first "Agent:", then "User: """,

#     # 5 ─── Deflection through social commentary — no opinion given
#     """Write two lines about {topic}.
# Agent: Bring up {topic} in a matter-of-fact way.
# User: Comment on what others tend to do or say (e.g., “I guess I’m open to anything, really.”), while staying neutral yourself. Don't give the example tho, choose something else.
# Output exactly two lines of complete sentences, first "Agent:", then "User:""",

#     """Write two lines about {topic}.
# Agent: Bring up {topic} in a matter-of-fact way.
# User: Comment on what others tend to do or say (e.g., “I don't know”), while staying neutral yourself. Don't give the example tho, choose something else.
# Output exactly two lines of complete sentences, first "Agent:", then "User:"""
# ]

TEMPLATES = [
    """Generate a conversation between an agent and user that is similar to what would happen in real life. Make it about {topic}.
    Line 1 — Agent: outputs random information about {topic}.
    Line 2 — User: responds with a one word answer, either affermative or negative.
    Output exactly two lines of complete sentences, first "Agent:", then "User:"""
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
MODEL="gpt-4.1-mini"   # cheaper / faster; swap for gpt-4o if desired

# ------------------------- Main loop ------------------------------------ #
length_targets = {name: BUCKET_QUOTA for name in BUCKET_NAMES}
missing = lambda: sum(length_targets.values())

progress = tqdm(total=3000, desc="Generating", colour="#00ff88")
random.seed(42)

i = 0;

while i < 3000:
    topic = random.choice(TOPICS)
    template = random.choice(TEMPLATES)

    # Sample creativity & length cap
    temperature = random.uniform(0.7,1.4)
    max_tokens  = random.choice([20,40,60,80])

    prompt = template.format(topic=topic)

    try:
        start = time.time()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        print(time.time() - start)
        text = response.choices[0].message.content.strip()
        print(text)
        lines = text.strip().splitlines()
        conversation = {}
        for line in lines:
            line = line.strip()
            if line.startswith("Agent:") and (line.endswith(".") or line.endswith("!") or line.endswith("?")):
                conversation["agent"] = line[len("Agent:"):].strip()
            elif line.startswith("User:") and (line.endswith(".") or line.endswith("!") or line.endswith("?")):
                conversation["user"] = line[len("User:"):].strip()

        print(conversation)

        # Preference guard
        if PREFERENCE_PHRASES.search(conversation["user"]):
            print("contains preference phrase, skipping")
            continue

        # tok_len = len(encoder.encode(user_line))
        # bucket  = bucket_name(tok_len)
        # if bucket is None or length_targets[bucket] == 0:
        #     print(f"Skipping {user_line!r} (bucket {bucket} full or too long)")
        #     continue

        # Similarity guard
        vec = embed(text)
        if too_similar(vec):
            print(f"Skipping {text!r} (too similar to existing)")
            continue

        # Passed all checks — store
        save_jsonl({"agent": conversation["agent"], "user": conversation["user"], "label": 0})
        faiss_index.add(vec[np.newaxis])
        progress.update(1)
        progress.set_postfix({b: q for b,q in length_targets.items()})

        i += 1

    except Exception as e:
        tqdm.write(f"⚠️  Misc error: {e}")
        time.sleep(1)

progress.close()
print("✅ Generation complete!  File:", OUTPUT_PATH)
