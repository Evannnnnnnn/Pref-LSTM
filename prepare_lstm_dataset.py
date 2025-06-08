import os, json, random, time, re, sys, math
from pathlib import Path
import time
from datasets import load_dataset

OUTPUT_PATH = Path("dataset/full_lstm_dataset.jsonl")

def save_jsonl(obj:dict, path:Path=OUTPUT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as f:
        json.dump(obj, f)
        f.write("\n")


# ===== Load & Preprocess Dataset =====
def extract_user_turns_with_context(dataset_split):
    id_to_msg = {m["message_id"]: m for m in dataset_split if m["lang"] == "en"}
    examples = []

    for msg in dataset_split:
        if msg["parent_id"] is None:
            continue
        parent = id_to_msg.get(msg["parent_id"])
        if not parent or parent["lang"] != "en":
            continue

        if parent["role"] == "assistant" and msg["role"] == "prompter":
            speaker = "user"
        elif parent["role"] == "prompter" and msg["role"] == "assistant":
            speaker = "assistant"
        else:
            continue

        save_jsonl({"agent": parent["text"].strip(), "user": msg["text"].strip(), "speaker": speaker})

if __name__ == "__main__":
    # Load the dataset
    raw_dataset = load_dataset("OpenAssistant/oasst1", split="train")
    extract_user_turns_with_context(raw_dataset)