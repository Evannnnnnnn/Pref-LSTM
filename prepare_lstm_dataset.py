
import os, json
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict

OUTPUT_PATH = Path("dataset/paired_conversations.jsonl")

def save_jsonl(obj: dict, path: Path = OUTPUT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as f:
        json.dump(obj, f)
        f.write("\n")

def merge_consecutive_speakers(turns):
    merged = []
    buffer = {"speaker": None, "text": []}
    for turn in turns:
        if turn["speaker"] == buffer["speaker"]:
            buffer["text"].append(turn["text"])
        else:
            if buffer["speaker"] is not None:
                merged.append({
                    "speaker": buffer["speaker"],
                    "text": " ".join(buffer["text"])
                })
            buffer = {"speaker": turn["speaker"], "text": [turn["text"]]}
    if buffer["text"]:
        merged.append({
            "speaker": buffer["speaker"],
            "text": " ".join(buffer["text"])
        })
    return merged

def pair_user_assistant_turns(merged_turns):
    pairs = []
    i = 0
    while i < len(merged_turns) - 1:
        curr = merged_turns[i]
        next_turn = merged_turns[i + 1]
        if curr["speaker"] == "assistant" and next_turn["speaker"] == "user":
            pairs.append({
                "assistant": curr["text"],
                "user": next_turn["text"]
            })
            i += 2
        else:
            i += 1  # skip unmatched pairs
    return pairs

def extract_conversations(dataset_split):
    conv_dict = defaultdict(list)

    for msg in dataset_split:
        if msg["lang"] != "en":
            continue
        conv_id = msg["message_tree_id"]
        conv_dict[conv_id].append(msg)

    for conv_id, messages in conv_dict.items():
        messages.sort(key=lambda x: x["message_id"])
        turns = []
        for msg in messages:
            if msg["role"] not in {"prompter", "assistant"}:
                continue
            speaker = "user" if msg["role"] == "prompter" else "assistant"
            turns.append({"speaker": speaker, "text": msg["text"].strip()})

        if turns and turns[0]["speaker"] == "user":
            turns.insert(0, {"speaker": "assistant", "text": ""})

        merged = merge_consecutive_speakers(turns)
        paired = pair_user_assistant_turns(merged)

        if paired:
            save_jsonl({"conversation_id": conv_id, "turns": paired})

if __name__ == "__main__":
    raw_dataset = load_dataset("OpenAssistant/oasst1", split="train")
    extract_conversations(raw_dataset)