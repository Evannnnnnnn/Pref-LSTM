import json
import random

# Input files
positive_file = "dataset/preference.jsonl"
negative_file = "dataset/non_preference.jsonl"
output_file = "dataset/merged_dataset.jsonl"

# Load both datasets with validation and deduplication
combined = []
seen_lines = set()

for file in [positive_file, negative_file]:
    with open(file, "r") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line in seen_lines:
                continue
            try:
                ex = json.loads(line)
                # Check required keys
                if not all(k in ex for k in ("agent", "user", "label")):
                    continue
                # Make sure label is 0 or 1 (optional strict check)
                if ex["label"] not in [0, 1]:
                    continue
                combined.append(ex)
                seen_lines.add(line)
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping {file} line {i}: {e}")

# Shuffle
random.shuffle(combined)

# Save to new .jsonl file
with open(output_file, "w", encoding="utf-8") as f:
    for ex in combined:
        f.write(json.dumps(ex) + "\n")

print(f"✅ Merged, cleaned, and shuffled {len(combined)} examples into {output_file}")
