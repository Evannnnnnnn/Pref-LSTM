import json
import random

# Input files
positive_file = "dataset/preference.jsonl"
negative_file = "dataset/non_preference.jsonl"
output_file = "dataset/merged_dataset.jsonl"

# Load both datasets
combined = []

for file in [positive_file, negative_file]:
    with open(file, "r") as f:
        for line in f:
            ex = json.loads(line)
            if "agent" in ex and "user" in ex and "label" in ex:
                combined.append(ex)

# Shuffle
random.shuffle(combined)

# Save to new .jsonl file
with open(output_file, "w") as f:
    for ex in combined:
        f.write(json.dumps(ex) + "\n")

print(f"✅ Merged and shuffled {len(combined)} examples into {output_file}")
