import json
import random

input_file = "dataset/merged_dataset.jsonl"
train_file = "dataset/train.jsonl"
val_file = "dataset/val.jsonl"
test_file = "dataset/test.jsonl"

# Load and shuffle all data
with open(input_file, "r") as f:
    data = [json.loads(line) for line in f if line.strip()]

random.shuffle(data)

# Compute split sizes
total = len(data)
train_end = int(0.8 * total)
val_end = int(0.9 * total)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

# Save helper
def save_jsonl(filename, examples):
    with open(filename, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

# Write files
save_jsonl(train_file, train_data)
save_jsonl(val_file, val_data)
save_jsonl(test_file, test_data)

print(f"✅ Saved {len(train_data)} to {train_file}")
print(f"✅ Saved {len(val_data)} to {val_file}")
print(f"✅ Saved {len(test_data)} to {test_file}")
