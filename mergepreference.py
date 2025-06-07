import re
import json

def split_json_objects(line):
    # Try to split multiple JSON objects crammed into one line
    return re.findall(r'\{.*?\}(?=\s*\{|\s*$)', line)

input_files = ["explicit.jsonl", "implicit_choice.jsonl", "implicit_persona.jsonl"]
output_file = "preference.jsonl"

combined = []

for file in input_files:
    with open(file, "r") as f:
        for i, line in enumerate(f, start=1):
            try:
                json_objects = split_json_objects(line)
                for obj in json_objects:
                    ex = json.loads(obj)
                    if "agent" in ex and "user" in ex:
                        combined.append({
                            "agent": ex["agent"],
                            "user": ex["user"],
                            "preference": 1
                        })
            except json.JSONDecodeError as e:
                print(f"❌ Bad line in {file}, line {i}: {e}")

# Save to JSONL format properly
with open(output_file, "w") as f:
    for ex in combined:
        f.write(json.dumps(ex) + "\n")

print(f"✅ Cleaned and saved {len(combined)} examples to {output_file}")
