import json
with open("dataset/merged_negative.jsonl") as f:
    for line in f:
        data = json.loads(line.strip())
        print(data["agent"])
        print(data["user"])
        print(data["preference"])
        print("---")
