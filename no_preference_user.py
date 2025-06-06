from openai import OpenAI
import json
import time

client = OpenAI()

def parse_conversation(text):
    """
    Given a string like:
      User: ...
    Returns a dict: { "agent": ..., "user": ... }
    """
    lines = text.strip().splitlines()
    conversation = {}
    for line in lines:
        if line.startswith("User:"):
            conversation["user"] = line[len("User:"):].strip()
    return conversation if "user" in conversation else None


def write_jsonl(convo, output_path):
    with open(output_path, "a", encoding="utf-8") as f:
        json.dump(convo, f)
        f.write("\n")

topics = ["hobbies", "movies", "food", "travel", "video games", "music", "books", "pets"
    "types of friendships", 
    "learning styles", "communication styles", "emotional expression"]

conversations = []

for topic in topics:
    
    prompt = f"""
    Generate a one sentence user's input.

    Constraints:
    - The input must be about the topic {topic}. However, the input should not be about preferences of the user. The input should be questions and instructions or random information.
    - Make each input very very vivid and different from typical examples. As much variation as possible. Try different word choices and sentence structures.

    Format:
    User: ...
    
    There must be one line, starting with "User:".
    """

    for i in range(100):
        print(f"Generating conversation {i+1} for topic '{topic}'...")
        try:
            response = client.responses.create(
                model="gpt-4.1-nano-2025-04-14",
                input=prompt
            )
            parsed = parse_conversation(response.output_text)
            print(f"Response: {response.output_text}")
            if parsed:
                write_jsonl(parsed, "dataset/negative_user.jsonl")
        except Exception as e:
            print(f"Error generating conversation {i+1} for topic '{topic}': {e}")
            time.sleep(1)