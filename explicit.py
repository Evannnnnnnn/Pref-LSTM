from openai import OpenAI
import json
import time

client = OpenAI()

def parse_conversation(text):
    """
    Given a string like:
      Agent: ...
      User: ...
    Returns a dict: { "agent": ..., "user": ... }
    """
    lines = text.strip().splitlines()
    conversation = {}
    for line in lines:
        if line.startswith("Agent:"):
            conversation["agent"] = line[len("Agent:"):].strip()
        elif line.startswith("User:"):
            conversation["user"] = line[len("User:"):].strip()
    return conversation if "agent" in conversation and "user" in conversation else None


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
    Generate a one-turn conversation between an AI agent and a user.

    Constraints:
    - The agent must ask about the user's long-term preference regarding the topic {topic}.
    - The user must reply with a clear, expressive, and specific long-term preference.
    - Make each conversation very very vivid and different from typical examples. As much variation as possible.
    - Change user's response and word choices to be more varied. User should use more verbs to describe their feelings.

    Format:
    Agent: ...
    User: ...
    
    There must be two lines, one starting with "Agent:" and one starting with "User:".
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
                write_jsonl(parsed, "explicit.jsonl")
        except Exception as e:
            print(f"Error generating conversation {i+1} for topic '{topic}': {e}")
            time.sleep(1)