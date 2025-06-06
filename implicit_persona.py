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
    
    prompt1 = f"""
    Generate a one-turn conversation between an AI agent and a user.

    Constraints:
    - The agent must engage the user's in topics regarding {topic} but without a question.
    - The user must reply with a response that is not a simple yes or no.
    - User can state their preference in a way that is not explicitly asking for it.
    - Make each conversation very very vivid and different from typical examples. As much variation as possible.

    Example conversation:
    Agent: "Absolutely! There are many flexible parenting resources available. You might find podcasts, e-books, or online articles helpful. These can be accessed anytime and allow you to learn at your own pace. Some popular parenting websites also offer downloadable guides or video series. Is there a particular parenting topic you're interested in exploring?",
    User: "I've tried planners, but with the kids' unpredictable schedules, it's hard to stick to them. I usually prefer flexible options that I can fit in whenever I have a free moment. Speaking of which, do you know of any good parenting resources that don't require attending specific classes?

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
                input=prompt1
            )
            parsed = parse_conversation(response.output_text)
            print(f"Response: {response.output_text}")
            if parsed:
                write_jsonl(parsed, "implicit_persona.jsonl")
        except Exception as e:
            print(f"Error generating conversation {i+1} for topic '{topic}': {e}")
            time.sleep(1)