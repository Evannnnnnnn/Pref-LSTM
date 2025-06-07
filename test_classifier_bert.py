import torch
from models import BertMLPClassifier
from transformers import BertTokenizer

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = BertMLPClassifier().to(device)
model.load_state_dict(torch.load("bert_mlp_preference.pt", map_location=device))
model.eval()

def predict_preference(agent, user, threshold=0.5):
    inputs = tokenizer(
        agent,
        user,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=150
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        inputs.pop("token_type_ids", None)
        logit = model(**inputs)
        prob = torch.sigmoid(logit).item()

    print(f"\n🧠 Vibe Test:")
    print(f"Agent: {agent}")
    print(f"User:  {user}")
    print(f"Preference score: {prob:.4f} → {'✅ Preference' if prob >= threshold else '❌ No Preference'}")

predict_preference(
    agent="Hello! It's great that you're looking to explore new hobbies in retirement. There are many options suitable for retirees. Have you considered activities like gardening, woodworking, or joining a local book club? These can be enjoyable and social ways to spend your time. What kind of activities did you enjoy before retirement?",
    user="Hello there! I've been thinking about taking up a new hobby now that I'm retired from the postal service. Any suggestions for an old-timer like me?"
)

predict_preference(
    agent="Hi",
    user="How are you doing"
)

predict_preference(
    agent="Can you recommend some resources to help me understand the principles of economics?",
    user="I strongly prefer learning through storytelling and narrative-based methods and dislike dry, factual textbooks."
)
