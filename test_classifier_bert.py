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
    agent="Hi",
    user="I don't really like teaching yoga"
)

predict_preference(
    agent="That's a great approach to staying alert! For quick, engaging activities, you could try mental math exercises, memory games, or even crime scenario puzzles. Many police departments have mobile apps with training modules or quizzes that you can access on your phone. These can help reinforce your knowledge of procedures and laws while keeping you engaged. Have you noticed any particular challenges during your night patrols so far?",
    user="I've been on night shifts for about a month now. It's still an adjustment. Speaking of mental exercises, I usually prefer interactive or gamified learning methods over passive ones like podcasts. Any ideas for quick, engaging activities I can do during slow periods to stay sharp?"
)

predict_preference(
    agent="Can you recommend some resources to help me understand the principles of economics?",
    user="I strongly prefer learning through storytelling and narrative-based methods and dislike dry, factual textbooks."
)

# Yes
predict_preference(
    agent="Do you enjoy reading?",
    user="Yeah, I love fantasy novels — Tolkien’s stuff especially."
)

predict_preference(
    agent="What’s your favorite way to relax?",
    user="I always listen to ambient music before bed."
)

predict_preference(
    agent="What kind of vacations do you prefer?",
    user="I prefer road trips to flying. I like the freedom."
)

predict_preference(
    agent="Do you enjoy cooking?",
    user="No, I actually hate cooking. Too stressful for me."
)

predict_preference(
    agent="What's your ideal morning routine?",
    user="Always start with coffee and a walk. I can't function without it."
)

predict_preference(
    agent="Do you like board games?",
    user="Yes. I especially enjoy strategy games like Catan or Risk."
)

predict_preference(
    agent="What's your opinion on pineapple on pizza?",
    user="Hate it. It ruins everything."
)

# No

predict_preference(
    agent="Hey, how was your day?",
    user="It was fine, just the usual."
)

predict_preference(
    agent="What hobbies are you into?",
    user="Not sure yet. Still exploring."
)

predict_preference(
    agent="Do you have any thoughts on this proposal?",
    user="Let me think about it and get back to you."
)

predict_preference(
    agent="What kind of art do you like?",
    user="I guess I’m open to anything, really."
)

predict_preference(
    agent="Do you like chocolate?",
    user="I’ve never really thought about it."
)

predict_preference(
    agent="Hi!",
    user="Hey there!"
)

predict_preference(
    agent="Why do you enjoy hiking?",
    user="I don’t know."
)