import torch
from models import BertMLPClassifier
from transformers import BertTokenizer
import config

# Load tokenizer and model

tokenizer = BertTokenizer.from_pretrained(config.pretrained_bert)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = BertMLPClassifier(pretrained_bert=config.pretrained_bert, dropout_rate=0.3).to(device)
mlp_state = torch.load("mlp_head_only.pt", map_location=device)
# Try to load into the MLP head part only
model.mlp_head.load_state_dict(mlp_state)
model.eval()

# ==== Test Set with Ground Truth ====

examples = [
    # ✅ True preference (label = 1)
    ("Hi", "I don't really like teaching yoga", 1),
    ("That's a great approach to staying alert...",
     "I usually prefer interactive or gamified learning methods over passive ones...", 1),
    ("Can you recommend some resources to help me understand economics?",
     "I strongly prefer learning through storytelling...", 1),
    ("Do you enjoy reading?", "Yeah, I love fantasy novels — Tolkien’s stuff especially.", 1),
    ("What’s your favorite way to relax?", "I always listen to ambient music before bed.", 1),
    ("What kind of vacations do you prefer?", "I prefer road trips to flying.", 1),
    ("Do you enjoy cooking?", "No, I actually hate cooking.", 1),
    ("What's your ideal morning routine?", "Always start with coffee and a walk.", 1),
    ("Do you like board games?", "I especially enjoy strategy games like Catan or Risk.", 1),
    ("What's your opinion on pineapple on pizza?", "Hate it. It ruins everything.", 1),
    ("Would you ever go skydiving?", "no", 1),
    ("Why do you enjoy hiking?", "Not really sure, I just do.", 1),
    ("What would you like to eat?", "Mexican food is my go-to comfort cuisine.", 1),
    ("Do you usually work out in the morning or evening?", "Evenings work best for me — I hate early mornings.", 1),
    ("How do you like your coffee?", "Black with a splash of oat milk.", 1),
    ("What's your favorite genre of movies?", "I’m really into psychological thrillers.", 1),
    ("Do you enjoy group projects?", "Not really. I prefer working solo when I can.", 1),
    ("How do you relax on weekends?", "I unwind by watching indie films or reading poetry.", 1),
    ("Do you like public speaking?", "I actually enjoy it—it’s a bit of a rush.", 1),
    ("Any preferred method for studying?", "Flashcards help me more than long notes.", 1),
    ("What's your preferred climate?", "I love colder weather — hate the heat.", 1),
    ("Do you enjoy spicy food?", "Yes, the spicier the better!", 1),
    ("What kind of pets do you like?", "Big dogs — labs, retrievers, that kind of thing.", 1),
    ("How do you like your steak cooked?", "Medium rare. Anything else is wrong.", 1),
    ("Would you rather travel or stay home?", "Travel. I get restless staying in one place.", 1),
    ("Do you listen to music while working?", "Always — lo-fi beats help me focus.", 1),
    ("How do you feel about horror movies?", "Can’t stand them. Too stressful.", 1),
    ("What's your ideal vacation?", "Exploring old European cities, especially Italy.", 1),
    ("Do you play video games?", "Yeah, I’m a big fan of RPGs and open-world games.", 1),
    ("How do you usually commute?", "I bike everywhere. It’s faster and healthier.", 1),
    ("Do you journal?", "Every night before bed. It helps me decompress.", 1),
    ("What shoes do you like?", "High-top Converse — always have, always will.", 1),
    ("Do you prefer texting or calling?", "Texting, hands down. I hate phone calls.", 1),
    ("What kind of workouts do you like?", "I like HIIT workouts. Quick and intense.", 1),
    ("What do you think of sushi?", "Love it — especially spicy tuna rolls.", 1),
    ("Would you want to live in a city?", "Absolutely. I thrive in busy environments.", 1),
    ("Do you enjoy driving?", "Yeah, especially long highway drives with music.", 1),
    ("", "I like japanese food", 1),
    ("", "I like to go to the gym in the morning", 1),


    # ❌ No preference (label = 0)
    ("Hey, how was your day?", "It was fine, just the usual.", 0),
    ("Do you want to meet up tomorrow or later this week?", "Either works for me.", 0),
    ("Do you care where we eat tonight?", "Nah, whatever you’re in the mood for.", 0),
    ("What do you think about classical music?", "I don’t really have an opinion on it.", 0),
    ("Do you want to take the train or drive?", "I’m fine with whatever’s easiest.", 0),
    ("Any thoughts on this idea?", "I’m not sure yet, still thinking it over.", 0),
    ("Want to go hiking or just chill at home?", "I’m okay with either.", 0),
    ("Which shirt looks better?", "They both look good to me.", 0),
    ("Are you excited for the movie?", "It should be fine.", 0),
    ("How’s your day going?", "Pretty normal, nothing special.", 0),
    ("What hobbies are you into?", "Not sure yet. Still exploring.", 0),
    ("Do you have any thoughts on this proposal?", "Let me think about it and get back to you.", 0),
    ("Do you like chocolate?", "I’ve never really thought about it.", 0),
    ("Hi!", "Hey there!", 0),
    ("Hi", "How are you doing", 0),
    ("Hey, long time no see! How’ve you been?", "I’ve been alright, just busy with work.", 0),
    ("Did you catch the game last night?", "No, I missed it. Was it good?", 0),
    ("Morning! Ready for the big meeting?", "As ready as I’ll ever be.", 0),
    ("That presentation went well, don’t you think?", "Yeah, seemed like people were paying attention.", 0),
    ("So, any fun weekend plans?", "Not yet, might just relax at home.", 0),
    ("Did you hear back from the recruiter?", "Yeah, just a short update, nothing major.", 0),
    ("How’s your project going?", "Making progress, still a few things to fix.", 0),
    ("Can you believe how hot it is today?", "Right? I didn’t even want to leave the house.", 0),
    ("You look tired. Long night?", "Sort of, couldn’t sleep well.", 0),
    ("Are you joining the call later?", "I think so, unless something comes up.", 0),
    ("Wanna grab a coffee before class?", "Sure, I’ve got some time.", 0),
    ("What time are we meeting again?", "I think we said 3 PM?", 0),
    ("Did you see that meme I sent?", "Haha yeah, cracked me up.", 0),
    ("Want me to forward the notes to you?", "Yes please, that would help a lot.", 0),
    ("Where did you park?", "In the garage under the library.", 0),
    ("Hey, do you know what chapter we're on?", "I think it’s Chapter 7.", 0),
    ("You watched the finale?", "Yeah, finally caught up!", 0),
    ("I heard they’re remodeling the office.", "Oh yeah? I hadn’t heard that yet.", 0),
    ("You okay with working from the library today?", "Yeah, that works.", 0),
    ("When’s your flight?", "Tomorrow morning. Super early.", 0),
    ("How was the dentist?", "Not bad, just a cleaning.", 0),
    ("Can you resend that file?", "Sure, give me a second.", 0),
    ("What’s the weather like today?", "Cloudy, looks like rain.", 0),
    ("Think it’ll be busy at the gym?", "Hard to say, maybe less crowded now.", 0),
    ("You going to the team lunch?", "Probably, depends on how work goes.", 0),
    ("whats good bro", "not much", 0),
    ("Do you like liverpool?", "helll nahhh", 1),
    ("", "i like pizza", 1),

]

# ==== Evaluation ====

threshold = 0.5
correct = 0

for i, (agent, user, label) in enumerate(examples):
    inputs = tokenizer(agent, user, return_tensors="pt", truncation=True, padding="max_length", max_length=150)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        logit = model(**inputs)
        prob = torch.sigmoid(logit).item()
        pred = 1 if prob >= threshold else 0

    match = "✅" if pred == label else "❌"
    print(f"\n🧠 Example {i+1} {match}")
    print(f"Agent: {agent}")
    print(f"User:  {user}")
    print(f"Label: {label} | Predicted: {pred} | Score: {prob:.4f}")

    if pred == label:
        correct += 1

accuracy = correct / len(examples)
print(f"\n🎯 Overall Accuracy: {accuracy * 100:.2f}% ({correct}/{len(examples)})")