import torch, pprint
sd = torch.load("trained_lstm.pt", map_location="cpu")
print("Top‑level keys:", sd.keys())