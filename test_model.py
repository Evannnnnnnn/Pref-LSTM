from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,    # Use float16 for GPU efficiency
    device_map="auto"             # Automatically place on available GPU(s)
)

# Prepare input prompt
prompt = "Write a short poem about the moon:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate text
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

# Decode and print output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
