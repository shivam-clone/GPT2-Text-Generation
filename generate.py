from transformers import AutoTokenizer, AutoModelForCausalLM

# Load fine-tuned model
model_path = "./gpt2-finetuned"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

prompt = input("Enter Prompt: ")

inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("\nGenerated Text:\n")
print(generated_text)