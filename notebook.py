from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="gpt2"
)

result = generator(
    "Artificial Intelligence is",
     max_new_tokens=80,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1,
    truncation=True
)

print(result[0]["generated_text"])