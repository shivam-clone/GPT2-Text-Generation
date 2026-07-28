from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Load Dataset
dataset = load_dataset(
    "text",
    data_files={"train": "data/train.txt"}
)

# Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# GPT-2 has no padding token
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Resize embeddings (good practice if tokenizer changes)
model.resize_token_embeddings(len(tokenizer))

# Tell model which token is padding
model.config.pad_token_id = tokenizer.pad_token_id

# -----------------------------
# Tokenize Dataset
# -----------------------------
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

#Training Arguments

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=5e-5,
    report_to="none"
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# -----------------------------
# Start Training
# -----------------------------
trainer.train()

# -----------------------------
# Save Model
# -----------------------------
trainer.save_model("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

print("\n✅ Training Complete!")
print("Model saved in: gpt2-finetuned/")