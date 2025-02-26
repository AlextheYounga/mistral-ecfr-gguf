import torch
from datasets import load_dataset
import zipfile
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

def unzip_file(zip_path):
	# Unzip the file into a json file
	with zipfile.ZipFile(zip_path, "r") as z:
		with z.open(z.namelist()[0]) as f:
			data = f.read()
			# Write the data to a json file
			output_path = zip_path.replace(".zip", "")
			with open(output_path, "wb") as j:
				j.write(data)
			return output_path

# This lets Tensor Cores use TF32 precision, speeding up matrix multiplications while reducing memory use.
# torch.backends.cuda.matmul.allow_tf32 = True

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# 1. Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Some LLaMA-based models don't have a pad token - safe fallback:
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,    # Mixed precision (FP16)
)

# Enable gradient checkpointing for memory savings
model.gradient_checkpointing_enable()

# Load dataset (assuming it's a JSON file)
data_file = unzip_file("./data/ecfr.json.zip")
dataset = load_dataset("json", data_files=data_file)

# First, split into 90% train, 10% temp
split = dataset["train"].train_test_split(test_size=0.10)

# Split temp set into 5% validation, 5% test
test_valid_split = split["test"].train_test_split(test_size=0.5)

# Assign final datasets
train_dataset = split["train"]
valid_dataset = test_valid_split["train"]
test_dataset = test_valid_split["test"]

# 3. Tokenize the dataset and add labels
max_length = 768  # Adjust according to GPU memory
def tokenize_function(examples):
    encoding = tokenizer(
        examples["text"], truncation=True, max_length=max_length
    )
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset = valid_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Convert dataset columns to torch.Tensor
train_dataset.set_format("torch")
eval_dataset.set_format("torch")
test_dataset.set_format("torch")

# 4. Define training arguments
training_args = TrainingArguments(
    output_dir="./ecfr-tuned-mistral",
    overwrite_output_dir=True,
    num_train_epochs=3,                    # Increase for real training
    per_device_train_batch_size=1,         # Small batch for memory savings
    gradient_accumulation_steps=8,         # Accumulate gradients to simulate bigger batch
    evaluation_strategy="steps",           # or "epoch"
    eval_steps=100,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    bf16=True,                             # Use FP16
    gradient_checkpointing=True,           # Saves memory
    optim="adamw_torch",                   # Standard optimizer
    report_to="none",                      # Disable wandb or similar if you prefer
)

# 5. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, 
)

# 6. Start training
trainer.train()

# 7. Evaluate the model on the test set after training
test_results = trainer.evaluate(test_dataset)
print("Test results:", test_results)
