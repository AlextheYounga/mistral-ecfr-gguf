import torch
from datasets import load_dataset
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	TrainingArguments,
	Trainer,
	DataCollatorForLanguageModeling
)
from peft import (
	LoraConfig,
	get_peft_model,
	prepare_model_for_kbit_training,
	TaskType
)

def main():
	# Configuration parameters (edit these as needed)
	config = {
		"model_name": "mistralai/Mistral-7B-Instruct-v0.3",
		"dataset_path": "./data/ecfr.json.zip",
		"dataset_split": "train",
		"text_column": "text",
		"output_dir": "./mistral-7b-lora-output",
		
		# LoRA parameters
		"lora_r": 8,
		"lora_alpha": 16,
		"lora_dropout": 0.05,
		
		# Training parameters
		"learning_rate": 3e-4,
		"batch_size": 4,
		"gradient_accumulation_steps": 4,
		"max_steps": 1000,
		"max_seq_length": 512,
		"save_steps": 100,
		"logging_steps": 10,
		
		# Use 8-bit quantization to reduce memory requirements
		"use_8bit": True,
	}
	
	print("Starting Mistral-7B LoRA fine-tuning process...")
	
	# Prepare tokenizer
	print(f"Loading tokenizer for {config['model_name']}...")
	tokenizer = AutoTokenizer.from_pretrained(config['model_name'], use_fast=True)
	tokenizer.pad_token = tokenizer.eos_token
	
	# Load and prepare model
	print("Loading model...")
	if config["use_8bit"]:
		print("Using 8-bit quantization to reduce memory usage")
		model = AutoModelForCausalLM.from_pretrained(
			config['model_name'],
			load_in_8bit=True,
			torch_dtype=torch.float16,
			device_map="auto"
		)
		model = prepare_model_for_kbit_training(model)
	else:
		model = AutoModelForCausalLM.from_pretrained(
			config['model_name'],
			torch_dtype=torch.float16,
			device_map="auto"
		)
	
	# Configure LoRA
	print("Applying LoRA configuration...")
	lora_config = LoraConfig(
		r=config['lora_r'],
		lora_alpha=config['lora_alpha'],
		lora_dropout=config['lora_dropout'],
		bias="none",
		task_type=TaskType.CAUSAL_LM,
		# Target attention layers and MLP layers
		target_modules=[
			"q_proj",
			"k_proj",
			"v_proj",
			"o_proj",
			"gate_proj",
			"up_proj",
			"down_proj"
		]
	)
	
	# Apply LoRA to model
	model = get_peft_model(model, lora_config)
	model.print_trainable_parameters()  # Print info about trainable params
	
	# Load and prepare dataset
	dataset = load_dataset("json", data_files=config['dataset_path'])["train"]
	
	# Print sample data
	print("\nSample data from dataset:")
	for i in range(min(2, len(dataset))):
		print(f"Example {i}: {dataset[i][config['text_column']][:100]}...\n")
	
	# Tokenize dataset
	print("Tokenizing dataset...")
	def tokenize_function(examples):
		return tokenizer(
			examples[config['text_column']],
			padding="max_length",
			truncation=True,
			max_length=config['max_seq_length']
		)
	
	tokenized_dataset = dataset.map(
		tokenize_function,
		batched=True,
		remove_columns=[col for col in dataset.column_names if col != config['text_column']]
	)
	
	# Data collator
	data_collator = DataCollatorForLanguageModeling(
		tokenizer=tokenizer,
		mlm=False  # Not using masked language modeling
	)
	
	# Set up training arguments
	print("Setting up training configuration...")
	training_args = TrainingArguments(
		output_dir=config['output_dir'],
		learning_rate=config['learning_rate'],
		per_device_train_batch_size=config['batch_size'],
		gradient_accumulation_steps=config['gradient_accumulation_steps'],
		max_steps=config['max_steps'],
		logging_steps=config['logging_steps'],
		save_steps=config['save_steps'],
		save_total_limit=3,
		fp16=True,
		remove_unused_columns=False,
		push_to_hub=False,
	)
	
	# Initialize Trainer
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_dataset,
		data_collator=data_collator,
	)
	
	# Train the model
	print("Starting training...")
	trainer.train()
	
	# Save the final model
	print(f"Saving final model to {config['output_dir']}")
	trainer.save_model(config['output_dir'])
	
	print("Training complete!")
	print(f"Model and adapter weights saved to {config['output_dir']}")
	print("\nTo use this model:")
	print(f"1. Load the base model: model = AutoModelForCausalLM.from_pretrained('{config['model_name']}')")
	print(f"2. Load the LoRA adapter: model = PeftModel.from_pretrained(model, '{config['output_dir']}')")

if __name__ == "__main__":
	main()