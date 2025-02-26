# train_lightweight_lora.py
import os
import torch
from datasets import load_dataset
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	TrainingArguments,
	Trainer,
	DataCollatorForLanguageModeling,
	EarlyStoppingCallback
)
from peft import (
	LoraConfig,
	get_peft_model,
	prepare_model_for_kbit_training,
	TaskType
)
import bitsandbytes as bnb
from tqdm import tqdm

def main():
	# Configuration parameters (edit these as needed)
	config = {
		"model_name": "mistralai/Mistral-7B-Instruct-v0.3",
		"dataset_path": "./data/ecfr.json.zip",
		"dataset_split": "train",
		"text_column": "text",
		"output_dir": "./mistral-7b-lightweight-lora",
		
		# Lightweight LoRA parameters - reduced for conceptual learning without style imitation
		"lora_r": 4,             # Reduced rank (from 8)
		"lora_alpha": 8,         # Reduced alpha (from 16)
		"lora_dropout": 0.1,     # Slightly higher dropout to prevent overfitting
		
		# Training parameters - gentler training approach
		"learning_rate": 5e-5,   # Lower learning rate (from 3e-4)
		"batch_size": 4,
		"gradient_accumulation_steps": 4,
		"warmup_steps": 10,     # Added warmup steps
		"max_steps": 60,        # Fewer steps for lightweight tuning
		"max_seq_length": 512,
		"save_steps": 10,
		"logging_steps": 10,
		
		# Early stopping parameters
		"early_stopping_patience": 3,  # Stop if no improvement after 3 evaluations
		
		# Use 8-bit quantization to reduce memory requirements
		"use_8bit": True,
		
		# Limited target modules for more focused learning
		"target_modules": [
			"q_proj",            # Query projection
			"k_proj",            # Key projection
			"v_proj",            # Value projection
			# Note: Removed o_proj, gate_proj, up_proj, down_proj to make tuning lighter
		]
	}
	
	print("Starting Mistral-7B Lightweight LoRA fine-tuning process...")
	print("This configuration is designed for conceptual understanding without style mimicry")
	
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
	
	# Configure LoRA with lightweight settings
	print("Applying lightweight LoRA configuration...")
	lora_config = LoraConfig(
		r=config['lora_r'],
		lora_alpha=config['lora_alpha'],
		lora_dropout=config['lora_dropout'],
		bias="none",
		task_type=TaskType.CAUSAL_LM,
		# Target only specific attention layers for more conceptual learning
		target_modules=config['target_modules']
	)
	
	# Apply LoRA to model
	model = get_peft_model(model, lora_config)
	model.print_trainable_parameters()  # Print info about trainable params
	
	# Load and prepare dataset
	dataset = load_dataset("json", data_files=config['dataset_path'])["train"]
	
	# Function to format data with instructional framing to prevent style mimicry
	def format_instruction(example):
		# Replace this template with one appropriate for your dataset
		# This framing helps the model learn content while maintaining its own voice
		text = f"""
Below is some information. Learn from it but use your own style when answering questions.

Content: {example[config['text_column']]}

Remember this information but keep your original voice when responding.
"""
		return {"formatted_text": text}
	
	# Apply the formatting
	print("Formatting dataset with instructional framing...")
	formatted_dataset = dataset.map(format_instruction)
	
	# Print sample data
	print("\nSample formatted data:")
	for i in range(min(1, len(formatted_dataset))):
		print(f"Example {i}: {formatted_dataset[i]['formatted_text'][:200]}...\n")
	
	# Tokenize dataset
	print("Tokenizing dataset...")
	def tokenize_function(examples):
		return tokenizer(
			examples['formatted_text'],
			padding="max_length",
			truncation=True,
			max_length=config['max_seq_length']
		)
	
	tokenized_dataset = formatted_dataset.map(
		tokenize_function,
		batched=True,
		remove_columns=formatted_dataset.column_names  # Remove ALL original columns including text
	)
	
	# Calculate training size for validation split
	train_size = int(0.9 * len(tokenized_dataset))
	eval_size = len(tokenized_dataset) - train_size
	
	# Create train/eval split for early stopping
	train_eval_split = tokenized_dataset.train_test_split(
		train_size=train_size,
		test_size=eval_size,
		shuffle=True,
		seed=42
	)
	
	# Data collator
	data_collator = DataCollatorForLanguageModeling(
		tokenizer=tokenizer,
		mlm=False  # Not using masked language modeling
	)
	
	# Set up training arguments with evaluation
	print("Setting up training configuration...")
	training_args = TrainingArguments(
		output_dir=config['output_dir'],
		learning_rate=config['learning_rate'],
		per_device_train_batch_size=config['batch_size'],
		per_device_eval_batch_size=config['batch_size'],
		gradient_accumulation_steps=config['gradient_accumulation_steps'],
		max_steps=config['max_steps'],
		warmup_steps=config['warmup_steps'],
		logging_steps=config['logging_steps'],
		save_steps=config['save_steps'],
		eval_strategy="steps",  # Changed from evaluation_strategy to avoid deprecation warning
		eval_steps=config['save_steps'],
		save_total_limit=3,
		fp16=True,
		remove_unused_columns=False,
		push_to_hub=False,
		load_best_model_at_end=True,
		metric_for_best_model="eval_loss",
		greater_is_better=False,
		# Weight decay for regularization to prevent overfitting
		weight_decay=0.01,
	)
	
	# Use the built-in EarlyStoppingCallback from transformers
	callbacks = [EarlyStoppingCallback(early_stopping_patience=config['early_stopping_patience'])]
	
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_eval_split["train"],
		eval_dataset=train_eval_split["test"],
		data_collator=data_collator,
		callbacks=callbacks,
	)
	
	# Train the model
	print("Starting lightweight fine-tuning...")
	trainer.train()
	
	# Save the final model
	print(f"Saving final model to {config['output_dir']}")
	trainer.save_model(config['output_dir'])
	
	print("\n============================")
	print("Lightweight fine-tuning complete!")
	print(f"Model and adapter weights saved to {config['output_dir']}")
	print("\nNext Steps:")
	print("1. Use this fine-tuned model as part of a retrieval system")
	print("2. For inference, combine this model with a vector database of your texts")
	print("3. This approach should give you content understanding without style mimicry")
	print("============================")

if __name__ == "__main__":
	main()