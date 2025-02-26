import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path = "../models/mistral-7b-lightweight-lora/checkpoint-100"
output_merged_model = "../models/mistral-7b-merged"

# Load base model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map={"": "cpu"}  # load directly onto CPU
)

print("Loading LORA adapter...")
model = PeftModel.from_pretrained(model, adapter_path)

# Merge weights (this creates a standalone fine-tuned model)
print("Merging weights...")
model = model.merge_and_unload()

# Save the merged model
print("Saving merged model...")
model.save_pretrained(output_merged_model, safe_serialization=True)

# Save tokenizer (optional but recommended)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(output_merged_model)
