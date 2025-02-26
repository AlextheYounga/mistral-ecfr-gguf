# Training Mistral Model on the eCFR

## Workspace for training a Mistral 7b model on the entire Code of Federal Regulations
This was an experiment for getting a simple Mistral model trained on the eCFR. 

There may be some promise here, but I am moving on to other projects for the time being. I spent one week on this, I am sure that more time on this could eventually turn into some fruitful. 

There is also certainly good reason to try this, as not even ChatGPT is trained on the eCFR. ChatGPT can only list the titles from memory and not much else. It can obviously search the internet to find the sections that you request, but it doesn't know enough specifics about the code to find things that others might have missed. 

The ideal outcome was to be able to create a small gguf model that can run on the CPU that was "good-enough" to use alongside a RAG system, or use as a quick reference for combing through the monster of the Code of Federal Regulations.

At the very least, I have created a great foundation for further research here. I have also included some helpful Runpod scripts for quick training on their remote GPUs. 

You can find the full eCFR training data [here](https://drive.proton.me/urls/VCV61T2R9M#V27jHni5kDmI)

Here are my training attempts and their outcomes:
- Full Fine Tune = catastrophic overfitting, massive forgetfulness
- Light Lora Training:
  - 60 steps = Incorrect information and generally ignorant of the code
  - 100 epochs = Closer, and did start reciting other sections from memory, but not quite the section asked for.
  - 200 epochs = Unstable, catastraphic forgetfulness, incoherent outputs.

See [Notes](./notes.md) for full output

The 100 epochs model, using the `train_light_lora.py` showed some promise, but more work is needed. 
I am making this 100 epochs model available [here](https://huggingface.co/alextheyounger/Mistral7b-eCFR)


### Getting Started
```sh
cd workspace

pip install -r requirements.txt

# Download the LLama.CPP library for creating and communicating with GGUF models
# This will also run the build scripts for llama
./scripts/download_llama_cpp
```

## Chatting with Model
1. Place model in a folder called models
2. Ensure that you have run the `download_llama_cpp` script from the last step
3. Run `./scripts/chat mistral-lora-cfr-checkpoint-100.gguf`


## Creating GGUFs from Models
If you have a model you've trained, I have left an example on how to convert your model into a small GGUF model
Check the `./scripts/make_gguf` file. 