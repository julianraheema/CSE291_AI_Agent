# Planner Setup Guide

This guide will help you set up the environment and download the required model for this project.


## Step 1: Install UV (Python Package Manager)

First, install `uv`, a fast Python package manager:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Restart your shell or source the config:
```bash
source $HOME/.cargo/env
```

## Step 2: Create Virtual Environment

Create and activate a Python virtual environment:
```bash
# Create environment
uv venv --python 3.12 --seed

# Activate environment
source .venv/bin/activate
```

## Step 3: Install vLLM

Install vLLM with automatic PyTorch backend detection:
```bash
uv pip install vllm --torch-backend=auto
```

This will automatically detect your CUDA version and install the appropriate PyTorch build.

## Step 4: Put Your Model in planner_module/models

Put whichever LLM model you are using in the directory planner_module/models. 
Change the code in planner_module/planner.py and set the model_path to your model's path. 

# How to Download a Llama Model


## Step 1: Install Hugging Face Hub

Install the Hub library (includes the `hf` CLI):
```bash
uv pip install -U "huggingface_hub"
```

## Step 2: Login to Hugging Face (Required for Llama models)

Llama models require authentication. Create a Hugging Face account and get access:

1. Create account at https://huggingface.co/
2. Request access to Llama 3.2 at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
3. Create an access token at: https://huggingface.co/settings/tokens
4. Login via CLI:
```bash
hf auth login
```

Enter your access token when prompted.

## Step 3: Download the Model

Download the Llama 3.2 1B Instruct model locally:
```bash
# Create models directory
mkdir -p models

# Download model
hf download meta-llama/Llama-3.2-1B-Instruct --local-dir ./models/llama-3.2-1b
```