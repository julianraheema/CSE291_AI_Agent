![Warning](./images/AI_Agent.png)


# ECUA — Edge Computer Unit Agent

This repository is a class project for **UCSD CSE (Fall 2025)**.  
We developed an AI agent capable of running on edge devices.

## Hardware Requirements
- **GPU:** NVIDIA 3080 or similar  
- **RAM:** 16 GB or more

## Software Requirements
- **Operating System:** Ubuntu Linux 22.04

## Setup

There are two ways to set up this repository for testing.

# Option 1 — End-to-End Setup (Local or VM)
> **Recommended:** Use a designated VM for OSWorld agent evaluation.[OSWorld](https://github.com/xlang-ai/OSWorld)
> **Not recommended:** Running directly on your main machine.

### 1. Clone Repository
```bash
git clone https://github.com/julianraheema/CSE291_AI_Agent.git
cd CSE291_AI_Agent
```

### 2. Install Dependencies

#### Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

#### Set up virtual environment
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements-uv.txt
```

## Installing LLM Models

Place your chosen LLM model inside:

```
planner_module/models/
```


## How to Download a Llama Model

### Step 1: Install Hugging Face Hub
```bash
uv pip install -U "huggingface_hub"
```

### Step 2: Log in to Hugging Face
1. Create account: https://huggingface.co 
2. Request access: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
3. Create access token: https://huggingface.co/settings/tokens 
4. Login:
```bash
hf auth login
```

### Step 3: Download Model
```bash
mkdir -p models
hf download meta-llama/Llama-3.2-3B-Instruct --local-dir ./models/llama-3.2-3b
```

## Running the Agent
```bash
cd CSE291_AI_Agent
python run_agent.py --path osworld_tasks/ --record
```

### Results
Outputs (screenshots, `.mp4` recordings) are saved under:

```
results/
```

# Option 2 — Using OSWorld Environment (Docker, Headless Mode)

> This option requires **~15GB** of extra space (Docker + OSWorld assets).

### 1. Install Prerequisites
- Install **Docker**
- Install **Conda**

### 2. Clone OSWorld and Agent
```bash
git clone https://github.com/xlang-ai/OSWorld
cd OSWorld
git clone https://github.com/julianraheema/CSE291_AI_Agent.git
```

### 3. Install OSWorld Environment
```bash
conda env create -f CSE291_AI_Agent/environment.yml
conda activate osworld
```

### 4. Model Placement

Ensure LLM models are located at:

```bash
ls OSWorld/CSE291_AI_Agent/ecua2_agent/planner_module/models/
```

### 5. Run Agent Inside OSWorld
Move ecua_run.py and ecua_lib_run_single.py to OSWorld. Fee free to change the --max_steps to 100 for better results.

```bash
mv CSE291_AI_Agent/ecua_run.py .
mv CSE291_AI_Agent/ecua_lib_run_single.py .
mv CSE291_AI_Agent/ecua_score.py .
```
Run with CPU
```bash
python ecua_run.py   --provider_name docker   --headless   --action_space computer_13   --observation_type screenshot   --sleep_after_execution 3   --max_steps 30   --result_dir ./results/test_custom_tasks_and_agent   --tasks_path ./osworld_tasks
```
Run with GPU
```bash
python ecua_run.py   --provider_name docker   --headless   --action_space computer_13   --observation_type screenshot   --sleep_after_execution 3   --max_steps 30   --result_dir ./results/osworld-human-os-tasks   --tasks_path ./osworld-human/os --v_gpu True
```

Results:
```bash
 OSWorld-Human metrics summary
------------------------------------
Number of Tasks : 24
Success Rate    : 4.17%
Avg Result      : 0.0417
WES+ (mean)     : 0.0056 (0.56%)
WES- (mean)     : -0.2875 (-28.75%)
```

### Results
Saved under the `results/` directory.

### Evaluation
Runs this command to get the WES+/WES- evalution
```bash
python ecua_score.py --results-dir ./results/osworld-human-os-tasks/computer_13/screenshot/Llama-3.2-1B/custom/
```

# Custom Tasks

These tasks require conversion into OSWorld format. 
Currently, they are **not** OSWorld-compatible, so we use OSWorld tasks for evaluation.

# AI Agent Tasks

The project includes automation tasks defined in JSON format.

## 1. cli1_microphone_detection
Uses **arecord** and **sox** to create a live terminal spectrogram.

## 2. cli2_CPU_visualizer
Live CPU visualization using:
- sar
- mpstat
- htop

## 3. cli3_image_manipulation
Downloads an image, processes it (resize, grayscale, blur), and displays via **jp2a**.

## 4. traffic_screenshot_sd
Captures San Diego traffic snapshot from California QuickMap using Chrome.

## 5. weather_screenshot_sd
Captures NWS San Diego radar screenshot.

## 6. copy_to
Copies latest traffic + weather screenshots into:
```
~/Documents/traffic_and_weather_data/
```

## 7. clone_trending_repo
Clones GitHub trending #1 repo and opens in VS Code.

## 8. leetcode_daily_question
Opens LeetCode Daily Challenge and screenshots it.

## 9. spotify_playlist
Creates a date‑named Spotify playlist.

## 10. google_drive_folder
Creates a date‑named folder in Google Drive.

## Notes
- Designed for Linux GNOME 
- Screenshots via `Alt+Print` 
- For educational & research purposes only


## Notice

This project is provided **as is** for educational and research purposes only. 
It may contain bugs, incomplete features, or unstable behavior.


The authors make **no guarantees or warranties** of any kind. 
Use of this software is at **your own risk**. 
The authors are not responsible for any damage, data loss, or issues resulting from its use.

By using this project, you agree to assume all risks and responsibilities.
