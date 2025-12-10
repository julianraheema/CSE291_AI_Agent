![Warning](./images/AI_Agent.png)


# ECUA — Edge Computer Unit Agent

This repository is a class project for **UCSD CSE291A Systems for LLM and AI Agents (Fall 2025)**.  
We developed an AI agent capable of running on edge devices.

## Hardware Requirements
- **GPU:** NVIDIA 3080 or similar  
- **RAM:** 16 GB or more

## OS Requirements
- **Operating System:** Ubuntu Linux 22.04


## Development Setup

To setup and run this repository for development, please follow the instructions bellow. 


# Using OSWorld Environment (Docker, Headless Mode)

> This option requires **~25GB** of extra storage (Docker + OSWorld assets + Llama3.2-3B).

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
ls CSE291_AI_Agent/ecua2_agent/planner_module/models/
```

If no model exist, please install using these instructions:


#### Step 1: Install Hugging Face Hub
```bash
pip install -U "huggingface_hub"
```

#### Step 2: Log in to Hugging Face
1. Create account: https://huggingface.co 
2. Request access: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
3. Create access token: https://huggingface.co/settings/tokens 
4. Login:
```bash
hf auth login
```

#### Step 3: Download Model
```bash
mkdir -p models
hf download meta-llama/Llama-3.2-3B-Instruct --local-dir ./CSE291_AI_Agent/ecua2_agent/planner_module/models/llama-3.2-3b
```

### 5. Run Agent Inside OSWorld
Move ecua_run.py and ecua_lib_run_single.py to OSWorld. Fee free to change the --max_steps to 100 for better results.

```bash
mv CSE291_AI_Agent/ecua_run.py .
mv CSE291_AI_Agent/ecua_lib_run_single.py .
mv CSE291_AI_Agent/ecua_score.py .
```
Run with CPU vision
```bash
python ecua_run.py   --provider_name docker   --headless   --action_space computer_13   --observation_type screenshot   --sleep_after_execution 3   --max_steps 30   --result_dir ./results/test_custom_tasks_and_agent   --tasks_path ./osworld_tasks
```
Run with GPU vision for faster and more accurate results
```bash
python ecua_run.py   --provider_name docker   --headless   --action_space computer_13   --observation_type screenshot   --sleep_after_execution 3   --max_steps 30   --result_dir ./results/osworld-human-os-tasks   --tasks_path ./osworld-human/os --v_gpu True
```

### Results
Saved under the `results/` directory.

### Evaluation
Runs this command to get the WES+/WES- evalution
```bash
python ecua_score.py --results-dir ./results/osworld-human-os-tasks/computer_13/screenshot/Llama-3.2-1B/custom/
```
Expected output:
```bash
 OSWorld-Human metrics summary
------------------------------------
Number of Tasks : 24
Success Rate    : 4.17%
Avg Result      : 0.0417
WES+ (mean)     : 0.0056 (0.56%)
WES- (mean)     : -0.2875 (-28.75%)
```

For custom tasks:
```bash
OSWorld-Human custom metrics summary
------------------------------------
Number of Tasks : 10
Success Rate    : 20.00%
Avg Result      : 0.2000
WES+ (mean)     : 0.0100 (1.00%)
WES- (mean)     : -0.0870 (-8.70%)
```

# Notes
If an error occurs during execution, I recommend running each module (vision, planner, control) as a standalone and verifying that it runs as expected. In each module, there is a README file that can help you set up and run it as a standalone. Most of the challenges we faced were related to dependencies and the model path. Therefore, we include our conda environment to avoid all the pain with Python dependencies.

# Future work:
## End-to-End Setup (In Progress)
This feature is still under active development. You may review it, but please be aware that running the end-to-end standalone version will execute commands directly on your machine.
Use extreme caution and run it only if you fully understand the risks. We cannot take responsibility for any unexpected behavior. This component is included only as future work and is not yet ready for testing.

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

# Our Custom osworld-human Tasks
You can find our custom tasks in the directory custom_class_human_tasks grouped into 4 groups: cli, LiberOffice_writer, multi_app, and spotify. 

# AI Agent Tasks

The project includes automation tasks defined in JSON format.

## 1. cli1MicrophoneDetection
Uses **arecord** and **sox** to create a live terminal spectrogram.

## 2. cli2CPUVisualizer
Live CPU visualization using:
- sar
- mpstat
- htop

## 3. cli3ImageManipulation
Downloads an image, processes it (resize, grayscale, blur), and displays via **jp2a**.

## 4. cliOpenTerminalAndType
This task opens Terminal and type Linux `top` command.

## 5. trafficScreenshotSD
Captures San Diego traffic snapshot from California QuickMap using Chrome.

## 6. weatherScreenshotSD
Captures NWS San Diego radar screenshot.

## 7. cloneTrendingRepo
Clones GitHub trending #1 repo and opens in VS Code.

## 8. leetcodeDailyQuestion
Opens LeetCode Daily Challenge and screenshots it.

## 9. spotifyPlaylist
Creates a date‑named Spotify playlist.

## 10. openOfficeAndWrite
Opens LiberOffice_writer app in linux and type A+

## Notes
- Agent and tasks are designed for Linux GNOME 
- Screenshots via `Alt+Print` 
- For educational & research purposes only


## Notice
This project is provided **as is** for educational and research purposes only. 
It may contain bugs, incomplete features, or unstable behavior.

The authors make **no guarantees or warranties** of any kind. 
Use of this software is at **your own risk**. 
The authors are not responsible for any damage, data loss, or issues resulting from its use.

By using this project, you agree to assume all risks and responsibilities.
