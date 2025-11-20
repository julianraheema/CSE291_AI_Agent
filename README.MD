# AI Agent Tasks

This project includes a set of automation tasks written in JSON format. These task are used for the AI agent research. Each task performs a specific action related to collecting, organizing, and saving visual data from online sources.

---

## 1. cli1_microphone_detection
This task turns your microphone input into a visual display inside the terminal.  
It uses the built in tool **arecord** and **sox** to live record, process, and generate spectrogram image.

---

## 2. cli2_CPU_visualizer

This task transforms your terminal into a live visualization of CPU activity.  
It uses installable tools such as **sar**, **mpstat**, and **htop** to enable system statistics, capture CPU load, and display them as numeric columns in the terminal.

---

## 3. cli3_image_manipulation

This CLI task is an image processing and manipulation pipeline.  
It pulls any image (in this example a black square from wiki) and resizes, grayscales and blurs the image. Then it blends the desaturated and blurred image together for a final product.  
It uses the **jp2a** tool to display the final product in your terminal.

---

## 4. traffic_screenshot_sd

This task opens Google Chrome and navigates to the **California QuickMap** website.  
It enables road condition overlays (Fast, Lane Closures, Full Closures, CHP Incidents), zooms into **San Diego**, switches to full-screen mode, and uses `Alt+Print` to capture and save a live traffic snapshot in the user’s Documents directory.

---

## 5. weather_screenshot_sd

This task opens **Google Chrome** and navigates to the **National Weather Service San Diego** page to view live weather and radar data. After ensuring the page is fully visible, it captures a window `Alt+Print` of the weather map and saves it automatically to the user’s Documents folder for later reference.

---

## 6. copy_to

This task opens a terminal and runs a command that creates a folder at  
`~/Documents/traffic_and_weather_data/` (if missing) and copies the **two most recent screenshots** into it.  
It ensures both the traffic and weather images are stored together in one organized directory for analysis or documentation.

---

## 7. clone_trending_repo

This task opens **Google Chrome**, navigates to GitHub's trending page, clones the #1 trending repository to the folder `~/Documents/trending_repos`, and then opens it in **VS Code**. 

---

## 8. leetcode_daily_question

This task automatically navigates to LeetCode's problem set page in **Google Chrome**, identifies and opens the daily challenge question, and captures a screenshot of the problem using **GNOME Screenshot**. The screenshots are stored in `~/Documents/leetcode_daily_challenges`. 

---

## 9. spotify_playlist

This task opens the **Spotify** desktop application and creates a new playlist named after the current date (e.g., "October 20 2025").

---

## 10. google_drive_folder

This task automatically opens Google Drive in **Google Chrome**, creates a new folder with a date based name, and navigates into the newly created folder.

---

### Notes
- All tasks are designed for **Linux systems** using GNOME desktop environment.  
- Screenshots are taken using `Alt+Print` screenshot commands and file management is handled via standard shell utilities.  
- These automations are intended for **educational and research use** only.

---
## Installation on Linux and Using Docker
sudo apt update && sudo apt -y upgrade
egrep -c '(vmx|svm)' /proc/cpuinfo
sudo apt -y install qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils virt-manager
sudo usermod -aG kvm,libvirt $USER

## Docker if your system didn't already has Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

## Python and Conda env
# Optional but recommended: miniconda
sudo apt -y install wget
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda create -y -n osworld python=3.10
conda activate osworld

## Install OSWorld
git clone https://github.com/xlang-ai/OSWorld
cd OSWorld
pip install -r requirements.txt

## Chose Model:

### Option1 Ollama:
#### Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
#### Example: pull a small model
ollama pull llama3.2:1b
#### Run the service (it autostarts on install, but you can ensure it’s up)
ollama serve

### Option2 VLLM inside osworld conda environment
pip install --upgrade pip
pip install vllm

python -c "import vllm; print('vLLM OK')"

## Running
## run vllm text only
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.9 \
  --port 8000

## run vllm text and image
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-VL-7B-Instruct-AWQ \
  --quantization awq \
  --dtype float16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --port 8000

export OPENAI_API_KEY="dummy"
export OPENAI_BASE_URL="http://localhost:8000/v1"

python run.py \
  --provider_name docker \
  --headless \
  --observation_type screenshot \
  --model llama3.2:8b \            
  --sleep_after_execution 3 \
  --max_steps 15 \
  --result_dir ./results/ecua-2_agent \

  python ecua-2_run.py   --provider_name docker   --headless   --action_space computer_13   --observation_type screenshot   --model gpt-qwen2vl-7b   --sleep_after_execution 3   --max_steps 30   --result_dir ./results/test_custom_tasks_qwen2vl_new   --tasks_path ./custom_tasks


# Assuming this structure:
# OSWorld/                       (git clone)
# ecua-2_agent/
#   agent/
#     __init__.py
#     agent.py
#     planner.py
#     controller.py
#     vision.py
#   scripts/
#     run_one_task.sh
#     run_10_tasks.sh
# tasks/
#   <task-folders>/<task.json>   (your task files)

# Make both repos importable:
export PYTHONPATH="$PWD/OSWorld:$PWD/ecua-2_agent:$PYTHONPATH"

## Show Results

