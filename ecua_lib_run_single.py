import datetime
import json
import logging
import os
import re
import cv2
import time
from wrapt_timeout_decorator import *
from lib_results_logger import log_task_completion
import subprocess, json, base64      

os.environ["VLLM_PLUGINS"] = "none"
from CSE291_AI_Agent.ecua2_agent.planner_module import planner

logger = logging.getLogger("desktopenv.experiment")

import tempfile
import numpy as np


ALLOWED_COMMANDS = {
    "MOVE_TO",
    "CLICK",
    "MOUSE_DOWN",
    "MOUSE_UP",
    "RIGHT_CLICK",
    "DOUBLE_CLICK",
    "DRAG_TO",
    "SCROLL",
    "TYPING",
    "PRESS",
    "KEY_DOWN",
    "KEY_UP",
    "HOTKEY",
    "WAIT",
    "FAIL",
    "DONE",
}

def save_actions_to_file(actions, domain, example_id):
    with open(f'{domain}_{example_id}.txt', "w") as f:
        for action in actions:
            f.write(action + "\n")

def normalize_actions(action_list, domain, example_id):
    fixed_actions = []
    not_allowed = []

    for action in action_list:
        cmd = action.split()

        cmd = action.split()[0]
        
        if cmd in ALLOWED_COMMANDS:
            fixed_actions.append(action)
        else:
            # Replace invalid action
            fixed_actions.append("MOVE_TO 500 500")
            not_allowed.append(action)

    if not_allowed:
        # for debugging the not allowed actions for a task:
        save_actions_to_file(not_allowed, domain, example_id)
    
    return fixed_actions


def run_vision_subprocess_cpu(screenshot_bytes: bytes,
                          bbox: tuple[int, int, int, int]):
    """
    Call vision_CPU.py in a subprocess, passing it a PNG file path and bbox.
    """

    # 1. Save screenshot bytes to temporary PNG file for subprocess
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
        tmp.write(screenshot_bytes)
        tmp.flush()

    bbox_str = ",".join(str(x) for x in bbox)  # "0,0,1920,1080"

    cmd = [
        "python3",
        "CSE291_AI_Agent/ecua2_agent/vision_module/vision_CPU.py",
        "--img", temp_path,
        "--bbox", bbox_str,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("[ERROR] vision_CPU.py failed")
        print("stderr:", result.stderr)
        raise RuntimeError("vision_CPU.py failed")

    stdout = result.stdout.strip()

    # Try to find a JSON object in stdout (last JSON-looking line)
    data = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            break
        except json.JSONDecodeError:
            continue

    if data is None:
        print("[ERROR] JSON parse failed")
        print(stdout)
        raise RuntimeError("Invalid JSON from vision_CPU.py")

    # Unpack result
    vision_dict = data["vision"]

    return vision_dict

def run_vision_subprocess_gpu(screenshot_bytes: bytes):
    """
    Call vision_gpu.py in a subprocess, passing it a PNG file 
    """

    # 1. Save screenshot bytes to temporary PNG file for subprocess
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
        tmp.write(screenshot_bytes)
        tmp.flush()

    cmd = [
        "python3",
        "CSE291_AI_Agent/ecua2_agent/vision_module/vision_GPU/vision_gpu.py",
        "--img", temp_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("[ERROR] vision_gpu.py failed")
        print("stderr:", result.stderr)
        raise RuntimeError("vision_gpu.py failed")

    stdout = result.stdout.strip()

    # Try to find a JSON object in stdout (last JSON-looking line)
    data = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            break
        except json.JSONDecodeError:
            continue

    if data is None:
        print("[ERROR] JSON parse failed")
        print(stdout)
        raise RuntimeError("Invalid JSON from vision_gpu.py")

    # Unpack result
    vision_dict = data["vision"]

    return vision_dict

def parse_planner_action(line: str):
    """
    Convert planner output like:'HOTKEY win+E' or'MOVE_TO 400 600'
    into OSWorld's required format.
    """
    if not line:
        return None
    line = line.strip()

    # Special string actions
    if line in ("WAIT", "FAIL", "DONE"):
        return line

    parts = line.split()
    cmd = parts[0].upper()
    rest = [r.replace(",", "") for r in parts[1:]]

    # MOVE_TO x y
    if cmd == "MOVE_TO":
        if len(rest) != 2:
            raise ValueError(f"Invalid MOVE_TO: {line}")
        x, y = map(int, rest)
        return {
            "action_type": "MOVE_TO",
            "parameters": {"x": x, "y": y}
        }

    # CLICK
    if cmd == "CLICK":
        return {"action_type": "CLICK", "parameters": {}}

    # HOTKEY win+E
    if cmd == "HOTKEY":
        if not rest:
            raise ValueError(f"Invalid HOTKEY: {line}")
        keys = rest[0].split("+")
        return {
            "action_type": "HOTKEY",
            "parameters": {"keys": keys}
        }

    if cmd == "TYPING":
        # Everything after TYPING is text
        text = line[len("TYPING"):].strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return {
            "action_type": "TYPING",
            "parameters": {"text": text}
        }

    raise ValueError(f"Unknown action format: {line}")


def run_single_example(domain, example_id, env, example, max_steps, instruction, args, example_result_dir):
    # runtime_logger = setup_logger(example, example_result_dir)
    global parsed_json, actions_raw, response

    if hasattr(env, "set_max_steps_budget"):
        env.set_max_steps_budget(max_steps)
    elif hasattr(env, "max_steps_budget"):
        env.max_steps_budget = max_steps

    # Reset environment first to get fresh VM IP
    env.reset(task_config=example)

    # TODO Reset agent call here

    time.sleep(60) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation
   
    # vision call here as subprocess
    if args.v_gpu:
        parsed_json = run_vision_subprocess_gpu(obs['screenshot'])
    else:
        parsed_json = run_vision_subprocess_cpu(obs['screenshot'],(0,0,1920,1080))

    response, actions_raw = planner.generate_plan(instruction, parsed_json, (0,0,1920,1080), "CSE291_AI_Agent/ecua2_agent/planner_module/models/llama-3.2-3B-Instruct")
    actions = normalize_actions(actions_raw, domain, example_id)
    print("**************** actions %s", actions)

    done = False
    step_idx = 0
    env.controller.start_recording()
    while not done and step_idx < max_steps:

        #planner call here
        if args.single_shot_planner:
            response, actions_raw = planner.generate_plan(instruction, parsed_json, (0,0,1920,1080), "CSE291_AI_Agent/ecua2_agent/planner_module/models/llama-3.2-3B-Instruct")
        else:
            print("****** I am in vision and planner iterative call ***********")
            step_idx +=1
            obs = env._get_obs() 
            if args.v_gpu:
                parsed_json = run_vision_subprocess_gpu(obs['screenshot'])
            else:
                parsed_json = run_vision_subprocess_cpu(obs['screenshot'],(0,0,1920,1080))
                # call one action planner
            response, actions_raw = planner.generate_plan(instruction, parsed_json, (0,0,1920,1080), "CSE291_AI_Agent/ecua2_agent/planner_module/models/llama-3.2-3B-Instruct")
            
        # filter the action command to only accept computer_13 but 
        # we log the not allowed command to study the agent output
        actions = normalize_actions(actions_raw, domain, example_id)

        for action_str in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            parsed = parse_planner_action(action_str)

            if step_idx >= max_steps:
                logger.info("Reached max_steps=%d, stopping further actions.", max_steps)
                done = True
                break

            # this is important for osworld-human wes+ and wes-
            step_idx += 1

            logger.info("Step %d parsed: %s", step_idx, parsed)

            if parsed is None:
                logger.warning("Could not parse action: %s", action_str)
                continue

            # Execute action in OSWorld env
            obs, reward, done, info = env.step(parsed, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)

            # Save screenshot
            screenshot_path = os.path.join(
                example_result_dir,
                f"step_{step_idx}_{action_timestamp}.png"
            )
            with open(screenshot_path, "wb") as f_img:
                f_img.write(obs["screenshot"])

            # Save trajectory record
            traj_record = {
                "step_num": step_idx,
                "action_timestamp": action_timestamp,
                "action": parsed,
                "response": response,
                "reward": reward,
                "done": done,
                "info": info,
                "screenshot_file": os.path.basename(screenshot_path),
            }
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps(traj_record) + "\n")

            # If DONE, stop early
            if parsed == "DONE" or done:
                logger.info("Episode completed.")
                break

    time.sleep(20) # Wait for the environment to settle
    result = env.evaluate()
    logger.info("Result (OSWorld metric): %.2f", result)

    # get WES+ and WES-
    wes_plus = getattr(env, "wes_plus", None)
    wes_minus = getattr(env, "wes_minus", None)

    if wes_plus is not None and wes_minus is not None:
        logger.info("WES+ = %.4f, WES- = %.4f", wes_plus, wes_minus)

    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")
        if wes_plus is not None and wes_minus is not None:
            f.write(f"WES_PLUS={wes_plus}\n")
            f.write(f"WES_MINUS={wes_minus}\n")
    
    log_task_completion(example, result, example_result_dir, args)

    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
