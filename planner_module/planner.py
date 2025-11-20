import os
import json
import argparse
from vllm import LLM, SamplingParams


AVAILABLE_ACTIONS = """
    Action Type | Parameters | Description
    MOVE_TO | x, y | Move the cursor to the specified position
    CLICK | button, x, y, num_clicks | Click the left button if not specified, otherwise click the specified button; click at current position if x,y not specified, otherwise click at specified position
    MOUSE_DOWN | button | Press the left button if not specified, otherwise press the specified button
    MOUSE_UP | button | Release the left button if not specified, otherwise release the specified button
    RIGHT_CLICK | x, y | Right click at current position if x,y not specified, otherwise right click at specified position
    DOUBLE_CLICK | x, y | Double click at current position if x,y not specified, otherwise double click at specified position
    DRAG_TO | x, y | Drag the cursor to the specified position with the left button pressed
    SCROLL | dx, dy | Scroll the mouse wheel up or down
    TYPING | text | Type the specified text
    PRESS | key | Press the specified key and release it
    KEY_DOWN | key | Press the specified key
    KEY_UP | key | Release the specified key
    HOTKEY | keys | Press the specified key combination
    WAIT | - | Wait until the next action
    FAIL | - | Decide the task cannot be performed
    DONE | - | Decide the task is done
"""


def load_vision_json(vision_path):
    """Load vision output JSON file."""
    with open(vision_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(task, vision_data):
    """Build the prompt for the LLM."""
    # Extract screen context
    elements = vision_data.get("elements", [])
    window = vision_data.get("window", {})
    bbox = window.get("bbox", [0, 0, 1920, 1080])
    
    # Format UI elements
    ui_context = ""
    for el in elements[:100]:  # Limit to first 100 elements
        text = el.get("text", "").strip()
        role = el.get("role", "unknown")
        el_bbox = el.get("bbox", [0, 0, 0, 0])
        
        if text:
            ui_context += f'[{role}] "{text}" at ({el_bbox[0]}, {el_bbox[1]})\n'
    
    # Build full prompt with few-shot examples
    prompt = f"""You are a computer control system. Generate action commands.

AVAILABLE ACTIONS:
{AVAILABLE_ACTIONS}

SCREEN:
{ui_context}

Example 1:
TASK: Click on Gmail
OUTPUT:
MOVE_TO 3523 213
CLICK
DONE

Example 2:
TASK: Type hello in search box
OUTPUT:
MOVE_TO 1380 588
CLICK
TYPING "hello"
DONE

Example 3:
TASK: Open terminal
OUTPUT:
HOTKEY ctrl+alt+t
DONE

Now complete this task:
TASK: {task}
OUTPUT:
"""
    return prompt


def generate_plan(task, vision_path, model_path="./models/llama-3.2-1b", 
                 temperature=0.3, max_tokens=512):
    """
    Generate action plan using LLM.
    
    Args:
        task: Task description
        vision_path: Path to vision JSON file
        model_path: Path to LLM model
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        
    Returns:
        Raw LLM output string
    """
    # Load vision data
    vision_data = load_vision_json(vision_path)
    
    # Build prompt
    prompt = build_prompt(task, vision_data)
    print("=== Prompt ===")
    print(prompt)
    
    # Load and run LLM
    llm = LLM(
        model=model_path,
        dtype="float16",
        gpu_memory_utilization=0.75,
        max_model_len=2048,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        enforce_eager=True,
    )
    
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate([prompt], sampling_params)
    
    return outputs[0].outputs[0].text.strip()


def main():
    task = "Click on Syllabus"
    vision_file = "./vision_files/cpu_vision_output.json"
    model_path = "planner_module/models/llama-3.2-1b"
    temperature = 0.3
    max_tokens = 512
    
    print(f"Task: {task}")
    print(f"Vision file: {vision_file}")
    print("\nGenerating plan...\n")
    
    output = generate_plan(
        task=task,
        vision_path=vision_file,
        model_path=model_path,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    print("="*80)
    print("LLM OUTPUT:")
    print("="*80)
    print(output)
    print("="*80)


if __name__ == "__main__":
    main()