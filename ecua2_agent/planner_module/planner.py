import os
import json
import argparse

os.environ["VLLM_PLUGINS"] = "none"
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


def build_prompt(task, vision_data, bbox):
    """Build the prompt for the LLM."""
    # Extract screen context
    elements = vision_data.get("elements", [])
    window = vision_data.get("window", {})
    bbox = window.get("bbox", bbox)
    
    # Format UI elements - only show elements with text
    ui_elements = []
    for el in elements[:50]:  # Limit to first 50 elements
        text = el.get("text", "").strip()
        el_bbox = el.get("bbox", [0, 0, 0, 0])
        
        if text and len(text) > 1:  # Only include meaningful text
            ui_elements.append(f'"{text}" at ({el_bbox[0]}, {el_bbox[1]})')
    
    ui_context = "\n".join(ui_elements)
    
    # Build focused prompt with strict examples
    prompt = f"""Convert the task into computer actions using ONLY these actions:
        MOVE_TO x y - move cursor
        CLICK - click mouse
        TYPING "text" - type text
        HOTKEY keys - keyboard shortcut
        DONE - finish

        Screen elements:
        {ui_context}

        Examples:

        Task: Click on Gmail
        MOVE_TO 3523 213
        CLICK
        DONE

        Task: Search for python
        MOVE_TO 1380 588
        CLICK
        TYPING "python"
        DONE

        Task: Open terminal
        HOTKEY ctrl+alt+t
        DONE

        Task: {task}
    """
    return prompt


def generate_plan(task, vision_data, bbox, model_path, temperature=0.3, max_tokens=512):
    """
    Generate action plan using LLM.
    
    Args:
        task: Task description
        vision_data: Vision output dictionary
        model_path: Path to LLM model
        bbox: Bounding box [x, y, width, height]
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        
    Returns:
        output (raw string) and actions (list of action strings)
    """
    
    # Build prompt
    prompt = build_prompt(task, vision_data, bbox)
    # print("=== Prompt ===")
    # print(prompt)
    
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
    
    sampling_params = SamplingParams(
        temperature=temperature, 
        max_tokens=max_tokens,
        stop=["DONE", "FAIL"]
    )
    outputs = llm.generate([prompt], sampling_params)
    
    # Get output and add back the stop token
    output = outputs[0].outputs[0].text.strip()
    stop_reason = outputs[0].outputs[0].stop_reason
    
    # Add back DONE or FAIL if it was the stop reason
    if stop_reason and stop_reason in ["DONE", "FAIL"]:
        output += f"\n{stop_reason}"
    
    # Parse output into action list
    actions = []
    for line in output.split('\n'):
        line = line.strip()
        if line:  # Skip empty lines
            actions.append(line)
    
    return output, actions


def main():
    task = "Search cat pictures in google search bar"
    vision_file = "ecua2_agent/vision_module/vision_files/cpu_vision_output.json"
    model_path = "ecua2_agent/planner_module/models/llama-3.2-1b"
    temperature = 0.3
    max_tokens = 512
    
    print(f"Task: {task}")
    print(f"Vision file: {vision_file}")
    print("\nGenerating plan...\n")

    # Load vision data
    vision_data = load_vision_json(vision_file)
    
    output, actions = generate_plan(
        task=task,
        vision_data=vision_data,
        bbox=[0, 0, 1920, 1080],
        model_path=model_path,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    print("="*80)
    print("LLM OUTPUT:")
    print("="*80)
    print(output)
    print("="*80)
    
    print("\nACTIONS LIST:")
    print("="*80)
    print(actions)
    print("="*80)


if __name__ == "__main__":
    main()