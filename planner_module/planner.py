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


def create_som_elements(vision_data):
    """
    Create Set-of-Marks (SoM) abstraction from vision data.
    Assigns unique IDs to each UI element for easier LLM interaction.
    
    Args:
        vision_data: Vision output dictionary
        
    Returns:
        Dictionary mapping element IDs to element data with coordinates
    """
    som_elements = {}
    elements = vision_data.get("elements", [])
    
    element_id = 1
    for el in elements:
        text = el.get("text", "").strip()
        bbox = el.get("bbox", [0, 0, 0, 0])
        el_type = el.get("type", "unknown")
        
        # Only include elements with meaningful text
        if text and len(text) > 1:
            # Calculate center coordinates for easier clicking
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            som_elements[element_id] = {
                "id": element_id,
                "text": text,
                "type": el_type,
                "bbox": bbox,
                "center": (center_x, center_y),
                "confidence": el.get("confidence", 0.0)
            }
            element_id += 1
    
    return som_elements


def build_prompt(task, vision_data, bbox, som_elements):
    """Build the prompt for the LLM using Set-of-Marks abstraction."""
    
    # Format UI elements using SoM - show ID and text
    ui_elements = []
    for elem_id, elem_data in sorted(som_elements.items()):
        text = elem_data["text"]
        elem_type = elem_data["type"]
        ui_elements.append(f'[{elem_id}] "{text}" ({elem_type})')
    
    ui_context = "\n".join(ui_elements)
    
    # Build prompt with SoM instructions
    prompt = f"""You are a computer control assistant. You have access to UI elements identified by IDs.

Available UI Elements (use ID numbers):
{ui_context}

Available Actions:
CLICK_ELEMENT <id> - click on element by ID
TYPING "text" - type text
HOTKEY keys - keyboard shortcut (e.g., ctrl+alt+t)
DONE - finish task

Instructions:
- Use CLICK_ELEMENT with the ID number to interact with UI elements
- Use TYPING to enter text after clicking input fields
- Use HOTKEY for keyboard shortcuts
- End with DONE when task is complete

Examples:

Task: Click on Gmail
CLICK_ELEMENT 2
DONE

Task: Search for python
CLICK_ELEMENT 4
TYPING "python"
DONE

Task: Open terminal
HOTKEY ctrl+alt+t
DONE

Task: {task}
"""
    return prompt


def is_valid_action(line):
    """Check if a line is a valid action command."""
    line = line.strip()
    if not line:
        return False
    
    # List of valid action prefixes (including new SoM actions)
    valid_actions = [
        "MOVE_TO", "CLICK", "MOUSE_DOWN", "MOUSE_UP", 
        "RIGHT_CLICK", "DOUBLE_CLICK", "DRAG_TO", 
        "SCROLL", "TYPING", "PRESS", "KEY_DOWN", 
        "KEY_UP", "HOTKEY", "WAIT", "FAIL", "DONE",
        "CLICK_ELEMENT"  # New SoM action
    ]
    
    # Check if line starts with any valid action
    first_word = line.split()[0] if line.split() else ""
    return first_word in valid_actions


def translate_som_to_coordinates(actions, som_elements):
    """
    Translate SoM actions (using IDs) to coordinate-based actions.
    
    Args:
        actions: List of action strings (may include CLICK_ELEMENT commands)
        som_elements: Dictionary mapping element IDs to element data
        
    Returns:
        List of translated actions with coordinates instead of IDs
    """
    translated_actions = []
    
    for action in actions:
        parts = action.strip().split(maxsplit=1)
        if not parts:
            continue
            
        action_type = parts[0]
        
        if action_type == "CLICK_ELEMENT":
            # Extract element ID and translate to coordinates
            if len(parts) > 1:
                try:
                    elem_id = int(parts[1])
                    if elem_id in som_elements:
                        elem = som_elements[elem_id]
                        center_x, center_y = elem["center"]
                        # Translate to CLICK at coordinates
                        translated_actions.append(f"CLICK {center_x} {center_y}")
                    else:
                        # Invalid ID - skip or log warning
                        print(f"Warning: Element ID {elem_id} not found")
                except ValueError:
                    print(f"Warning: Invalid element ID in action: {action}")
            else:
                print(f"Warning: CLICK_ELEMENT missing ID: {action}")
        else:
            # Pass through other actions unchanged
            translated_actions.append(action)
    
    return translated_actions


def generate_plan(task, vision_data, model_path, bbox, temperature=0.3, max_tokens=512):
    """
    Generate action plan using LLM with Set-of-Marks abstraction.
    
    Args:
        task: Task description
        vision_data: Vision output dictionary
        model_path: Path to LLM model
        bbox: Bounding box [x, y, width, height]
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        
    Returns:
        tuple: (raw_output, som_actions, coordinate_actions, som_elements)
            - raw_output: Raw LLM output string
            - som_actions: Actions using element IDs
            - coordinate_actions: Actions translated to coordinates
            - som_elements: SoM element mapping for reference
    """
    
    # Create SoM abstraction
    som_elements = create_som_elements(vision_data)
    
    # Build prompt with SoM
    prompt = build_prompt(task, vision_data, bbox, som_elements)
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
    
    # Parse output into action list - filter to valid actions only
    som_actions = []
    for line in output.split('\n'):
        line = line.strip()
        if is_valid_action(line):
            som_actions.append(line)
    
    # Translate SoM actions to coordinate-based actions
    coordinate_actions = translate_som_to_coordinates(som_actions, som_elements)

    print("\nSOM ACTIONS (Element IDs):")
    print("="*80)
    for action in som_actions:
        print(action)
    print("="*80)
    
    return output, coordinate_actions


def main():
    task = "Open Gmail and search for 'meeting notes'"
    vision_file = "./vision_files/gpu_output.json"
    model_path = "planner_module/models/llama-3.2-1b"
    temperature = 0.3
    max_tokens = 512
    
    print(f"Task: {task}")
    print(f"Vision file: {vision_file}")
    print("\nGenerating plan with Set-of-Marks abstraction...\n")

    # Load vision data
    vision_data = load_vision_json(vision_file)
    
    output, coordinate_actions = generate_plan(
        task=task,
        vision_data=vision_data,
        model_path=model_path,
        bbox=[0, 0, 3840, 2160],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    print("="*80)
    print("LLM OUTPUT (Raw):")
    print("="*80)
    print(output)
    print("="*80)
    
    print("\nTRANSLATED ACTIONS (Coordinates):")
    print("="*80)
    for action in coordinate_actions:
        print(action)
    print("="*80)


if __name__ == "__main__":
    main()