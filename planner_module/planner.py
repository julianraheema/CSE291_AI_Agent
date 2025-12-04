import os
import json
import argparse
from difflib import SequenceMatcher

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


def is_contained(bbox_a, bbox_b):
    """
    Check if bbox_a is fully contained within bbox_b.
    bbox format: [x1, y1, x2, y2]
    
    Args:
        bbox_a: Bounding box to check if contained
        bbox_b: Potential parent bounding box
        
    Returns:
        bool: True if bbox_a is inside bbox_b
    """
    return (bbox_a[0] >= bbox_b[0] and 
            bbox_a[1] >= bbox_b[1] and 
            bbox_a[2] <= bbox_b[2] and 
            bbox_a[3] <= bbox_b[3])


def build_spatial_tree(som_elements):
    """
    Build hierarchical containment tree from flat element list.
    Inspired by ScreenAI's structural layout understanding.
    
    Args:
        som_elements: Dictionary mapping element IDs to element data
        
    Returns:
        Dictionary mapping element IDs to their parent IDs and children IDs
    """
    hierarchy = {}
    
    for elem_id, elem_data in som_elements.items():
        hierarchy[elem_id] = {
            'parent': None,
            'children': [],
            'depth': 0
        }
    
    # Find parent-child relationships
    for child_id, child_data in som_elements.items():
        potential_parents = []
        
        for parent_id, parent_data in som_elements.items():
            if child_id == parent_id:
                continue
                
            # Check if child is contained in parent
            if is_contained(child_data['bbox'], parent_data['bbox']):
                # Calculate containment area to find the tightest parent
                parent_area = ((parent_data['bbox'][2] - parent_data['bbox'][0]) * 
                              (parent_data['bbox'][3] - parent_data['bbox'][1]))
                potential_parents.append((parent_id, parent_area))
        
        # Select the smallest containing element as the direct parent
        if potential_parents:
            potential_parents.sort(key=lambda x: x[1])  # Sort by area
            direct_parent = potential_parents[0][0]
            hierarchy[child_id]['parent'] = direct_parent
            hierarchy[direct_parent]['children'].append(child_id)
    
    # Calculate depth for each element
    def calculate_depth(elem_id, visited=None):
        if visited is None:
            visited = set()
        
        if elem_id in visited:
            return 0  # Prevent infinite loops
        visited.add(elem_id)
        
        parent_id = hierarchy[elem_id]['parent']
        if parent_id is None:
            hierarchy[elem_id]['depth'] = 0
            return 0
        else:
            depth = 1 + calculate_depth(parent_id, visited)
            hierarchy[elem_id]['depth'] = depth
            return depth
    
    for elem_id in hierarchy:
        calculate_depth(elem_id)
    
    return hierarchy


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
    window_dims = vision_data.get("window dims")
    screen_center_x = window_dims[0] / 2
    screen_center_y = window_dims[1] / 2
    
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
            
            # Calculate spatial centrality (inverse distance from screen center, normalized)
            distance_from_center = ((center_x - screen_center_x)**2 + (center_y - screen_center_y)**2)**0.5
            max_distance = ((screen_center_x)**2 + (screen_center_y)**2)**0.5
            centrality = 1.0 - (distance_from_center / max_distance)
            
            # Get confidence scores (handle both text and UI elements)
            confidence = el.get("confidence", el.get("det_confidence", 0.5))
            
            som_elements[element_id] = {
                "id": element_id,
                "text": text,
                "type": el_type,
                "bbox": bbox,
                "center": (center_x, center_y),
                "confidence": confidence,
                "centrality": centrality,
                "ui_class": el.get("ui_class", None)
            }
            element_id += 1
    
    return som_elements


def robust_find_element(user_query, som_elements):
    """
    Finds element ID based on fuzzy text matching + confidence scores + spatial centrality.
    Mitigates OCR errors (e.g., 'Getatgle' -> 'Google').
    
    Implements semantic-geometric grounding inspired by Pix2Struct and Mind2Web.
    
    Args:
        user_query: User's text query (e.g., "Google", "Gmail")
        som_elements: Dictionary of SoM elements
        
    Returns:
        tuple: (best_id, highest_score, match_details) or (None, 0.0, {}) if no match
    """
    best_id = None
    highest_score = 0.0
    match_details = {}
    
    for eid, data in som_elements.items():
        # Normalized Levenshtein distance using SequenceMatcher
        text_score = SequenceMatcher(None, user_query.lower(), data['text'].lower()).ratio()
        
        # Boost score if element is a UI button/icon rather than random text
        type_boost = 1.2 if data['type'] == 'ui' else 1.0
        
        # Boost for interactive UI elements (icons)
        ui_boost = 1.15 if data.get('ui_class') == 'icon' else 1.0
        
        # Combine text match, detection confidence, type, and spatial centrality
        # Text matching is most important, but confidence and centrality help disambiguate
        final_score = (
            text_score * 0.6 +           # 60% weight on text similarity
            data['confidence'] * 0.2 +   # 20% weight on OCR/detection confidence
            data['centrality'] * 0.2     # 20% weight on spatial centrality
        ) * type_boost * ui_boost
        
        # Threshold to filter out spurious matches
        if final_score > highest_score and text_score > 0.4:  # Require minimum text similarity
            highest_score = final_score
            best_id = eid
            match_details = {
                'text': data['text'],
                'text_score': text_score,
                'confidence': data['confidence'],
                'centrality': data['centrality'],
                'final_score': final_score,
                'type': data['type']
            }
    
    if best_id is not None:
        print(f"\n[Fuzzy Match] Query='{user_query}' -> Element {best_id}: '{match_details['text']}'")
        print(f"  Text similarity: {match_details['text_score']:.3f}")
        print(f"  Confidence: {match_details['confidence']:.3f}")
        print(f"  Centrality: {match_details['centrality']:.3f}")
        print(f"  Final score: {match_details['final_score']:.3f}")
    
    return best_id, highest_score, match_details


def build_prompt(task, vision_data, bbox, som_elements):
    """
    Build the prompt for the LLM using Set-of-Marks abstraction with hierarchical context.
    Implements spatial tree structure inspired by ScreenAI for better UI understanding.
    """
    
    # Build spatial containment tree
    hierarchy = build_spatial_tree(som_elements)
    
    # Sort elements by depth (roots first) and then by confidence
    sorted_elements = sorted(
        som_elements.items(),
        key=lambda x: (hierarchy[x[0]]['depth'], -(x[1]['confidence'] * x[1]['centrality']))
    )
    
    # Format UI elements with hierarchical structure
    ui_elements = []
    for elem_id, elem_data in sorted_elements:
        text = elem_data["text"]
        elem_type = elem_data["type"]
        ui_class = elem_data.get("ui_class", "")
        ui_class_str = f", {ui_class}" if ui_class else ""
        
        # Get hierarchy info
        parent_id = hierarchy[elem_id]['parent']
        children_ids = hierarchy[elem_id]['children']
        depth = hierarchy[elem_id]['depth']
        
        # Format with indentation to show hierarchy
        indent = "  " * depth
        
        # Build element description
        elem_desc = f'{indent}[{elem_id}] "{text}" ({elem_type}{ui_class_str})'
        
        # Add parent/children context
        if parent_id is not None:
            parent_text = som_elements[parent_id]['text']
            elem_desc += f' → inside [{parent_id}] "{parent_text}"'
        
        if children_ids:
            children_info = ", ".join([str(cid) for cid in children_ids[:3]])  # Show first 3 children
            if len(children_ids) > 3:
                children_info += f", +{len(children_ids) - 3} more"
            elem_desc += f' → contains [{children_info}]'
        
        ui_elements.append(elem_desc)
    
    ui_context = "\n".join(ui_elements)
    
    # Build prompt with hierarchical SoM instructions
    prompt = f"""You are a computer control assistant. You have access to UI elements identified by IDs.

Available UI Elements (hierarchical structure, indented by containment):
{ui_context}

Structural Context:
- Elements are organized by parent-child containment relationships
- Indentation shows depth in the UI hierarchy (like a DOM tree)
- "→ inside [X]" means this element is contained within element X
- "→ contains [...]" lists child elements inside this element
- Use structural context to disambiguate similar elements (e.g., Search in Navbar vs Search in Footer)

Available Actions:
CLICK_ELEMENT <id> - click on element by ID
TYPING "text" - type text
HOTKEY keys - keyboard shortcut (e.g., ctrl+alt+t)
DONE - finish task

Instructions:
- Use CLICK_ELEMENT with the ID number to interact with UI elements
- OCR text may contain errors (e.g., 'Getatgle' might be 'Google')
- UI icons are more reliable than plain text elements
- Consider parent-child relationships when selecting elements
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
    Falls back to fuzzy matching if exact ID is not found.
    
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
                    # Not an integer - might be text query, try fuzzy matching
                    query_text = parts[1].strip()
                    print(f"\n[Attempting fuzzy match for non-numeric ID: '{query_text}']")
                    best_id, score, details = robust_find_element(query_text, som_elements)
                    if best_id is not None:
                        elem = som_elements[best_id]
                        center_x, center_y = elem["center"]
                        translated_actions.append(f"CLICK {center_x} {center_y}")
                        print(f"  -> Successfully mapped to element {best_id}")
                    else:
                        print(f"Warning: No fuzzy match found for '{query_text}'")
            else:
                print(f"Warning: CLICK_ELEMENT missing ID: {action}")
        else:
            # Pass through other actions unchanged
            translated_actions.append(action)
    
    return translated_actions


def generate_plan(task, vision_data,  bbox, model_path, temperature=0.3, max_tokens=512):
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
        tuple: (raw_output, coordinate_actions)
            - raw_output: Raw LLM output string
            - coordinate_actions: Actions translated to coordinates
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
        max_model_len=4048,
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
    print("\nGenerating plan...\n")

    # Load vision data
    vision_data = load_vision_json(vision_file)
    
    output, coordinate_actions = generate_plan(
        task=task,
        vision_data=vision_data,
        bbox=[0, 0, 3840, 2160],
        model_path=model_path,
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


#####################old planner below##########################


# import os
# import json
# import argparse
# from difflib import SequenceMatcher
# from vllm import LLM, SamplingParams


# AVAILABLE_ACTIONS = """
#     Action Type | Parameters | Description
#     MOVE_TO | x, y | Move the cursor to the specified position
#     CLICK | button, x, y, num_clicks | Click the left button if not specified, otherwise click the specified button; click at current position if x,y not specified, otherwise click at specified position
#     MOUSE_DOWN | button | Press the left button if not specified, otherwise press the specified button
#     MOUSE_UP | button | Release the left button if not specified, otherwise release the specified button
#     RIGHT_CLICK | x, y | Right click at current position if x,y not specified, otherwise right click at specified position
#     DOUBLE_CLICK | x, y | Double click at current position if x,y not specified, otherwise double click at specified position
#     DRAG_TO | x, y | Drag the cursor to the specified position with the left button pressed
#     SCROLL | dx, dy | Scroll the mouse wheel up or down
#     TYPING | text | Type the specified text
#     PRESS | key | Press the specified key and release it
#     KEY_DOWN | key | Press the specified key
#     KEY_UP | key | Release the specified key
#     HOTKEY | keys | Press the specified key combination
#     WAIT | - | Wait until the next action
#     FAIL | - | Decide the task cannot be performed
#     DONE | - | Decide the task is done
# """


# def load_vision_json(vision_path):
#     """Load vision output JSON file."""
#     with open(vision_path, "r", encoding="utf-8") as f:
#         return json.load(f)


# def is_contained(bbox_a, bbox_b):
#     """
#     Check if bbox_a is fully contained within bbox_b.
#     bbox format: [x1, y1, x2, y2]
    
#     Args:
#         bbox_a: Bounding box to check if contained
#         bbox_b: Potential parent bounding box
        
#     Returns:
#         bool: True if bbox_a is inside bbox_b
#     """
#     return (bbox_a[0] >= bbox_b[0] and 
#             bbox_a[1] >= bbox_b[1] and 
#             bbox_a[2] <= bbox_b[2] and 
#             bbox_a[3] <= bbox_b[3])


# def build_spatial_tree(som_elements):
#     """
#     Build hierarchical containment tree from flat element list.
#     Inspired by ScreenAI's structural layout understanding.
    
#     Args:
#         som_elements: Dictionary mapping element IDs to element data
        
#     Returns:
#         Dictionary mapping element IDs to their parent IDs and children IDs
#     """
#     hierarchy = {}
    
#     for elem_id, elem_data in som_elements.items():
#         hierarchy[elem_id] = {
#             'parent': None,
#             'children': [],
#             'depth': 0
#         }
    
#     # Find parent-child relationships
#     for child_id, child_data in som_elements.items():
#         potential_parents = []
        
#         for parent_id, parent_data in som_elements.items():
#             if child_id == parent_id:
#                 continue
                
#             # Check if child is contained in parent
#             if is_contained(child_data['bbox'], parent_data['bbox']):
#                 # Calculate containment area to find the tightest parent
#                 parent_area = ((parent_data['bbox'][2] - parent_data['bbox'][0]) * 
#                               (parent_data['bbox'][3] - parent_data['bbox'][1]))
#                 potential_parents.append((parent_id, parent_area))
        
#         # Select the smallest containing element as the direct parent
#         if potential_parents:
#             potential_parents.sort(key=lambda x: x[1])  # Sort by area
#             direct_parent = potential_parents[0][0]
#             hierarchy[child_id]['parent'] = direct_parent
#             hierarchy[direct_parent]['children'].append(child_id)
    
#     # Calculate depth for each element
#     def calculate_depth(elem_id, visited=None):
#         if visited is None:
#             visited = set()
        
#         if elem_id in visited:
#             return 0  # Prevent infinite loops
#         visited.add(elem_id)
        
#         parent_id = hierarchy[elem_id]['parent']
#         if parent_id is None:
#             hierarchy[elem_id]['depth'] = 0
#             return 0
#         else:
#             depth = 1 + calculate_depth(parent_id, visited)
#             hierarchy[elem_id]['depth'] = depth
#             return depth
    
#     for elem_id in hierarchy:
#         calculate_depth(elem_id)
    
#     return hierarchy


# def create_som_elements(vision_data):
#     """
#     Create Set-of-Marks (SoM) abstraction from vision data.
#     Assigns unique IDs to each UI element for easier LLM interaction.
    
#     Args:
#         vision_data: Vision output dictionary
        
#     Returns:
#         Dictionary mapping element IDs to element data with coordinates
#     """
#     som_elements = {}
#     elements = vision_data.get("elements", [])
#     window_dims = vision_data.get("window dims")
#     screen_center_x = window_dims[0] / 2
#     screen_center_y = window_dims[1] / 2
    
#     element_id = 1
#     for el in elements:
#         text = el.get("text", "").strip()
#         bbox = el.get("bbox", [0, 0, 0, 0])
#         el_type = el.get("type", "unknown")
        
#         # Only include elements with meaningful text
#         if text and len(text) > 1:
#             # Calculate center coordinates for easier clicking
#             center_x = (bbox[0] + bbox[2]) // 2
#             center_y = (bbox[1] + bbox[3]) // 2
            
#             # Calculate spatial centrality (inverse distance from screen center, normalized)
#             distance_from_center = ((center_x - screen_center_x)**2 + (center_y - screen_center_y)**2)**0.5
#             max_distance = ((screen_center_x)**2 + (screen_center_y)**2)**0.5
#             centrality = 1.0 - (distance_from_center / max_distance)
            
#             # Get confidence scores (handle both text and UI elements)
#             confidence = el.get("confidence", el.get("det_confidence", 0.5))
            
#             som_elements[element_id] = {
#                 "id": element_id,
#                 "text": text,
#                 "type": el_type,
#                 "bbox": bbox,
#                 "center": (center_x, center_y),
#                 "confidence": confidence,
#                 "centrality": centrality,
#                 "ui_class": el.get("ui_class", None)
#             }
#             element_id += 1
    
#     return som_elements


# def robust_find_element(user_query, som_elements):
#     """
#     Finds element ID based on fuzzy text matching + confidence scores + spatial centrality.
#     Mitigates OCR errors (e.g., 'Getatgle' -> 'Google').
    
#     Implements semantic-geometric grounding inspired by Pix2Struct and Mind2Web.
    
#     Args:
#         user_query: User's text query (e.g., "Google", "Gmail")
#         som_elements: Dictionary of SoM elements
        
#     Returns:
#         tuple: (best_id, highest_score, match_details) or (None, 0.0, {}) if no match
#     """
#     best_id = None
#     highest_score = 0.0
#     match_details = {}
    
#     for eid, data in som_elements.items():
#         # Normalized Levenshtein distance using SequenceMatcher
#         text_score = SequenceMatcher(None, user_query.lower(), data['text'].lower()).ratio()
        
#         # Boost score if element is a UI button/icon rather than random text
#         type_boost = 1.2 if data['type'] == 'ui' else 1.0
        
#         # Boost for interactive UI elements (icons)
#         ui_boost = 1.15 if data.get('ui_class') == 'icon' else 1.0
        
#         # Combine text match, detection confidence, type, and spatial centrality
#         # Text matching is most important, but confidence and centrality help disambiguate
#         final_score = (
#             text_score * 0.6 +           # 60% weight on text similarity
#             data['confidence'] * 0.2 +   # 20% weight on OCR/detection confidence
#             data['centrality'] * 0.2     # 20% weight on spatial centrality
#         ) * type_boost * ui_boost
        
#         # Threshold to filter out spurious matches
#         if final_score > highest_score and text_score > 0.4:  # Require minimum text similarity
#             highest_score = final_score
#             best_id = eid
#             match_details = {
#                 'text': data['text'],
#                 'text_score': text_score,
#                 'confidence': data['confidence'],
#                 'centrality': data['centrality'],
#                 'final_score': final_score,
#                 'type': data['type']
#             }
    
#     if best_id is not None:
#         print(f"\n[Fuzzy Match] Query='{user_query}' -> Element {best_id}: '{match_details['text']}'")
#         print(f"  Text similarity: {match_details['text_score']:.3f}")
#         print(f"  Confidence: {match_details['confidence']:.3f}")
#         print(f"  Centrality: {match_details['centrality']:.3f}")
#         print(f"  Final score: {match_details['final_score']:.3f}")
    
#     return best_id, highest_score, match_details


# def build_prompt(task, vision_data, bbox, som_elements):
#     """
#     Build the prompt for the LLM using Set-of-Marks abstraction with hierarchical context.
#     Implements spatial tree structure inspired by ScreenAI for better UI understanding.
#     """
    
#     # Build spatial containment tree
#     hierarchy = build_spatial_tree(som_elements)
    
#     # Sort elements by depth (roots first) and then by confidence
#     sorted_elements = sorted(
#         som_elements.items(),
#         key=lambda x: (hierarchy[x[0]]['depth'], -(x[1]['confidence'] * x[1]['centrality']))
#     )
    
#     # Format UI elements with hierarchical structure
#     ui_elements = []
#     for elem_id, elem_data in sorted_elements:
#         text = elem_data["text"]
#         elem_type = elem_data["type"]
#         ui_class = elem_data.get("ui_class", "")
#         ui_class_str = f", {ui_class}" if ui_class else ""
        
#         # Get hierarchy info
#         parent_id = hierarchy[elem_id]['parent']
#         children_ids = hierarchy[elem_id]['children']
#         depth = hierarchy[elem_id]['depth']
        
#         # Format with indentation to show hierarchy
#         indent = "  " * depth
        
#         # Build element description
#         elem_desc = f'{indent}[{elem_id}] "{text}" ({elem_type}{ui_class_str})'
        
#         # Add parent/children context
#         if parent_id is not None:
#             parent_text = som_elements[parent_id]['text']
#             elem_desc += f' → inside [{parent_id}] "{parent_text}"'
        
#         if children_ids:
#             children_info = ", ".join([str(cid) for cid in children_ids[:3]])  # Show first 3 children
#             if len(children_ids) > 3:
#                 children_info += f", +{len(children_ids) - 3} more"
#             elem_desc += f' → contains [{children_info}]'
        
#         ui_elements.append(elem_desc)
    
#     ui_context = "\n".join(ui_elements)
    
#     # Build prompt with hierarchical SoM instructions
#     prompt = f"""You are a computer control assistant. You have access to UI elements identified by IDs.

# Available UI Elements (hierarchical structure, indented by containment):
# {ui_context}

# Structural Context:
# - Elements are organized by parent-child containment relationships
# - Indentation shows depth in the UI hierarchy (like a DOM tree)
# - "→ inside [X]" means this element is contained within element X
# - "→ contains [...]" lists child elements inside this element
# - Use structural context to disambiguate similar elements (e.g., Search in Navbar vs Search in Footer)

# Available Actions:
# CLICK_ELEMENT <id> - click on element by ID
# TYPING "text" - type text
# HOTKEY keys - keyboard shortcut (e.g., ctrl+alt+t)
# DONE - finish task

# Instructions:
# - Use CLICK_ELEMENT with the ID number to interact with UI elements
# - OCR text may contain errors (e.g., 'Getatgle' might be 'Google')
# - UI icons are more reliable than plain text elements
# - Consider parent-child relationships when selecting elements
# - Use TYPING to enter text after clicking input fields
# - Use HOTKEY for keyboard shortcuts
# - End with DONE when task is complete

# Examples:

# Task: Click on Gmail
# CLICK_ELEMENT 2
# DONE

# Task: Search for python
# CLICK_ELEMENT 4
# TYPING "python"
# DONE

# Task: Open terminal
# HOTKEY ctrl+alt+t
# DONE

# Task: {task}
# """
#     return prompt


# def is_valid_action(line):
#     """Check if a line is a valid action command."""
#     line = line.strip()
#     if not line:
#         return False
    
#     # List of valid action prefixes (including new SoM actions)
#     valid_actions = [
#         "MOVE_TO", "CLICK", "MOUSE_DOWN", "MOUSE_UP", 
#         "RIGHT_CLICK", "DOUBLE_CLICK", "DRAG_TO", 
#         "SCROLL", "TYPING", "PRESS", "KEY_DOWN", 
#         "KEY_UP", "HOTKEY", "WAIT", "FAIL", "DONE",
#         "CLICK_ELEMENT"  # New SoM action
#     ]
    
#     # Check if line starts with any valid action
#     first_word = line.split()[0] if line.split() else ""
#     return first_word in valid_actions


# def translate_som_to_coordinates(actions, som_elements):
#     """
#     Translate SoM actions (using IDs) to coordinate-based actions.
#     Falls back to fuzzy matching if exact ID is not found.
    
#     Args:
#         actions: List of action strings (may include CLICK_ELEMENT commands)
#         som_elements: Dictionary mapping element IDs to element data
        
#     Returns:
#         List of translated actions with coordinates instead of IDs
#     """
#     translated_actions = []
    
#     for action in actions:
#         parts = action.strip().split(maxsplit=1)
#         if not parts:
#             continue
            
#         action_type = parts[0]
        
#         if action_type == "CLICK_ELEMENT":
#             # Extract element ID and translate to coordinates
#             if len(parts) > 1:
#                 try:
#                     elem_id = int(parts[1])
#                     if elem_id in som_elements:
#                         elem = som_elements[elem_id]
#                         center_x, center_y = elem["center"]
#                         # Translate to CLICK at coordinates
#                         translated_actions.append(f"CLICK {center_x} {center_y}")
#                     else:
#                         # Invalid ID - skip or log warning
#                         print(f"Warning: Element ID {elem_id} not found")
#                 except ValueError:
#                     # Not an integer - might be text query, try fuzzy matching
#                     query_text = parts[1].strip()
#                     print(f"\n[Attempting fuzzy match for non-numeric ID: '{query_text}']")
#                     best_id, score, details = robust_find_element(query_text, som_elements)
#                     if best_id is not None:
#                         elem = som_elements[best_id]
#                         center_x, center_y = elem["center"]
#                         translated_actions.append(f"CLICK {center_x} {center_y}")
#                         print(f"  -> Successfully mapped to element {best_id}")
#                     else:
#                         print(f"Warning: No fuzzy match found for '{query_text}'")
#             else:
#                 print(f"Warning: CLICK_ELEMENT missing ID: {action}")
#         else:
#             # Pass through other actions unchanged
#             translated_actions.append(action)
    
#     return translated_actions


# def generate_plan(task, vision_data, model_path, bbox, temperature=0.3, max_tokens=512):
#     """
#     Generate action plan using LLM with Set-of-Marks abstraction.
    
#     Args:
#         task: Task description
#         vision_data: Vision output dictionary
#         model_path: Path to LLM model
#         bbox: Bounding box [x, y, width, height]
#         temperature: Sampling temperature
#         max_tokens: Max tokens to generate
        
#     Returns:
#         tuple: (raw_output, coordinate_actions)
#             - raw_output: Raw LLM output string
#             - coordinate_actions: Actions translated to coordinates
#     """
    
#     # Create SoM abstraction
#     som_elements = create_som_elements(vision_data)
    
#     # Build prompt with SoM
#     prompt = build_prompt(task, vision_data, bbox, som_elements)
#     print("=== Prompt ===")
#     print(prompt)
    
#     # Load and run LLM
#     llm = LLM(
#         model=model_path,
#         dtype="float16",
#         gpu_memory_utilization=0.75,
#         max_model_len=2048,
#         max_num_seqs=1,
#         max_num_batched_tokens=512,
#         enforce_eager=True,
#     )
    
#     sampling_params = SamplingParams(
#         temperature=temperature, 
#         max_tokens=max_tokens,
#         stop=["DONE", "FAIL"]
#     )
#     outputs = llm.generate([prompt], sampling_params)
    
#     # Get output and add back the stop token
#     output = outputs[0].outputs[0].text.strip()
#     stop_reason = outputs[0].outputs[0].stop_reason
    
#     # Add back DONE or FAIL if it was the stop reason
#     if stop_reason and stop_reason in ["DONE", "FAIL"]:
#         output += f"\n{stop_reason}"
    
#     # Parse output into action list - filter to valid actions only
#     som_actions = []
#     for line in output.split('\n'):
#         line = line.strip()
#         if is_valid_action(line):
#             som_actions.append(line)
    
#     # Translate SoM actions to coordinate-based actions
#     coordinate_actions = translate_som_to_coordinates(som_actions, som_elements)

#     print("\nSOM ACTIONS (Element IDs):")
#     print("="*80)
#     for action in som_actions:
#         print(action)
#     print("="*80)
    
#     return output, coordinate_actions


# def main():
#     task = "Open Gmail and search for 'meeting notes'"
#     vision_file = "./vision_files/gpu_output.json"
#     model_path = "planner_module/models/llama-3.2-1b"
#     temperature = 0.3
#     max_tokens = 512
    
#     print(f"Task: {task}")
#     print(f"Vision file: {vision_file}")
#     print("\nGenerating plan...\n")

#     # Load vision data
#     vision_data = load_vision_json(vision_file)
    
#     output, coordinate_actions = generate_plan(
#         task=task,
#         vision_data=vision_data,
#         model_path=model_path,
#         bbox=[0, 0, 3840, 2160],
#         temperature=temperature,
#         max_tokens=max_tokens,
#     )
    
#     print("="*80)
#     print("LLM OUTPUT (Raw):")
#     print("="*80)
#     print(output)
#     print("="*80)
    
#     print("\nTRANSLATED ACTIONS (Coordinates):")
#     print("="*80)
#     for action in coordinate_actions:
#         print(action)
#     print("="*80)


# if __name__ == "__main__":
#     main()