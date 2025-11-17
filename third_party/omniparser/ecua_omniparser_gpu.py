#!/usr/bin/env python3
"""
OmniParser v2 GPU pipeline:
- Captures full screen (primary monitor)
- Runs OmniParser v2 (YOLOv8 UI detector + Florence-2 captioner + OCR merge)
- Produces a single elements[] array similar to your OCR/YOLO schema
"""

import os, json, uuid, tempfile, base64
from io import BytesIO
from typing import List, Dict, Any, Tuple

import torch
import time
import numpy as np
from mss import mss
from PIL import Image
import cv2

# --- IMPORTANT ---
# Run this from the OmniParser repo root, or set PYTHONPATH to it:
#   export PYTHONPATH=$PYTHONPATH:/path/to/OmniParser
# These utilities are the common OmniParser entry points used in their demo scripts.
from util.utils import (
    check_ocr_box,                 # OCR boxes + texts
    get_yolo_model,                # YOLOv8 UI detector loader
    get_caption_model_processor,   # Florence-2 captioner loader
    get_som_labeled_img            # OmniParser core (merge + labels)
)

# -------- Config (edit paths to your weights) --------
YOLO_MODEL_PATH      = "weights/icon_detect/model.pt"            # e.g., a fine-tuned YOLOv8 (n/s)
CAPTION_MODEL_DIR    = "weights/icon_caption_florence"           # Florence-2 Base (finetuned) dir
BOX_THRESHOLD        = 0.05                                      # OmniParser box threshold (tune)
IOU_THRESHOLD        = 0.10                                      # merge threshold for overlap
OCR_TEXT_THRESHOLD   = 0.90                                     # OCR internal text threshold
OUTPUT_COORD_RATIO   = True                                      # return coords normalized [0,1]

# -------- Device --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _is_base64_image(s: str) -> bool:
    # Detect "data:image/...;base64,xxxx" or big base64-looking strings
    if not isinstance(s, str): 
        return False
    if s.startswith("data:image/") and ";base64," in s:
        return True
    # Heuristic: long, mostly base64 chars
    if len(s) > 200 and all(c.isalnum() or c in "+/=\n\r" for c in s[:400]):
        return True
    return False

def _base64_to_pil(s: str) -> Image.Image:
    if s.startswith("data:image/"):
        s = s.split(",", 1)[1]
    raw = base64.b64decode(s)
    return Image.open(BytesIO(raw)).convert("RGB")

def _to_pil_image(x):
    """Coerce OmniParser overlay to PIL.Image without printing."""
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            return Image.fromarray(x)
        if x.ndim == 3:
            if x.shape[0] in (1,3,4) and x.shape[-1] not in (1,3,4):
                x = np.transpose(x, (1,2,0))
            return Image.fromarray(x.astype(np.uint8))
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1,3,4) and t.shape[-1] not in (1,3,4):
            t = t.permute(1,2,0)
        return _to_pil_image(t.numpy())
    if isinstance(x, (bytes, bytearray)):
        return Image.open(BytesIO(x)).convert("RGB")
    if isinstance(x, str):
        # 1) base64 image string
        if _is_base64_image(x):
            return _base64_to_pil(x)
        # 2) filesystem path
        if os.path.exists(x):
            return Image.open(x).convert("RGB")
        # Not a path? Treat as unsupported overlay type; skip image.
        raise TypeError("Overlay string is neither a file path nor base64 image.")
    raise TypeError(f"Unsupported overlay type: {type(x)}")

# -------- Screen capture --------
def capture_fullscreen_rgb() -> Tuple[Image.Image, Tuple[int, int]]:
    """Return PIL RGB image and (width, height) of primary monitor."""
    with mss() as sct:
        mon = sct.monitors[1]  # 0 = virtual full, 1 = primary
        raw = sct.grab(mon)    # BGRA
        img = Image.frombytes("RGB", raw.size, raw.rgb)  # RGB
        return img, (raw.size.width, raw.size.height)

# -------- OmniParser init --------
def init_models():
    print(f"[omni] device: {DEVICE}")
    print("[omni] loading YOLO (UI detector)…")
    yolo_model = get_yolo_model(model_path=YOLO_MODEL_PATH)

    print("[omni] loading Florence-2 captioner…")
    caption_model_processor = get_caption_model_processor(
        model_name="florence2",
        model_name_or_path=CAPTION_MODEL_DIR,
        device=DEVICE,
    )
    return yolo_model, caption_model_processor

# -------- Run OmniParser on a screenshot path --------
def run_omni(
    image_path: str,
    yolo_model,
    caption_model_processor,
    save_overlay: bool = False,
    overlay_out_path: str | None = None,
):
    """
    Returns:
      annotated_pil: PIL.Image or None
      label_coordinates: dict[str, list]  # {"0": [x,y,w,h], ...}
      parsed_content_list: list[str]      # e.g., ["Text Box ID 0: ...", ...]
      overlay_path: str | None            # saved file if save_overlay=True
    """
    (text_list, ocr_bbox), _ = check_ocr_box(
        image_path,
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": OCR_TEXT_THRESHOLD},
    )

    img_w, img_h = Image.open(image_path).size
    overlay_ratio = max(img_w, img_h) / 3200.0
    draw_bbox_config = {
        "text_scale": 0.8 * overlay_ratio,
        "text_thickness": max(int(2 * overlay_ratio), 1),
        "text_padding": max(int(3 * overlay_ratio), 1),
        "thickness": max(int(3 * overlay_ratio), 1),
    }

    annotated_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_source=image_path,
        model=yolo_model,
        BOX_TRESHOLD=BOX_THRESHOLD,
        output_coord_in_ratio=OUTPUT_COORD_RATIO,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text_list,
        use_local_semantics=True,
        iou_threshold=IOU_THRESHOLD,
    )

    # ---- coerce overlay to PIL (handles ndarray/torch/path/base64) ----
    try:
        annotated_pil = _to_pil_image(annotated_img)
    except Exception:
        annotated_pil = None

    # ---- (3) save overlay here (if asked) ----
    overlay_path = None
    if save_overlay and annotated_pil is not None:
        if overlay_out_path is None:
            overlay_out_path = image_path.replace(".png", "_annotated.png")
        annotated_pil.save(overlay_out_path)
        overlay_path = overlay_out_path

    return annotated_pil, label_coordinates, parsed_content_list, overlay_path


# -------- Convert OmniParser output → elements[] --------
def parse_labels_to_elements(parsed_lines, label_coords, screen_wh):
    """
    Accepts:
      - parsed_lines: list[str] like "Text Box ID 0: Compose"
                      OR list[dict] like {"id":0,"type":"Text Box","desc":"Compose",...}
      - label_coords: dict[str->list] or dict[int->list] mapping ID -> [x,y,w,h] (ratio or px)
      - screen_wh: (W,H) to convert ratio -> pixels

    Returns:
      elements: list[dict] with {id, role, text, bbox[pixels], conf, source, meta}
    """
    W, H = screen_wh
    elements = []

    def to_pixels(box):
        """box can be [x,y,w,h] either ratio (<=1) or pixels. Return pixel ints."""
        if not box or len(box) != 4:
            return None
        x, y, w, h = map(float, box)
        # ratio heuristic: all values <= 1.2 -> treat as ratio
        if max(abs(x), abs(y), abs(w), abs(h)) <= 1.2:
            x, y, w, h = x * W, y * H, w * W, h * H
        return [int(x), int(y), int(w), int(h)]

    # --- handle each line/dict ---
    for index, item in enumerate(parsed_lines):
        # Case A: dict format
        if isinstance(item, dict):
            # Try to read fields with several common aliases
            prefix = item.get("type", item.get("kind", item.get("box_type", "Icon Box")))
            text = item.get("content", item.get("desc", item.get("caption", "")))
            box = item.get("bbox", item.get("box", None))
            interactive = item.get("interactivity")
            src = item.get("source")

            px = to_pixels(box) if box is not None else None

            elements.append({
                "id": f"elem_{index}",
                "role": prefix,
                "text": (text or "").strip(),
                "bbox": px,
                "interactivity": interactive,
                "source": src
            })
            continue

        # Unknown type: skip gracefully
        continue

    return elements

# -------- Main: capture → OmniParser → elements[] --------
def run_once(save_overlay=True) -> Dict[str, Any]:
    print("Pausing 3 seconds for Screen Capture ...")
    time.sleep(3)
    img, (W, H) = capture_fullscreen_rgb()
    print("Done.")

    # OmniParser expects a path; use a temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name)
        path = tmp.name

    yolo_model, caption_model_processor = init_models()

    annotated_pil, label_coords, parsed_lines, overlay_path = run_omni(
        path, yolo_model, caption_model_processor,
        save_overlay=save_overlay
    )

    elements = parse_labels_to_elements(parsed_lines, label_coords, (W, H))

    # clean up temp screenshot; keep overlay if saved
    try:
        os.remove(path)
    except OSError:
        pass

    return {
        "window": {"bbox": [0, 0, W, H], "scale": 1.0},
        "elements": elements,
        "debug": {
            "overlay_path": overlay_path,
            "num_elements": len(elements),
            "raw_lines": parsed_lines[:10],
        },
    }

if __name__ == "__main__":
    out = run_once(save_overlay=True)
    print(json.dumps({
        "window": out["window"],
        "num_elements": out["debug"]["num_elements"],
        "overlay_path": out["debug"]["overlay_path"],
        "elements_preview": out["elements"][:45]  # avoid giant dumps
    }, indent=2, ensure_ascii=False))
