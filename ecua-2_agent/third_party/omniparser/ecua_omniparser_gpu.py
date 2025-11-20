import os, json, uuid, tempfile, base64, time
from io import BytesIO
from typing import List, Dict, Any, Tuple
from functools import lru_cache

import torch, numpy as np, cv2
from mss import mss
from PIL import Image

from util.utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
)

# config macros
YOLO_MODEL_PATH      = "weights/icon_detect/model.pt"
CAPTION_MODEL_DIR    = "weights/icon_caption_florence"
BOX_THRESHOLD        = 0.05
IOU_THRESHOLD        = 0.10
OCR_TEXT_THRESHOLD   = 0.55 
OUTPUT_COORD_RATIO   = True
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"

# helper functions for converting annoted image into png 
def _is_base64_image(s: str) -> bool:
    if not isinstance(s, str): return False
    if s.startswith("data:image/") and ";base64," in s: return True
    return len(s) > 200 and all(c.isalnum() or c in "+/=\n\r" for c in s[:400])

def _base64_to_pil(s: str) -> Image.Image:
    if s.startswith("data:image/"): s = s.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(s))).convert("RGB")

def _to_pil_image(x):
    if isinstance(x, Image.Image): return x
    import numpy as np
    if isinstance(x, np.ndarray):
        if x.ndim == 2: return Image.fromarray(x)
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
        if _is_base64_image(x): return _base64_to_pil(x)
        if os.path.exists(x):   return Image.open(x).convert("RGB")
        raise TypeError("Overlay string is neither a file path nor base64 image.")
    raise TypeError(f"Unsupported overlay type: {type(x)}")

# convert bbox into pixels
def _to_pixels(box, W, H):
    if not box or len(box) != 4: return None
    x, y, w, h = map(float, box)
    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.2:  # ratios → pixels
        x, y, w, h = x * W, y * H, w * W, h * H
    return [int(round(x)), int(round(y)), int(round(w)), int(round(h))]

# load yolo and florence captioner
@lru_cache(maxsize=1)
def _load_yolo():
    print(f"[omni] device={DEVICE} | loading YOLO…")
    return get_yolo_model(model_path=YOLO_MODEL_PATH)

@lru_cache(maxsize=1)
def _load_captioner():
    print(f"[omni] device={DEVICE} | loading captioner…")
    return get_caption_model_processor(
        model_name="florence2",
        model_name_or_path=CAPTION_MODEL_DIR,
        device=DEVICE,
    )

# wrap into a class, so you don't need to contine
class OmniEngine:
    def __init__(self):
        self.yolo = _load_yolo()
        self.captioner = _load_captioner()
        self.sct = mss()

    def capture_fullscreen_rgb(self) -> Tuple[Image.Image, Tuple[int,int]]:
        mon = self.sct.monitors[1]
        raw = self.sct.grab(mon)
        img = Image.frombytes("RGB", raw.size, raw.rgb)
        return img, (raw.size.width, raw.size.height)

    @torch.no_grad()
    def run_omni(self, image_path: str, save_overlay: bool=False, overlay_out_path: str|None=None):
        (text_list, ocr_bbox), _ = check_ocr_box(
            image_path,
            display_img=False,
            output_bb_format="xyxy",
            goal_filtering=None,
            easyocr_args={"paragraph": False, "text_threshold": OCR_TEXT_THRESHOLD},
        )
        img_w, img_h = Image.open(image_path).size
        r = max(img_w, img_h) / 3200.0
        draw_cfg = {
            "text_scale": 0.8 * r,
            "text_thickness": max(int(2 * r), 1),
            "text_padding": max(int(3 * r), 1),
            "thickness": max(int(3 * r), 1),
        }

        annotated_img, label_coords, parsed = get_som_labeled_img(
            image_source=image_path,
            model=self.yolo,
            BOX_TRESHOLD=BOX_THRESHOLD,
            output_coord_in_ratio=OUTPUT_COORD_RATIO,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_cfg,
            caption_model_processor=self.captioner,
            ocr_text=text_list,
            use_local_semantics=True,
            iou_threshold=IOU_THRESHOLD,
        )

        overlay_pil = None
        try:
            overlay_pil = _to_pil_image(annotated_img)
        except Exception:
            pass

        overlay_path = None
        if save_overlay and overlay_pil is not None:
            overlay_path = overlay_out_path or image_path.replace(".png", "_annotated.png")
            overlay_pil.save(overlay_path)

        return overlay_pil, label_coords, parsed, overlay_path

    def parse_labels_to_elements(self, parsed_lines, label_coords, screen_wh):
        W, H = screen_wh
        elements = []
        for i, item in enumerate(parsed_lines or []):
            if isinstance(item, dict):
                prefix = item.get("type")
                text = item.get("content")
                box = item.get("bbox", item.get("box"))
                px = _to_pixels(box, W, H) if box is not None else None

                elements.append({
                    "id": f"elem_{i}",
                    "role": prefix,
                    "text": text.strip(),
                    "bbox": px,
                    "source": item.get("source", "omniparser"),
                })
                
        return elements

    # external call for a parsed screen, once engine is up 
    def parse_fullscreen(self, save_overlay: bool=True) -> Dict[str, Any]:
        img, (W, H) = self.capture_fullscreen_rgb()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp.name)
            path = tmp.name

        overlay_pil, label_coords, parsed, overlay_path = self.run_omni(
            path, save_overlay=save_overlay
        )
        elements = self.parse_labels_to_elements(parsed, label_coords, (W, H))

        try: os.remove(path)
        except OSError: pass

        return {
            "window": {"bbox": [0,0,W,H], "scale": 1.0},
            "elements": elements,
            "meta": {
                "overlay_path": overlay_path,
                "num_elements": len(elements),
            }
        }

if __name__ == "__main__":
    
    engine = OmniEngine()  # model load
    print("Pausing 3s for capture…"); 
    time.sleep(3)

    # call this, no model reload
    iters = 1
    for _ in range(iters):
        out = engine.parse_fullscreen(save_overlay=True)
        print(json.dumps({
            "window": out["window"],
            "num_elements": out["meta"]["num_elements"],
            "overlay_path": out["meta"]["overlay_path"],
            "elements_preview": out["elements"][:15]
        }, indent=2, ensure_ascii=False))
