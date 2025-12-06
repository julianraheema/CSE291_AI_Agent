#!/usr/bin/env python3
import os
import json
import time
from typing import List, Dict, Any

import numpy as np
import torch
from mss import mss
from PIL import Image
from doctr.models import ocr_predictor
from ultralytics import YOLO

import argparse
import sys
import base64
import cv2

current_directory = os.getcwd()

# Print the current working directory
print("Current Working Directory:", current_directory)

# configs
OMNI_YOLO_WEIGHTS = f'{current_directory}/CSE291_AI_Agent/ecua2_agent/vision_module/vision_GPU/weights/icon_detect/model.pt'

print("the full model dir is: ", OMNI_YOLO_WEIGHTS)

# Seconds to wait before capturing
COUNTDOWN_SECONDS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# docTR model
ocr = ocr_predictor(
    det_arch="db_resnet50",
    reco_arch="parseq",
    pretrained=True,
    resolve_lines=True,
    resolve_blocks=True,
    assume_straight_pages=True,
    preserve_aspect_ratio=True,
    export_as_straight_boxes=True,
    symmetric_pad=True,
    det_bs=1,
    reco_bs=128,
).to(device)

# omniparser yolo weights
ui_model = YOLO(OMNI_YOLO_WEIGHTS)

# screen capture weights
def grab_fullscreen_pil() -> Image.Image:
    
    with mss() as sct:
        mon = sct.monitors[1]  # 0 = all, 1 = primary
        shot = sct.grab(mon)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        return img


# geometry for attaching icons and weights
def iou_xyxy(box1, box2) -> float:
    
    x0 = max(box1[0], box2[0])
    y0 = max(box1[1], box2[1])
    x1 = min(box1[2], box2[2])
    y1 = min(box1[3], box2[3])

    inter_w = max(0, x1 - x0)
    inter_h = max(0, y1 - y0)
    inter = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def attach_nearest_text(
    ui_bbox,
    text_elements: List[Dict[str, Any]],
    iou_thresh: float = 0.05,
    max_dist: float = 100.0,
):
    best_text = ""
    best_score = 0.0

    ux0, uy0, ux1, uy1 = ui_bbox
    ucx = 0.5 * (ux0 + ux1)
    ucy = 0.5 * (uy0 + uy1)

    # try iou
    for te in text_elements:
        tx0, ty0, tx1, ty1 = te["bbox"]
        i = iou_xyxy(ui_bbox, [tx0, ty0, tx1, ty1])
        if i >= iou_thresh and i > best_score:
            best_score = i
            best_text = te["text"]

    if best_text:
        return best_text, best_score

    # nearest center
    max_dist2 = max_dist * max_dist
    best_dist2 = max_dist2
    for te in text_elements:
        tx0, ty0, tx1, ty1 = te["bbox"]
        tcx = 0.5 * (tx0 + tx1)
        tcy = 0.5 * (ty0 + ty1)
        dx = tcx - ucx
        dy = tcy - ucy
        d2 = dx * dx + dy * dy
        if d2 < best_dist2:
            best_dist2 = d2
            best_text = te["text"]

    return best_text, best_score


# main pipeline
def parse_obs(img=None) -> List[Dict[str, Any]]:
    
    # no observation passed
    if img is None:
        print(f"Capturing screen in {COUNTDOWN_SECONDS} seconds...")
        time.sleep(COUNTDOWN_SECONDS)
        img = grab_fullscreen_pil()
        print("Captured.")

    np_img = np.array(img)

    # docTR OCR
    with torch.inference_mode():
        result = ocr([np_img])

    exported = result.export()
    elements: List[Dict[str, Any]] = []

    for page in exported.get("pages", []):
        page_h, page_w = page.get("dimensions", (1, 1))

        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                words = line.get("words", [])
                line_text = " ".join(w["value"] for w in words).strip()
                if not line_text:
                    continue

                (x0n, y0n), (x1n, y1n) = line["geometry"]

                x0 = int(x0n * page_w)
                y0 = int(y0n * page_h)
                x1 = int(x1n * page_w)
                y1 = int(y1n * page_h)

                if words:
                    confidences = [w.get("confidence", 0.0) or 0.0 for w in words]
                    line_conf = float(sum(confidences) / len(confidences))
                else:
                    line_conf = None

                elements.append(
                    {
                        "type": "text",
                        "text": line_text,
                        "confidence": line_conf,
                        "bbox": [x0, y0, x1, y1],
                    }
                )

    # keep a separate list of text elements for UI attachment
    text_elements = [e for e in elements if e["type"] == "text"]

    # omniparser yolo model
    ui_results = ui_model(np_img, verbose=False)

    if ui_results:
        r = ui_results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()   
            cls_ids = r.boxes.cls.cpu().numpy()  
            scores = r.boxes.conf.cpu().numpy()  

            class_names = ui_model.names

            ui_elements: List[Dict[str, Any]] = []

            for bbox, cls_id, score in zip(boxes, cls_ids, scores):
                x0, y0, x1, y1 = [int(v) for v in bbox]
                cls_id = int(cls_id)
                label = class_names.get(cls_id, f"class_{cls_id}")

                # attach nearest text line if any
                attached_text, match_score = attach_nearest_text(
                    [x0, y0, x1, y1],
                    text_elements,
                    iou_thresh=0.05,
                    max_dist=120.0,
                )
                if attached_text != "":
                    ui_elements.append(
                        {
                            "type": "ui",
                            "ui_class": label,
                            "text": attached_text,
                            "det_confidence": float(score),
                            "text_match_score": float(match_score),
                            "bbox": [x0, y0, x1, y1],
                        }
                    )

            elements.extend(ui_elements)

    return {"window dims":(page_w, page_h), "elements": elements}


# cli
if __name__ == "__main__":
    # elems = parse_obs()
    # print(json.dumps(elems, indent=2, ensure_ascii=False))
    ##################################

    cli = argparse.ArgumentParser(
        description="Run ScreenParser GPU on either fullscreen or a provided image"
    )
    cli.add_argument(
        "--img",
        type=str,
        help="Path to an existing screenshot image. If not provided, capture fullscreen.",
    )

    args = cli.parse_args()

    # --- Mode 1: use provided image + bbox (subprocess mode) ---
    if args.img is not None:
        img_bgr = cv2.imread(args.img)
        if img_bgr is None:
            print(f"ERROR: Failed to load image: {args.img}", file=sys.stderr)
            sys.exit(1)

        # parse_obs returns a dict
        result_dict = parse_obs(img_bgr)

    # --- Mode 2: fallback to fullscreen capture for manual testing ---
    else:
        print("Capturing Screen....")
        time.sleep(3)  # sleep 3 seconds so you can move to whatever screen you want

        # parse_fullscreen returns (dict, img_bgr)
        result_dict = parse_obs()

    payload = {
        "vision": result_dict,
    }

    # Print JSON as a single line so the parent process can json.loads(stdout)
    print(json.dumps(payload, ensure_ascii=False))

