import uuid
import time
import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Tuple

import numpy as np
import cv2
from mss import mss
from paddleocr import PaddleOCR
from ultralytics import YOLO

# class for storing detected elements
@dataclass
class Element:
    id: str
    role: str           # "button", "textbox", "icon", "text", etc.
    text: str           # label or "" for pure icons
    bbox: List[int]     # [x, y, w, h] in screen (pixel) coords
    conf: float
    source: str         # "ocr" or "detector" (yolo)
    meta: Dict[str, Any] = field(default_factory=dict)

# Parser Class
class ScreenParserCPU:
    def __init__(
        self,
        ocr_lang: str = "en",
        ocr_conf_thresh: float = 0.55,
        ocr_box_thresh: float = 0.60,
        max_side_for_ocr: int = 1600,
        yolo_weights: str = "yolov8n.pt",
        yolo_conf_thresh: float = 0.25,
        yolo_imgsz: int = 640,
    ):
        # Capture
        self.sct = mss()

        # ocr init
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            lang=ocr_lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            text_recognition_batch_size=64,   
            cpu_threads=14,
            text_det_box_thresh=0.65,
            text_det_unclip_ratio=1.6,
            det_db_thresh=0.30
        )
        self.ocr_conf_thresh = ocr_conf_thresh
        self.ocr_box_thresh = ocr_box_thresh
        self.max_side_for_ocr = max_side_for_ocr

        # YOLO init
        self.det_model = YOLO(yolo_weights)
        self.yolo_class_roles = self._init_yolo_class_roles_from_names(self.det_model.names)
        self.yolo_conf_thresh = yolo_conf_thresh
        self.yolo_imgsz = yolo_imgsz

    # capture screen util function
    def capture_fullscreen_bgr(self) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        monitor = self.sct.monitors[1]  # 0 = virtual, 1 = primary
        raw = np.array(self.sct.grab(monitor))  # BGRA
        img = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        bbox = (monitor["left"], monitor["top"], monitor["width"], monitor["height"])
        return img, bbox

    # ocr code
    def _resize_for_ocr(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > self.max_side_for_ocr:
            scale = self.max_side_for_ocr / float(max(h, w))
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        return img, scale

    # runs ocr pipeline on screen capture
    def run_ocr(self, img_bgr: np.ndarray) -> List[Element]:
        img_resized, s = self._resize_for_ocr(img_bgr)

        # prefer predict(), falls back to ocr()
        try:
            result = self.ocr.predict(img_resized)
        except AttributeError:
            result = self.ocr.ocr(img_resized)

        elements: List[Element] = []
        if not result:
            return elements

        first = result[0]

        # ocr v3 style
        if isinstance(first, dict) and ("rec_texts" in first or "rec_boxes" in first):
            rec_texts  = first.get("rec_texts")
            rec_scores = first.get("rec_scores")
            rec_boxes  = first.get("rec_boxes")

            # none -> empty
            rec_texts  = [] if rec_texts  is None else rec_texts
            rec_scores = [] if rec_scores is None else rec_scores
            # rec_boxes can be a numpy array shape (N,4) or a python list
            if rec_boxes is None:
                rec_boxes = []
            elif hasattr(rec_boxes, "shape"):  # numpy array
                rec_boxes = rec_boxes.tolist()

            # zip safely to the shortest length
            n = min(len(rec_texts), len(rec_scores), len(rec_boxes))
            for i in range(n):
                text = rec_texts[i] or ""
                conf = float(rec_scores[i]) if rec_scores[i] is not None else 0.0
                if conf < self.ocr_conf_thresh:
                    continue

                x1, y1, x2, y2 = [float(v) for v in rec_boxes[i]]
                # map back to original scale
                x1, y1, x2, y2 = x1 / s, y1 / s, x2 / s, y2 / s
                w, h = x2 - x1, y2 - y1

                elements.append(
                    Element(
                        id=f"t_{uuid.uuid4().hex[:6]}",
                        role=self._guess_role_from_text(text),
                        text=text.strip(),
                        bbox=[int(x1), int(y1), int(w), int(h)],
                        conf=conf,
                        source="ocr",
                    )
                )
            return elements

        # ocr v2 usage
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple)):
            for bbox, (text, conf) in first:
                if conf is None or float(conf) < self.ocr_conf_thresh:
                    continue
                xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
                x1, y1 = min(xs)/s, min(ys)/s
                x2, y2 = max(xs)/s, max(ys)/s
                w, h = x2 - x1, y2 - y1
                elements.append(
                    Element(
                        id=f"t_{uuid.uuid4().hex[:6]}",
                        role=self._guess_role_from_text(text or ""),
                        text=(text or "").strip(),
                        bbox=[int(x1), int(y1), int(w), int(h)],
                        conf=float(conf),
                        source="ocr",
                    )
                )
            return elements

        print("[OCR] Unrecognized output format:", type(first))
        return elements

    def _guess_role_from_text(self, text: str) -> str:
        t = (text or "").lower()
        if t in {"ok", "cancel", "close", "save", "send", "submit", "apply", "next", "back"}:
            return "button"
        if "search" in t or "find" in t:
            return "textbox"
        if any(k in t for k in ("email", "password", "username", "phone")):
            return "textbox"
        return "text"

    # yolo detection fields
    def _init_yolo_class_roles_from_names(self, names: Dict[int, str]) -> Dict[int, Tuple[str, str]]:
        mapping = {}
        for cid, cname in names.items():
            lower = cname.lower()
            if "button" in lower:
                role, label = "button", cname
            elif any(k in lower for k in ("text", "input", "field", "box")):
                role, label = "textbox", cname
            elif any(k in lower for k in ("close", "x", "cross")):
                role, label = "icon", "close"
            elif any(k in lower for k in ("gear", "settings")):
                role, label = "icon", "settings"
            elif any(k in lower for k in ("search", "magnifier")):
                role, label = "icon", "search"
            else:
                role, label = "icon", cname
            mapping[cid] = (role, label)
        return mapping

    # run yolo model on screen for icon detection
    def run_yolo(self, img_bgr: np.ndarray) -> List[Element]:

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res_list = self.det_model.predict(
            source=img_rgb,
            conf=self.yolo_conf_thresh,
            imgsz=self.yolo_imgsz,
            verbose=False,
            device="cpu"
        )
        if not res_list:
            return []
        r = res_list[0]
        elements: List[Element] = []
        if r.boxes is None or len(r.boxes) == 0:
            return elements

        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()  # [x1,y1,x2,y2]
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = xyxy
            w, h = x2 - x1, y2 - y1
            role, name = self.yolo_class_roles.get(cls_id, ("icon", f"class_{cls_id}"))
            elements.append(Element(
                id=f"d_{uuid.uuid4().hex[:6]}",
                role=role,
                text=name,
                bbox=[int(x1), int(y1), int(w), int(h)],
                conf=conf,
                source="detector",
                meta={"class_id": cls_id}
            ))
        return elements

    # function call for ECUA pipeline usage
    def parse_fullscreen(self) -> Dict[str, Any]:
        img_bgr, bbox = self.capture_fullscreen_bgr()
        t0 = time.time()
        ocr_elems = self.run_ocr(img_bgr)
        t1 = time.time()
        det_elems = self.run_yolo(img_bgr)
        t2 = time.time()

        elements = [asdict(e) for e in ocr_elems + det_elems]
        return {
            "window": {"bbox": [bbox[0], bbox[1], bbox[2], bbox[3]], "scale": 1.0},
            "elements": elements,
            "timing_ms": {
                "ocr": int((t1 - t0) * 1000),
                "yolo": int((t2 - t1) * 1000),
                "total": int((t2 - t0) * 1000),
            }
        }
    
    # function call for OSWorld pipeline usage
    def parse_obs(self, img_bgr, bbox) -> Dict[str, Any]:
        t0 = time.time()
        ocr_elems = self.run_ocr(img_bgr)
        t1 = time.time()
        det_elems = self.run_yolo(img_bgr)
        t2 = time.time()

        elements = [asdict(e) for e in ocr_elems + det_elems]
        return {
            "window": {"bbox": [bbox[0], bbox[1], bbox[2], bbox[3]], "scale": 1.0},
            "elements": elements,
            "timing_ms": {
                "ocr": int((t1 - t0) * 1000),
                "yolo": int((t2 - t1) * 1000),
                "total": int((t2 - t0) * 1000),
            }
        }

# small test code to run standalone
if __name__ == "__main__":
    parser = ScreenParserCPU(
        ocr_lang="en",
        ocr_conf_thresh=0.65,
        ocr_box_thresh=0.6,
        max_side_for_ocr=1600,
        yolo_weights="yolov8n.pt",   # swap yolov8s.pt if you want stronger model
        yolo_conf_thresh=0.25,
        yolo_imgsz=640,
    )

    print("Capturing Screen....")
    time.sleep(3) # sleep 3 seconds so you can move to whatever screen you want
    
    result = parser.parse_fullscreen()

    # print for debugging
    print(json.dumps({
        "window": result["window"],
        "timing_ms": result["timing_ms"],
        "elements": result["elements"]
    }, indent=2, ensure_ascii=False))
    
    file = "cpu_vision_output.json"
    with open(file, "w") as json_file:
        json.dump(result, json_file, indent=3)
