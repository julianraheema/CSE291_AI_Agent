#!/usr/bin/env python3

import os
import argparse
import json
import sys
import time
from pathlib import Path
from threading import Thread, Event

import cv2
import mss
import numpy as np

import subprocess, json, base64

# from ecua2_agent.vision_module.vision_CPU import ScreenParserCPU

os.environ["VLLM_PLUGINS"] = "none"
from ecua2_agent.planner_module import planner
from ecua2_agent.controller_module.controller import Controller


class ScreenRecorder:
    def __init__(self, output_path: Path, fps: int = 10, monitor_index: int = 1):
        self.output_path = Path(output_path)
        self.fps = fps
        self.monitor_index = monitor_index

        self._thread: Thread | None = None
        self._stop_event = Event()

    # start recording
    def start(self):
        self._stop_event.clear()
        self._thread = Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    # stop recording
    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None

    def _record_loop(self):
        with mss.mss() as sct:
            monitor = sct.monitors[self.monitor_index]
            width = monitor["width"]
            height = monitor["height"]

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                (width, height),
            )

            frame_interval = 1.0 / self.fps
            last_time = time.time()

            try:
                while not self._stop_event.is_set():
                    img = sct.grab(monitor)
                    frame = np.array(img)       # BGRA
                    frame = frame[:, :, :3]     # BGR
                    writer.write(frame)

                    now = time.time()
                    sleep_time = frame_interval - (now - last_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    last_time = time.time()
            finally:
                writer.release()

# run vision as subprocess because it dies runing with vllm
def run_vision_subprocess():
    cmd = [
        "python3",
        "ecua2_agent/vision_module/vision_CPU.py",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("error in vision_CPU.py failed")
        print("returncode:", result.returncode)
        print("stderr:\n", result.stderr)
        raise RuntimeError("vision_CPU.py failed")

    stdout = result.stdout.strip()
    if not stdout:
        print("error in vision_CPU.py produced no stdout")
        print("stderr:\n", result.stderr)
        raise RuntimeError("Empty stdout from vision_CPU.py")

    # First try: parse the whole stdout as JSON
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
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
            print("error could not parse JSON from vision_worker.py")
            print("stdout:\n", stdout)
            print("stderr:\n", result.stderr)
            raise

    vision_dict = data["vision"]
    b64 = data["screenshot"]
    png_bytes = base64.b64decode(b64)
    img_array = np.frombuffer(png_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return vision_dict, img_bgr

# load json from vision subprocess
def load_vision_json(vision_path):
    with open(vision_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_task_json(path: Path) -> dict | None:
    """Safely load a JSON file. Returns None if invalid."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Skipping invalid JSON {path}: {e}", file=sys.stderr)
        return None


def get_json_file(path: Path) -> str | None:
    """Load one JSON task file and print its instruction."""
    task = load_task_json(path)
    if task is None:
        return None

    instruction = task.get("instruction")
    if instruction is None:
        print(f"[WARN] No 'instruction' field in {path}")
        return None

    print(f"\n=== TASK FILE: {path} ===")
    print(f"Instruction: {instruction}")
    return instruction


def run_all(dir_path: Path, record: bool = False):
    """Recursively walk a directory and process all .json files."""
    dir_path = dir_path.resolve()

    for path in sorted(dir_path.rglob("*.json")):
        instruc = get_json_file(path)
        if instruc is None:
            continue

        # Derive domain + task_id from path: <...>/<domain>/<task_id>.json
        domain = path.parent.name
        task_id = path.stem

        # Results directory: results/<domain>/<task_id>
        result_dir = Path("results") / domain / task_id
        result_dir.mkdir(parents=True, exist_ok=True)

        # --- Start video recording (saved in same dir as vision/screenshot) ---
        recorder = None
        if record:
            video_path = result_dir / "recording.mp4"
            print(f"[INFO] Starting recording: {video_path}")
            recorder = ScreenRecorder(video_path, fps=10)
            recorder.start()

        try:

            vision_dict, img_bgr = run_vision_subprocess()

            # Save screenshot
            screenshot_path = result_dir / "screenshot.png"
            cv2.imwrite(str(screenshot_path), img_bgr)
            print(f"[INFO] Saved screenshot to {screenshot_path}")

            # Save vision JSON
            vision_json_path = result_dir / "vision.json"
            with vision_json_path.open("w", encoding="utf-8") as f:
                json.dump(vision_dict, f, indent=2)
            print(f"[INFO] Saved vision JSON to {vision_json_path}")


            output, actions = planner.generate_plan(
            task=instruc,
            vision_data=vision_dict,
            model_path="ecua2_agent/planner_module/models/llama-3.2-3B-Instruct",
            bbox=[0, 0, 1920, 1080],
            temperature=0.3,
            max_tokens=512,
        )
            print("action is:", actions)
            if recorder:
                time.sleep(3)

            if len(actions) >1:
                cont = Controller()
                for action in actions:
                    # status = cont.execute_action(action)
                    # if status in ("FAIL", "DONE"):
                    #     print(status)
                    print(action)


            

        finally:
            if recorder:
                print("[INFO] Stopping recording")
                recorder.stop()


def run_single_file(root: Path, record: bool = False):
    """Handle the case where --path is a single JSON task file."""
    instruc = get_json_file(root)
    if instruc is None:
        return

    domain = root.parent.name
    task_id = root.stem
    result_dir = Path("results") / domain / task_id
    result_dir.mkdir(parents=True, exist_ok=True)

    recorder = None
    if record:
        video_path = result_dir / "recording.mp4"
        print(f"[INFO] Starting recording: {video_path}")
        recorder = ScreenRecorder(video_path, fps=10)
        recorder.start()

    try:
        vision_dict, img_bgr = run_vision_subprocess()

        # Save screenshot
        screenshot_path = result_dir / "screenshot.png"
        cv2.imwrite(str(screenshot_path), img_bgr)
        print(f"[INFO] Saved screenshot to {screenshot_path}")

        # Save vision JSON
        vision_json_path = result_dir / "vision.json"
        with vision_json_path.open("w", encoding="utf-8") as f:
            json.dump(vision_dict, f, indent=2)
        print(f"[INFO] Saved vision JSON to {vision_json_path}")

        output, actions = planner.generate_plan(
        task=instruc,
        vision_data=vision_dict,
        model_path="ecua2_agent/planner_module/models/llama-3.2-1b",
        bbox=[0, 0, 1920, 1080],
        temperature=0.3,
        max_tokens=512,
    )
        print("action is:", actions)

        if recorder:
            time.sleep(3)
    finally:
        if recorder:
            print("[INFO] Stopping recording")
            recorder.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Read OSWorld JSON task(s), print instruction(s), "
                    "capture vision, and optionally record screen."
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=True,
        help="Path to a JSON file or directory containing JSON tasks.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="If set, record an .mp4 per task in the same directory as vision.json.",
    )
    args = parser.parse_args()

    root = Path(args.path)

    if not root.exists():
        print(f"[ERROR] Path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    # Case 1: Single JSON file
    if root.is_file() and root.suffix == ".json":
        run_single_file(root, record=args.record)
        return

    # Case 2: Directory (recursive)
    if root.is_dir():
        run_all(root, record=args.record)
        return

    print(
        f"[ERROR] Path must be a .json file or a directory: {root}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
