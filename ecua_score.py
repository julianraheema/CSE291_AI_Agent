#!/usr/bin/env python3

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TaskMetrics:
    task_id: str
    result: float
    wes_plus: Optional[float] = None
    wes_minus: Optional[float] = None


WES_PLUS_RE = re.compile(r"^\s*WES_PLUS\s*=\s*([+-]?\d+(\.\d+)?)\s*$")
WES_MINUS_RE = re.compile(r"^\s*WES_MINUS\s*=\s*([+-]?\d+(\.\d+)?)\s*$")


def parse_result_txt(path: str, task_id: str) -> Optional[TaskMetrics]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return None

    if not lines:
        return None

    # First line is the base OSWorld metric
    try:
        base_result = float(lines[0])
    except ValueError:
        # If first line isn't numeric, treat as invalid file
        return None

    wes_plus = None
    wes_minus = None

    for line in lines[1:]:
        m_plus = WES_PLUS_RE.match(line)
        m_minus = WES_MINUS_RE.match(line)
        if m_plus:
            wes_plus = float(m_plus.group(1))
        elif m_minus:
            wes_minus = float(m_minus.group(1))

    return TaskMetrics(
        task_id=task_id,
        result=base_result,
        wes_plus=wes_plus,
        wes_minus=wes_minus,
    )


def load_tasks_from_results_dir(results_dir: str) -> List[TaskMetrics]:
    tasks: List[TaskMetrics] = []

    for entry in os.listdir(results_dir):
        task_dir = os.path.join(results_dir, entry)
        if not os.path.isdir(task_dir):
            continue

        result_path = os.path.join(task_dir, "result.txt")
        if not os.path.exists(result_path):
            continue

        metrics = parse_result_txt(result_path, task_id=entry)
        if metrics is not None:
            tasks.append(metrics)

    return tasks


def aggregate_metrics(tasks: List[TaskMetrics]) -> Tuple[float, float, float, float]:
    if not tasks:
        raise ValueError("No task metrics found.")

    n = len(tasks)

    # Success rate and avg result
    successes = 0
    result_sum = 0.0

    wes_plus_sum = 0.0
    wes_minus_sum = 0.0
    wes_plus_count = 0
    wes_minus_count = 0

    for t in tasks:
        result_sum += t.result
        if t.result >= 1.0 - 1e-6:
            successes += 1

        if t.wes_plus is not None:
            wes_plus_sum += t.wes_plus
            wes_plus_count += 1

        if t.wes_minus is not None:
            wes_minus_sum += t.wes_minus
            wes_minus_count += 1

    success_rate = successes / n
    avg_result = result_sum / n

    # avarage if WES is missing
    if wes_plus_count > 0:
        wes_plus_mean = wes_plus_sum / wes_plus_count
    else:
        wes_plus_mean = 0.0

    if wes_minus_count > 0:
        wes_minus_mean = wes_minus_sum / wes_minus_count
    else:
        wes_minus_mean = 0.0

    return success_rate, avg_result, wes_plus_mean, wes_minus_mean


def main():
    parser = argparse.ArgumentParser(description="Compute WES+ and WES-")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Root directory containing per-task result.",
    )
    
    args = parser.parse_args()

    tasks = load_tasks_from_results_dir(args.results_dir)
    if not tasks:
        raise SystemExit(f"No tasks found under {args.results_dir}")

    success_rate, avg_result, wes_plus_mean, wes_minus_mean = aggregate_metrics(tasks)

    # Convert to percentages
    sr_pct = success_rate * 100.0
    wes_plus_pct = wes_plus_mean * 100.0
    wes_minus_pct = wes_minus_mean * 100.0

    # Pretty print
    print(" OSWorld-Human metrics summary")
    print("------------------------------------")
    print(f"Number of Tasks : {len(tasks)}")
    print(f"Success Rate    : {sr_pct:.2f}%")
    print(f"Avg Result      : {avg_result:.4f}")
    print(f"WES+ (mean)     : {wes_plus_mean:.4f} ({wes_plus_pct:.2f}%)")
    print(f"WES- (mean)     : {wes_minus_mean:.4f} ({wes_minus_pct:.2f}%)")


if __name__ == "__main__":
    main()
