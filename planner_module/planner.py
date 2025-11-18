import os
import json
import argparse
from vllm import LLM, SamplingParams


def load_cpu_vision_json():
    """Load the cpu_vision_output.json file relative to this module."""
    base_dir = os.path.dirname(__file__)
    json_path = os.path.abspath(os.path.join(base_dir, "..", "vision_files", "cpu_vision_output.json"))
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(question: str, raw_json: dict, max_text_items: int = 120, max_total_chars: int = 4000) -> str:
    timing = raw_json.get("timing_ms", {})
    elements = raw_json.get("elements", [])
    texts = []
    for el in elements:
        txt = el.get("text", "").strip()
        if txt:
            texts.append(txt)
        if len(texts) >= max_text_items:
            break
    combined = "\n".join(texts)
    if len(combined) > max_total_chars:
        combined = combined[:max_total_chars] + "... (truncated)"
    prompt = (
        "You are an assistant analyzing condensed OCR/vision data extracted from a JSON capture. "
        "Use ONLY these text lines and timing info to answer.\n"
        f"Timing: {timing}\n"
        "Text Lines (subset):\n" + combined + "\n\n" + f"Question: {question}\nAnswer:"
    )
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Ask a question about cpu_vision_output.json")
    parser.add_argument("question", nargs="?", default="What date is shown?", help="Question about the captured screen contents")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens to generate")
    args = parser.parse_args()

    data = load_cpu_vision_json()
    prompt = build_prompt(args.question, data)

    # Configure LLM (kept small for resource constraints)
    llm = LLM(
        model="./models/llama-3.2-1b",
        dtype="float16",
        gpu_memory_utilization=0.75,
        max_model_len=1024,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    outputs = llm.generate([prompt], sampling_params)
    answer = outputs[0].outputs[0].text.strip()

    print(f"Question: {args.question}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()