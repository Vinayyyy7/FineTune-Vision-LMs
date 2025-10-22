"""
Convert Alpaca-style SFT data (JSON or JSONL) --> UnsloTh conversation format for finetuning Vision LMs
Handles input formats:
 - JSONL (one JSON object per line)
 - JSON file containing a list of objects: [ {...}, {...}, ... ]

Output formats:
 - JSONL (default): one UnsloTh conversation JSON per line
 - JSON: single JSON array of conversation objects

Behavior:
 - If an example contains no image, the converter omits the image element (preferred).
 - If you set --force-empty-image, the converter will insert {"type":"image","image": null} (NOT recommended).
 - Normalizes several Alpaca variants: (instruction,input,output), (prompt,response), (prompt,completion), (text,response), etc.

Usage:
    python SFT_TO_VISION_SFT.py -i alpaca.json -o converted.jsonl
    python SFT_TO_VISION_SFT.py --input alpaca.json --output converted.json --output-format json
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

COMMON_DATA_KEYS = ["data", "items", "examples", "records"]

def read_input_file(path: str) -> List[Dict[str, Any]]:
    """
    Read either JSONL or JSON (list or object) and return a list of objects.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # Try to detect JSONL by extension or content heuristics
    ext = os.path.splitext(path)[1].lower()
    # If .jsonl or .ndjson -> treat as JSONL
    if ext in [".jsonl", ".ndjson"]:
        return read_jsonl(path)
    # If .json -> try json load; if top-level is list, return it; if dict, try to find common list key
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        if isinstance(j, list):
            return j
        if isinstance(j, dict):
            # try common keys
            for k in COMMON_DATA_KEYS:
                if k in j and isinstance(j[k], list):
                    return j[k]
            # maybe it's a dict of id->obj
            # convert values to list
            if all(isinstance(v, dict) for v in j.values()):
                return list(j.values())
            # fallback: wrap dict as single element
            return [j]
    # fallback: attempt to parse as JSONL by reading line by line
    try:
        return read_jsonl(path)
    except Exception as e:
        raise ValueError(f"Could not parse input file as JSON or JSONL: {e}")

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON on line {i}: {err}")
            items.append(obj)
    return items

def normalize_alpaca_item(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize different Alpaca-like formats into canonical fields:
      - instruction (string)
      - input (string)
      - output (string)
      - image (optional)   # left untouched if present
      - raw (original obj) # for provenance if needed
    """
    instruction = ""
    input_text = ""
    output = ""
    image = None

    # Instruction-like fields
    for k in ("instruction", "prompt", "text", "question", "query"):
        if k in obj and isinstance(obj[k], str) and obj[k].strip():
            instruction = obj[k].strip()
            break

    # Input-like fields
    for k in ("input", "context", "source", "examples"):
        if k in obj and isinstance(obj[k], str) and obj[k].strip():
            input_text = obj[k].strip()
            break

    # Output-like fields
    for k in ("output", "response", "completion", "answer", "reply"):
        if k in obj and isinstance(obj[k], str) and obj[k].strip():
            output = obj[k].strip()
            break

    # Some datasets embed prompt/response together in a single 'prompt' or 'instruction' that already includes expected output.
    # If output missing, try to find `label` or nested fields.
    if not output:
        # Try top-level fields that might contain the answer
        for k in ("label", "labels", "gold", "target"):
            if k in obj:
                v = obj[k]
                if isinstance(v, str) and v.strip():
                    output = v.strip()
                    break
                # if list of labels, take first string
                if isinstance(v, list) and v and isinstance(v[0], str):
                    output = v[0].strip()
                    break

    # Image handling: if present (string path/base64 or object), leave as-is
    if "image" in obj:
        image = obj.get("image")
    # some datasets use 'img' or 'image_path'
    elif "img" in obj:
        image = obj.get("img")
    elif "image_path" in obj:
        image = obj.get("image_path")

    # Final fallback: if instruction contains both instruction and input separated by newline markers,
    # attempt not to split to avoid losing user intent — keep as-is.
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "image": image,
        "raw": obj
    }

def build_messages(item: Dict[str, Any],
                   system_instruction: Optional[str] = None,
                   force_empty_image: bool = False) -> Dict[str, Any]:
    """
    Build the UnsloTh 'messages' structure for one example.
    """
    user_content: List[Dict[str, Any]] = []

    if system_instruction:
        user_content.append({"type": "text", "text": system_instruction})

    # Combine instruction + input
    text_parts = []
    if item.get("instruction"):
        text_parts.append(item["instruction"])
    if item.get("input"):
        # separate clearly
        text_parts.append("Input:\n" + item["input"])
    user_text = "\n\n".join(text_parts).strip()

    # If nothing found for instruction/input, but original object has a 'prompt' key which maybe full prompt, include that
    if not user_text:
        raw = item.get("raw", {})
        for fallback in ("prompt", "instruction", "text"):
            if fallback in raw and isinstance(raw[fallback], str) and raw[fallback].strip():
                user_text = raw[fallback].strip()
                break

    # Ensure at least an empty text entry exists
    user_content.append({"type": "text", "text": user_text})

    # Only include image element when an image reference is present, unless forced
    if item.get("image") is not None:
        user_content.append({"type": "image", "image": item.get("image")})
    else:
        if force_empty_image:
            user_content.append({"type": "image", "image": None})
        # otherwise omit image entry

    assistant_content = [{"type": "text", "text": item.get("output", "")}]

    return {"messages": [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]}

def write_output_jsonl(objs: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def write_output_json_array(objs: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(objs, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Convert Alpaca-style JSON/JSONL -> UnsloTh conversation JSON/JSONL")
    parser.add_argument("--input", "-i", required=True, help="Input JSON or JSONL file (Alpaca-style)")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--output-format", "-f", choices=["jsonl", "json"], default="jsonl",
                        help="Output format: jsonl (default) or json (single array)")
    parser.add_argument("--system-instruction", "-s", default=None,
                        help="Optional system instruction text to prepend to user content")
    parser.add_argument("--force-empty-image", action="store_true",
                        help="If set, will insert {'type':'image','image': null} when no image exists (not recommended)")
    parser.add_argument("--preview", "-p", type=int, default=0,
                        help="Print first N converted records to stdout for quick preview (0 = none)")
    args = parser.parse_args()

    items = read_input_file(args.input)
    if not items:
        print("No items found in input.", file=sys.stderr)
        sys.exit(2)

    converted = []
    for idx, raw in enumerate(items, 1):
        normalized = normalize_alpaca_item(raw)
        conv = build_messages(normalized, system_instruction=args.system_instruction, force_empty_image=args.force_empty_image)
        converted.append(conv)

    # Write output
    if args.output_format == "jsonl":
        write_output_jsonl(converted, args.output)
    else:
        write_output_json_array(converted, args.output)

    # Preview if requested
    if args.preview > 0:
        preview_n = min(len(converted), args.preview)
        print(f"\n--- Preview first {preview_n} converted examples ---")
        for i in range(preview_n):
            print(json.dumps(converted[i], ensure_ascii=False, indent=2))
            print("---")

    print(f"Converted {len(converted)} items -> {args.output} (format={args.output_format})")

if __name__ == "__main__":
    main()
