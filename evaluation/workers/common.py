"""
Shared utilities and globals for all inference backend workers.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from decord import VideoReader, cpu
from jinja2 import Template
from PIL import Image
from qwen_vl_utils import fetch_image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parents[1]
DEBUG = os.environ['DEBUG'] if 'DEBUG' in os.environ else False
USE_RAY = not DEBUG


def conditional_decorator(condition, decorator):
    """Conditionally apply a decorator; used to toggle Ray remote on/off."""
    def wrapper(func):
        if condition:
            return decorator(func)
        return func
    return wrapper


def clean_text(text, exclude_chars=["\n", "\r"]):
    """
    Clean model output text and extract the final answer.

    Processing order:
    1. Extract content inside <answer>...</answer> tags
    2. Extract content inside \\boxed{...}
    3. Handle text leaked after </answer>
    4. Remove newlines and special characters
    5. Detect multiple-choice option format (e.g. "A." "B)" "C ")
    6. Lowercase and strip trailing period
    """
    answer_matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_matches:
        text = answer_matches[-1]

    box_matches = re.findall(r"\\boxed{(.*?)}", text, re.DOTALL)
    if box_matches:
        text = box_matches[-1]

    answer_leaked_matches = re.findall(r'</answer>\s*(.+)$', text, re.DOTALL)
    if answer_leaked_matches:
        text = answer_leaked_matches[-1]

    for char in exclude_chars:
        if char in ["\n", "\r"]:
            text = re.sub(r"(?<=\s)" + re.escape(char), "", text)
            text = re.sub(r"(?<!\s)" + re.escape(char), " ", text)
        else:
            text = text.replace(char, " ")

    processed_text = text.strip()
    mc_option_match = re.match(r"^([a-zA-Z])[\.\)\s]", processed_text)
    if mc_option_match:
        text = mc_option_match.group(1)
    else:
        text = processed_text

    if "<|im_end|>" in text:
        text = text.replace("<|im_end|>", "")

    return text.strip().rstrip(".").lower()


def extract_frames(video_path: str, num_frames: int):
    """Uniformly sample num_frames frames from a video, returning a list of PIL Images."""
    extracted_frames = []
    vr = VideoReader(video_path, ctx=cpu(0))
    frame_count = len(vr)
    step = frame_count // num_frames
    indices = [i * step for i in range(num_frames)]
    for idx in indices:
        try:
            frame = vr[idx].asnumpy()
            pil_image = Image.fromarray(frame)
            extracted_frames.append(pil_image)
        except Exception as e:
            print(e)
            print(f"[WARNING] Unable to extract frame {idx} from video {video_path}")
    return extracted_frames


def resolve_media_paths(data, data_path):
    """Resolve relative media paths in JSONL rows to absolute paths (relative to the JSONL file directory)."""
    data_dir = str(Path(data_path).resolve().parent)
    for row in data:
        for key in ["video", "image", "images"]:
            if key not in row:
                continue
            if isinstance(row[key], str) and not os.path.isabs(row[key]):
                row[key] = os.path.join(data_dir, row[key])
            elif isinstance(row[key], list):
                row[key] = [
                    os.path.join(data_dir, p) if not os.path.isabs(p) else p
                    for p in row[key]
                ]
    return data


def build_message(row_dict, args):
    """
    Build a multimodal chat message from a single data row.

    Assembles the benchmark-specific prompt, format_prompt Jinja template,
    and image/video inputs into an OpenAI-style message list.
    """
    value = args.build_prompt(row_dict)

    if args.format_prompt:
        with open(args.format_prompt, encoding="utf-8") as f:
            format_prompt_content = f.read()
        format_prompt_template = Template(format_prompt_content.strip())
        value = format_prompt_template.render(content=value)

    if "image" in row_dict:
        image_list = row_dict["image"]
    elif "images" in row_dict:
        image_list = row_dict["images"]
    elif "video" in row_dict:
        image_list = extract_frames(row_dict["video"], args.num_frames)
        value = value.replace("<video>", "<image>" * len(image_list))
        if len(image_list) != value.count("<image>"):
            print(f"[WARNING] Row: {row_dict}\n Number of images ({len(image_list)}) does not match number of placeholders ({value.count('<image>')}) in value for video {row_dict['video']},")
    else:
        image_list = []

    assert len(image_list) == value.count("<image>")

    if isinstance(image_list, str):
        image_list = [image_list]

    content = []
    chunks = value.split('<image>')
    cur_img_num = 0
    for chunk in chunks:
        if chunk.strip():
            content.append({"type": "text", "text": chunk.strip()})
        if cur_img_num < len(image_list):
            img = image_list[cur_img_num]
            img_dict = {"type": "image", "image": img, "max_pixels": args.max_pixels, "min_pixels": args.min_pixels}
            img_pil = fetch_image(img_dict)
            img_dict["image"] = img_pil
            content.append(img_dict)
            cur_img_num += 1

    message = [{"role": "user", "content": content}]
    return message


def build_messages(data, args):
    """Build messages for all data rows in parallel using a thread pool."""
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(build_message, row_dict, args) for row_dict in data]
        messages = []
        for future in tqdm(futures, desc="Building messages", position=1, disable=True):
            messages.append(future.result())
    return messages


def get_unprocessed_data(args):
    """
    Load JSONL data and skip already-processed samples (resume support).

    Compares IDs in the existing output_path to return only unprocessed rows.
    Also resolves relative media paths in JSONL to absolute paths.
    """
    data = []
    processed_ids = set()

    if os.path.exists(args.output_path) and not args.overwrite:
        print(args.output_path)
        with open(args.output_path, "r") as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                processed_ids.add(item.get("id", None))

    with open(args.data_path, "r") as f:
        for item in f:
            row = json.loads(item)
            if row.get("id", None) not in processed_ids:
                data.append(row)

    data = resolve_media_paths(data, args.data_path)
    print(f"Loaded {len(data)} unprocessed samples...")
    return data
