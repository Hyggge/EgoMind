"""
EgoMind Evaluation Script

Supports multiple inference backends (HuggingFace / vLLM / LMDeploy / OpenAI API)
with Ray-based multi-GPU parallel inference and automatic benchmark metric computation.

Usage:
    python run_eval.py \\
        --model_path models/EgoMind-7B \\
        --output_path outputs/vsibench.jsonl \\
        --benchmark vsibench

Supported benchmarks: vsibench, sparbench, sitebench, spbench
"""

import json
import torch
import ray
from argparse import ArgumentParser
from qwen_vl_utils import process_vision_info, fetch_image
from transformers import set_seed, AutoModel, AutoProcessor, AutoTokenizer, AutoModelForCausalLM, GenerationConfig, Qwen2_5_VLForConditionalGeneration
from decord import VideoReader, cpu
from typing import List
from jinja2 import Template
import openai
import base64
from pathlib import Path
import requests
from PIL import Image
import io
import re
import os
from urllib.parse import urlparse, parse_qs, urlunparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time

# Directory of this script; used to resolve relative paths for datasets and format_prompt
SCRIPT_DIR = Path(__file__).resolve().parent

# Set DEBUG=1 to skip Ray and run single-GPU debug mode
DEBUG = os.environ['DEBUG'] if 'DEBUG' in os.environ else False
USE_RAY = not DEBUG
print(f"DEBUG: {DEBUG}, USE_RAY: {USE_RAY}")


# ==================== Utilities ==================== #

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
    # Extract <answer> tag content (use the last match)
    answer_matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_matches:
        text = answer_matches[-1]
    
    # Extract \boxed{} content
    box_matches = re.findall(r"\\boxed{(.*?)}", text, re.DOTALL)
    if box_matches:
        text = box_matches[-1]

    # Handle text leaked after </answer> tag
    answer_leaked_matches = re.findall(r'</answer>\s*(.+)$', text, re.DOTALL)
    if answer_leaked_matches:
        text = answer_leaked_matches[-1]

    # Remove special characters
    for char in exclude_chars:
        if char in ["\n", "\r"]:
            text = re.sub(r"(?<=\s)" + re.escape(char), "", text)
            text = re.sub(r"(?<!\s)" + re.escape(char), " ", text)
        else:
            text = text.replace(char, " ")
    
    # Detect multiple-choice option (e.g. "A." "B)" "C ")
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
    # Generate prompt text with benchmark-specific function
    value = args.build_prompt(row_dict)

    # Wrap with Jinja template for CoT format instructions
    if args.format_prompt:
        with open(args.format_prompt, encoding="utf-8") as f:
            format_prompt_content = f.read()
        format_prompt_template = Template(format_prompt_content.strip())
        value = format_prompt_template.render(content=value)

    # Extract visual inputs (image list or sampled video frames)
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

    # Split text by <image> placeholders and interleave images
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

    # Collect IDs of already-processed samples from existing output
    if os.path.exists(args.output_path) and not args.overwrite:
        print(args.output_path)
        with open(args.output_path, "r") as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                processed_ids.add(item.get("id", None))

    # Keep only unprocessed samples
    with open(args.data_path, "r") as f:
        for item in f:
            row = json.loads(item)
            if row.get("id", None) not in processed_ids:
                data.append(row)

    # Resolve relative media paths to absolute
    data = resolve_media_paths(data, args.data_path)
    print(f"Loaded {len(data)} unprocessed samples...")
    return data


# ==================== Inference Backend: HuggingFace ==================== #

@conditional_decorator(USE_RAY, ray.remote(num_gpus=1))
class HFRayWorker:
    """
    HuggingFace native inference worker.

    Supported models: Qwen2.5-VL, Qwen3-VL, MIMO-VL, Ovis2.5, Keye-VL,
                      GLM-4.1V/4.5V, Kimi-VL, MiniCPM-V-4.5
    Each worker occupies one GPU, dispatched via Ray.
    """
    def __init__(self, model_path, args):
        self.args = args
        set_seed(42)
        self.model = None
        self.processor = None
        self.tokenizer = None

        # Load the corresponding model and processor based on model_type
        if self.args.model_type in ["Qwen2.5-VL", "MIMO-VL"]:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.args.model_path, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True,
                device_map="cuda"
            )
            self.processor = AutoProcessor.from_pretrained(self.args.model_path, use_fast=True, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer
        elif self.args.model_type in ["Qwen3-VL"]:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.args.model_path, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True,
                device_map="cuda"
            )
            self.processor = AutoProcessor.from_pretrained(self.args.model_path, use_fast=True, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer

        elif self.args.model_type == "Ovis2.5":
            from transformers import AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_path, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True,
                device_map="cuda"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, use_fast=True, trust_remote_code=True)
        
        elif self.args.model_type == "Keye-VL":
            from transformers import AutoProcessor
            self.model = AutoModel.from_pretrained(
                self.args.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="cuda",
                attn_implementation="flash_attention_2",
            )
            self.processor = AutoProcessor.from_pretrained(self.args.model_path, use_fast=True, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer

        elif self.args.model_type in ["GLM-4.1V", "GLM-4.5V"]:
            from transformers import AutoProcessor, Glm4vForConditionalGeneration
            self.model = Glm4vForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=self.args.model_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            )
            self.processor = AutoProcessor.from_pretrained(self.args.model_path, use_fast=True)
            self.tokenizer = self.processor.tokenizer
        
        elif self.args.model_type == "Kimi-VL":
            from transformers import AutoProcessor
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_path, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                device_map="cuda"
            )
            self.processor = AutoProcessor.from_pretrained(self.args.model_path, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer

        elif self.args.model_type == "MiniCPM-V-4.5":
            from transformers import AutoTokenizer
            self.model = AutoModel.from_pretrained(
                self.args.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="cuda",
                attn_implementation="flash_attention_2",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True)

        else:
            raise NotImplementedError(f"Model type {self.args.model_type} not supported in HFRayWorker")
            

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.temperature > 0,
            num_return_sequences=args.rollout,
        )
        print(f"HF Worker initialized with model from {model_path}")
    
    def build_hf_input(self, msg, row_dict, args):
        """Convert the unified message format into model-specific input tensors."""
        if self.args.model_type in ["Qwen2.5-VL", "MIMO-VL", "Qwen2-VL"]:
            prompt_text = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(msg)
            inputs = self.processor(
                text=[prompt_text],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)
            return inputs, prompt_text
        elif self.args.model_type in ["Qwen3-VL"]:
            prompt_text = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inputs = self.processor.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            return inputs, prompt_text

        elif self.args.model_type == "Ovis2.5":
            prompt_text = self.tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.args.enable_thinking
            )
            input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
                messages=msg,
                add_generation_prompt=True,
                min_pixels=self.args.min_pixels,
                max_pixels=self.args.max_pixels,
                enable_thinking=self.args.enable_thinking
            )
            inputs = {
                "inputs": input_ids.to(self.model.device),
                "pixel_values": pixel_values.to(self.model.device) if pixel_values is not None else None,
                "grid_thws": grid_thws.to(self.model.device) if grid_thws is not None else None,
                "enable_thinking": self.args.enable_thinking,
            }
            return inputs, prompt_text
        
        elif self.args.model_type == "Keye-VL":
            from keye_vl_utils import process_vision_info as  keye_process_vision_info

            if msg[0]["content"][-1]["type"] != "text":
                    msg[0]["content"].append({"type": "text", "text": ""})
            if self.args.enable_thinking:
                msg[0]["content"][-1]["text"] += "/think"
            else:
                msg[0]["content"][-1]["text"] += "/no_think"
            
            prompt_text = self.processor.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True
            )
            image_inputs, video_inputs = keye_process_vision_info(msg)
            inputs = self.processor(
                text=[prompt_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)
            return inputs, prompt_text

        elif self.args.model_type == "GLM-4.1V":
            prompt_text = self.processor.tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.processor.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)
            return inputs, prompt_text
        
        elif self.args.model_type == "Kimi-VL":
            prompt_text = self.processor.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.args.enable_thinking
            )
            image_inputs, _ = process_vision_info(msg)
            inputs = self.processor(
                text=prompt_text,
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)
            return inputs, image_inputs
        
        elif self.args.model_type == "MiniCPM-V-4.5":
            prompt_text = ""
            content_new = []
            for item in msg[0]["content"]:
                if item["type"] == "text":
                    content_new.append(item["text"])
                    prompt_text += item["text"]
                elif item["type"] == "image":
                    content_new.append(item["image"])
                    prompt_text += "<image>"
            msg[0]["content"] = content_new
            return msg, prompt_text            
        else:
            raise NotImplementedError(f"Model type {self.args.model_type} not supported in HFRayWorker")

    def post_process(self, text):
        return text

    def inference(self, data_chunk):
        """Run inference on a data chunk sample-by-sample, compute rewards, and return results."""
        results = []
        
        try:
            messages = build_messages(data_chunk, self.args)
            
            for row_dict, msg in tqdm(list(zip(data_chunk, messages)), desc="Inference batched messages", position=1):
                inputs, prompt_text = self.build_hf_input(msg, row_dict, self.args)
                
                # MiniCPM uses its built-in chat interface
                if self.args.model_type == "MiniCPM-V-4.5":
                    answer = self.model.chat(
                        msgs=inputs,
                        tokenizer=self.tokenizer,
                        enable_thinking=self.args.enable_thinking,
                        sampling=False,
                        stream=False
                    )
                    output_texts = [answer]
                
                else:
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            generation_config=self.generation_config,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                    
                    # Strip prompt tokens, keep only generated tokens
                    if self.args.model_type != "Ovis2.5":
                        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]

                    output_texts = self.tokenizer.batch_decode(
                        generated_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False
                    )
                
                output_texts = [self.post_process(text) for text in output_texts]

                # When rollout > 1, return multiple generation results
                if self.args.rollout == 1:
                    output_text = output_texts[0]
                else:
                    output_text = output_texts
                
                # Compute reward (accuracy / MRA / benchmark-specific metric)
                cleaned_gt_answer = clean_text(row_dict["answer"])
                output_text_list = [output_text] if type(output_text) != list else output_text
                response_list, cleaned_response_list, reward_list, correct_list = [], [], [], []
                
                for item in output_text_list:
                    cleaned_response = clean_text(item)
                    try:
                        reward = self.args.calculate_reward(cleaned_gt_answer, cleaned_response, row_dict)
                    except:
                        print(f"Error calculating reward: {cleaned_gt_answer}, {cleaned_response}, {row_dict}")
                        reward = 0
                    
                    response_list.append(item)
                    cleaned_response_list.append(cleaned_response)
                    reward_list.append(reward)
                    correct_list.append(reward == 1.0)
                
                row_dict["prompt"] = prompt_text
                row_dict["response"] = response_list[0] if self.args.rollout == 1 else response_list
                row_dict["cleaned_response"] = cleaned_response_list[0] if self.args.rollout == 1 else cleaned_response_list
                row_dict["cleaned_gt_answer"] = cleaned_gt_answer
                row_dict["reward"] = reward_list[0] if self.args.rollout == 1 else reward_list
                row_dict["correct"] = correct_list[0] if self.args.rollout == 1 else correct_list
                results.append(row_dict)
                
                if len(results) % 10 == 0:
                    torch.cuda.empty_cache()
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing chunk in HFRayWorker: {e}")
        
        return results


# ==================== Inference Backend: vLLM ==================== #

@conditional_decorator(USE_RAY, ray.remote(num_gpus=1))
class VLLMRayWorker:
    """
    vLLM inference worker for high-throughput batched generation.

    Supported models: Qwen2.5-VL, MIMO-VL, Qwen2-VL, Qwen3-VL
    """
    def __init__(self, model_path, args):
        from vllm import LLM, SamplingParams
        self.args = args
        set_seed(42)
        self.sampling_params = SamplingParams(
            n=args.rollout, 
            temperature=args.temperature, 
            max_tokens=args.max_new_tokens, 
            top_p=args.top_p, 
            top_k=args.top_k, 
            min_p=0.0
        )
        self.llm = LLM(
            model=model_path,
            max_model_len=12800,
            gpu_memory_utilization=0.9,
            max_num_seqs=256,
            limit_mm_per_prompt={"image": 32, "video": 0},
            tensor_parallel_size=1,
            dtype=torch.bfloat16,
            trust_remote_code=True
        )

        self.processor = None
        self.tokenizer = None
        print(f"Worker initialized with model from {model_path}")
    
    def build_vllm_input(self, msg):
        """Convert message into vLLM input format (prompt + multi_modal_data)."""
        if self.args.model_type in ["Qwen2.5-VL", "MIMO-VL", "Qwen2-VL", "Qwen3-VL"]:
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(self.args.model_path, use_fast=True, trust_remote_code=True)
                self.tokenizer = self.processor.tokenizer
            image_data, _ = process_vision_info(msg)
            prompt_text = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            return {"prompt": prompt_text, "multi_modal_data": {"image": image_data}}
        else:
            raise NotImplementedError(f"Model type {self.args.model_type} not supported in VLLMRayWorker")
    
    def inference(self, data_chunk):
        """Batched inference: feed the entire chunk into the vLLM engine at once."""
        results = []
        
        try:
            messages = build_messages(data_chunk, self.args)
            with ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:
                futures = [executor.submit(self.build_vllm_input, msg) for msg in messages]
                vllm_inputs = [f.result() for f in futures]
            outputs = self.llm.generate(vllm_inputs, sampling_params=self.sampling_params)
            if self.args.rollout == 1:
                output_texts = [o.outputs[0].text for o in outputs]
            else:
                output_texts = [[item.text for item in o.outputs] for o in outputs]
            for row_dict, output_text, vllm_input in zip(data_chunk, output_texts, vllm_inputs):
                cleaned_gt_answer = clean_text(row_dict["answer"])
                output_text_list = [output_text] if type(output_text) != list else output_text
                response_list, cleaned_response_list, reward_list, correct_list = [], [], [], []
                
                for item in output_text_list:
                    cleaned_response = clean_text(item)
                    try:
                        reward = self.args.calculate_reward(cleaned_gt_answer, cleaned_response, row_dict)
                    except:
                        print(f"Error calculating reward: {cleaned_gt_answer}, {cleaned_response}, {row_dict}")
                        reward = 0

                    response_list.append(item)
                    cleaned_response_list.append(cleaned_response)
                    reward_list.append(reward)
                    correct_list.append(reward == 1.0)
                    
                row_dict["prompt"] = vllm_input["prompt"]
                row_dict["response"] = response_list[0] if self.args.rollout == 1 else response_list
                row_dict["cleaned_response"] = cleaned_response_list[0] if self.args.rollout == 1 else cleaned_response_list
                row_dict["cleaned_gt_answer"] = cleaned_gt_answer
                row_dict["reward"] = reward_list[0] if self.args.rollout == 1 else reward_list
                row_dict["correct"] = correct_list[0] if self.args.rollout == 1 else correct_list
                results.append(row_dict)
    
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing chunk: {e}")
        
        return results


# ==================== Inference Backend: LMDeploy ==================== #

@conditional_decorator(USE_RAY, ray.remote(num_gpus=1))
class LMDeployRayWorker:
    """
    LMDeploy inference worker.

    Supported models: InternVL3.5
    """
    def __init__(self, model_path, args):
        from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig as LMDeployGenerationConfig
        self.args = args
        set_seed(42)
        
        backend_config = PytorchEngineConfig(
            session_len=640000,
            tp=1,
            cache_max_entry_count=0.8,
        )
        
        self.pipe = pipeline(
            model_path, 
            backend_config=backend_config,
            log_level='WARNING'
        )
        
        self.gen_config = LMDeployGenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if args.temperature > 0 else 0.0,
            top_p=args.top_p,
            top_k=args.top_k if args.top_k > 0 else 0,
            repetition_penalty=1.0,
        )

        self.processor = None
        self.tokenizer = None
        
        print(f"LMDeploy Worker initialized with model from {model_path}")
    
    def build_lmdeploy_input(self, msg):
        """Convert message into LMDeploy input format (prompt_text, images)."""
        from lmdeploy.vl import load_image
        from lmdeploy.vl.constants import IMAGE_TOKEN
        if self.args.model_type in ["InternVL3.5"]:
            images = []
            prompt_text = ""
            for item in msg[0]["content"]:
                if item["type"] == "text":
                    prompt_text += item["text"]
                elif item["type"] == "image":
                    images.append(load_image(item["image"]))
                    prompt_text += IMAGE_TOKEN
            lmdeploy_input = (prompt_text, images)
        
        return lmdeploy_input, prompt_text
    
    def inference(self, data_chunk):
        """Batched inference via the LMDeploy pipeline."""
        results = []
        
        try:
            messages = build_messages(data_chunk, self.args)
            
            with ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:
                futures = [executor.submit(self.build_lmdeploy_input, msg) for msg in messages]
                lmdeploy_inputs = [f.result()[0] for f in futures]
                prompt_texts = [f.result()[1] for f in futures]
            
            if self.args.rollout == 1:
                outputs = self.pipe(
                    lmdeploy_inputs,
                    gen_config=self.gen_config
                )
                output_texts = [output.text for output in outputs]
                print(output_texts)
            else:
                output_texts = []
                for lmdeploy_input in tqdm(lmdeploy_inputs, desc="Processing with rollout", position=1):
                    rollout_outputs = []
                    for _ in range(self.args.rollout):
                        output = self.pipe([lmdeploy_input], gen_config=self.gen_config)
                        rollout_outputs.append(output[0].text)
                    output_texts.append(rollout_outputs)
            
            for row_dict, output_text, prompt_text in zip(data_chunk, output_texts, prompt_texts):
                cleaned_gt_answer = clean_text(row_dict["answer"])
                output_text_list = [output_text] if type(output_text) != list else output_text
                response_list, cleaned_response_list, reward_list, correct_list = [], [], [], []
                
                for item in output_text_list:
                    cleaned_response = clean_text(item)
                    try:
                        reward = self.args.calculate_reward(cleaned_gt_answer, cleaned_response, row_dict)
                    except:
                        print(f"Error calculating reward: {cleaned_gt_answer}, {cleaned_response}, {row_dict}")
                        reward = 0
                    
                    response_list.append(item)
                    cleaned_response_list.append(cleaned_response)
                    reward_list.append(reward)
                    correct_list.append(reward == 1.0)
                
                row_dict["prompt"] = prompt_text
                row_dict["response"] = response_list[0] if self.args.rollout == 1 else response_list
                row_dict["cleaned_response"] = cleaned_response_list[0] if self.args.rollout == 1 else cleaned_response_list
                row_dict["cleaned_gt_answer"] = cleaned_gt_answer
                row_dict["reward"] = reward_list[0] if self.args.rollout == 1 else reward_list
                row_dict["correct"] = correct_list[0] if self.args.rollout == 1 else correct_list
                results.append(row_dict)
                
                if len(results) % 10 == 0:
                    torch.cuda.empty_cache()
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing chunk in LMDeployRayWorker: {e}")
        return results


# ==================== Inference Backend: OpenAI / Azure API ==================== #

@conditional_decorator(USE_RAY, ray.remote(num_gpus=0))
class OpenAIRayWorker:
    """
    OpenAI / Azure API inference worker (no GPU required).

    model_path format: "https://<endpoint>?ak=<api_key>"
    Supported models: GPT-4o, GPT-4.1, GPT-5, Gemini, Claude, and other closed-source models
    """
    def __init__(self, model_path, args):
        self.args = args
        
        # Parse endpoint and api_key from the URL
        up = urlparse(model_path)
        query = parse_qs(up.query)
        self.endpoint = urlunparse((up.scheme, up.netloc, up.path, "", "", ""))
        self.api_key = query.get("ak", [""])[0]
        assert self.api_key, "No 'ak' parameter found in model_path"
        
        self.model_name = args.model_type
        
        self.api_version = getattr(args, "azure_api_version", "2024-03-01-preview")
        self.max_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.n = args.rollout
        self.is_azure = args.is_azure
        
        if self.is_azure:
            self.client = openai.AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
                api_key=self.api_key,
            )
        else:
            print(f"endpoint: {self.endpoint}")
            print(f"api_key: {self.api_key}")
            self.client = openai.OpenAI(
                base_url=self.endpoint,
                api_key=self.api_key
            )
        
        print(f"OpenAIRayWorker initialized:")
        print(f"  - Endpoint: {self.endpoint}")
        print(f"  - Model: {self.model_name}")
        print(f"  - API Version: {self.api_version}")
    
    def build_api_input(self, msg):
        """Convert message into OpenAI API format (images encoded as base64 data URLs)."""
        message_content = []
        prompt_text = ""
        
        for item in msg[0]["content"]:
            if item["type"] == "text":
                text_content = {"type": "text", "text": item["text"]}
                message_content.append(text_content)
                prompt_text += item["text"]
                
            elif item["type"] == "image":
                image_input = item["image"]
                
                if hasattr(image_input, 'save'):
                    buffer = io.BytesIO()
                    format_name = image_input.format if image_input.format else 'PNG'
                    image_input.save(buffer, format=format_name)
                    img_data = buffer.getvalue()
                    img_format = format_name.lower()
                    base64_data = base64.b64encode(img_data).decode('utf-8')
                    dataurl = f"data:image/{img_format};base64,{base64_data}"
                else:
                    raise ValueError(f"Unsupported image type: {type(image_input)}")
                    
                image_content = {
                    "type": "image_url", 
                    "image_url": {"url": dataurl}
                }
                message_content.append(image_content)
                prompt_text += "<image>"
                
            else:
                print(f"Warning: Unsupported content type: {item['type']}")
        
        return message_content, prompt_text
    
    def call_api(self, message_content, prompt_text, row_dict, max_retries=50, retry_interval=2):
        """Call the API with automatic retries, then compute reward."""
        try:
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": message_content}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        n=self.n
                    )
                    output_texts = [choice.message.content for choice in response.choices]
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_interval)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing chunk in OpenAIRayWorker: {e}")
        
        cleaned_gt_answer = clean_text(row_dict["answer"])
        output_text_list = output_texts if self.n > 1 else [output_texts[0]]
        response_list, cleaned_response_list, reward_list, correct_list = [], [], [], []
        
        for item in output_text_list:
            cleaned_response = clean_text(item)
            try:
                reward = self.args.calculate_reward(cleaned_gt_answer, cleaned_response, row_dict)
            except Exception as ee:
                print(f"Error calculating reward: {ee}")
                print(f"  GT: {cleaned_gt_answer}, Response: {cleaned_response}")
                reward = 0
            
            response_list.append(item)
            cleaned_response_list.append(cleaned_response)
            reward_list.append(reward)
            correct_list.append(reward == 1.0)
        
        row_dict["prompt"] = prompt_text
        row_dict["response"] = response_list[0] if self.n == 1 else response_list
        row_dict["cleaned_response"] = cleaned_response_list[0] if self.n == 1 else cleaned_response_list
        row_dict["cleaned_gt_answer"] = cleaned_gt_answer
        row_dict["reward"] = reward_list[0] if self.n == 1 else reward_list
        row_dict["correct"] = correct_list[0] if self.n == 1 else correct_list
        
        return row_dict
    
    def inference(self, data_chunk):
        """Process a data chunk with concurrent API calls via thread pool."""
        results = []
        
        try:
            messages = build_messages(data_chunk, self.args)
            
            with ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:
                futures = []
                
                for row_dict, msg in zip(data_chunk, messages):
                    message_content, prompt_text = self.build_api_input(msg)
                    future = executor.submit(self.call_api, message_content, prompt_text, row_dict)
                    futures.append(future)
                
                for future in tqdm(futures, desc="API Inference", position=1):
                    result = future.result()
                    results.append(result)
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing chunk in OpenAIRayWorker: {e}")
        
        return results


# ==================== Inference Dispatch ==================== #

def inference(args):
    """
    Main Ray distributed inference function.

    Pipeline: load data -> init Ray -> create workers -> dispatch batches ->
    collect results -> append to JSONL output
    """
    data = get_unprocessed_data(args)
    
    if not data:
        print("No data to process.")
        return []
    ray.init(ignore_reinit_error=True)
    num_ray_workers = torch.cuda.device_count()
    if USE_RAY and num_ray_workers == 0:
        print("No available GPUs")
        exit(1)

    print(f"Using {num_ray_workers} Ray workers")
    
    RayWorker = args.RayWorker
    workers = [RayWorker.remote(args.model_path, args) for _ in range(num_ray_workers)]
    
    all_results = []
    
    for i in tqdm(range(0, len(data), args.batch_size), desc="Processing batches", position=0):
        batch = data[i:i + args.batch_size]
        batch_size = len(batch)
        
        # Evenly distribute the batch across workers
        chunk_size = batch_size // num_ray_workers
        
        data_chunks = []
        for worker_idx in range(num_ray_workers):
            start_idx = worker_idx * chunk_size
            if worker_idx == num_ray_workers - 1:
                end_idx = batch_size
            else:
                end_idx = (worker_idx + 1) * chunk_size
            
            if start_idx < batch_size:
                chunk = batch[start_idx:end_idx]
                if len(chunk) > 0:
                    data_chunks.append((worker_idx, chunk))
        
        print(f"Batch {i//args.batch_size + 1}: split into chunks of sizes {[len(chunk[1]) for chunk in data_chunks]}")
        
        # Submit inference tasks in parallel
        futures = []
        for worker_idx, chunk in data_chunks:
            future = workers[worker_idx].inference.remote(chunk)
            futures.append(future)
        
        # Collect results and append to output file
        batch_results = []
        for future in futures:
            results = ray.get(future)
            batch_results.extend(results)
        
        with open(args.output_path, "a") as fout:
            for result in batch_results:
                fout.write(json.dumps(result) + "\n")
                fout.flush()
        
        all_results.extend(batch_results)
    
    ray.shutdown()
    return all_results


def debug_inference(args):
    """Single-GPU debug mode without Ray (enabled by setting DEBUG=1 env var)."""
    data = get_unprocessed_data(args)
    
    if not data:
        print("No data to process.")
        return []
    
    RayWorker = args.RayWorker
    worker = RayWorker(args.model_path, args)
    
    all_results = []
    
    for i in tqdm(range(0, len(data), args.batch_size), desc="Processing batches", position=0):
        batch = data[i:i + args.batch_size]
        
        batch_results = worker.inference(batch)
        
        with open(args.output_path, "a") as fout:
            for result in batch_results:
                fout.write(json.dumps(result) + "\n")
                fout.flush()
        
        all_results.extend(batch_results)
    return all_results


def save_results(output_path: str, results, final_acc):
    """Save evaluation results and aggregated metrics to a JSON file."""
    try:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"results": results, "final_acc": [final_acc]},
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error writing results to output file: {e}")


# ==================== Model & Benchmark Adaptation ==================== #

def adapt_for_model(args):
    """Select the appropriate inference backend worker based on model_type."""
    print(f">>> Model Type: {args.model_type}")
    if args.model_type in ["MIMO-VL", "Qwen2-VL", "Qwen2.5-VL"]:
        args.RayWorker = VLLMRayWorker
    elif args.model_type in ["InternVL3.5"]:
        args.RayWorker = LMDeployRayWorker
    elif args.model_type in ["Ovis2.5", "Keye-VL", "GLM-4.1V", "Kimi-VL", "MiniCPM-V-4.5"]:
        args.RayWorker = HFRayWorker
    elif args.model_type in ["gemini-2.5-pro", "gpt-4.1-2025-04-14", "gpt-5-2025-08-07"]:
        args.RayWorker = OpenAIRayWorker
        args.is_azure = True
    else:
        raise ValueError(f"Model type {args.model_type} not supported")


def adapt_for_benchmark(args):
    """Load the data path, prompt builder, reward calculator, and metrics function for the given benchmark."""
    datasets_dir = SCRIPT_DIR / "datasets"

    if args.benchmark in ["vsibench", "vsibench_debiased", "vsibench_metric"]:
        from utils.vsibench import build_vsibench_prompt, calculate_vsibench_reward, calculate_vsibench_metrics
        args.data_path = str(datasets_dir / "VSI-Bench" / "qa_processed.jsonl")
        args.build_prompt = build_vsibench_prompt
        args.calculate_reward = calculate_vsibench_reward
        args.calculate_metrics = calculate_vsibench_metrics
    elif args.benchmark == "sparbench":
        from utils.sparbench import build_sparbench_prompt, calculate_sparbench_reward, calculate_sparbench_metrics
        args.data_path = str(datasets_dir / "SPAR-Bench" / "qa_processed.jsonl")
        args.build_prompt = build_sparbench_prompt
        args.calculate_reward = calculate_sparbench_reward
        args.calculate_metrics = calculate_sparbench_metrics
    elif args.benchmark == "sitebench":
        from utils.sitebench import build_sitebench_prompt, calculate_sitebench_reward, calculate_sitebench_metrics
        args.data_path = str(datasets_dir / "SITE-Bench" / "qa_processed.jsonl")
        args.build_prompt = build_sitebench_prompt
        args.calculate_reward = calculate_sitebench_reward
        args.calculate_metrics = calculate_sitebench_metrics
    elif args.benchmark == "spbench":
        from utils.spbench import build_spbench_prompt, calculate_spbench_reward, calculate_spbench_metrics
        args.data_path = str(datasets_dir / "SPBench" / "qa_processed.jsonl")
        args.build_prompt = build_spbench_prompt
        args.calculate_reward = calculate_spbench_reward
        args.calculate_metrics = calculate_spbench_metrics
    else:
        raise ValueError(f"Benchmark {args.benchmark} not supported")


# ==================== Main Entry ==================== #

if __name__=="__main__":
    set_seed(42)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    parser = ArgumentParser(description="EgoMind evaluation script")
    parser.add_argument("--model_path", "-m", type=str, default=None, help="HuggingFace model path or local checkpoint")
    parser.add_argument("--output_path", "-o", type=str, default=None, help="JSONL output file path")
    parser.add_argument("--model_type", "-t", type=str, default="Qwen2.5-VL", help="model architecture type")
    parser.add_argument("--benchmark", type=str, default="vsibench", help="benchmark name: vsibench/sparbench/sitebench/spbench")
    parser.add_argument("--custom_data_path", "-a", type=str, default=None, help="override default data path")
    parser.add_argument("--precision", type=str, default="bf16", help="model precision")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="max generation length")
    parser.add_argument("--max_pixels", type=int, default=200704, help="max pixels per image (256*28*28)")
    parser.add_argument("--min_pixels", type=int, default=3136, help="min pixels per image")
    parser.add_argument("--batch_size", type=int, default=512, help="samples per Ray dispatch batch")
    parser.add_argument("--num_frames", type=int, default=16, help="frames to sample from each video")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing output file")
    parser.add_argument("--format_prompt", type=str, default=str(SCRIPT_DIR / "format_prompt" / "simple_r1v_format.jinja"), help="Jinja format prompt template")
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature (0 = greedy)")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p sampling")
    parser.add_argument("--top_k", type=int, default=-1, help="top-k sampling (-1 = disabled)")
    parser.add_argument("--rollout", type=int, default=1, help="number of generation rollouts per sample")
    parser.add_argument("--only_eval", action="store_true", help="skip inference, only re-evaluate existing output")
    parser.add_argument("--enable_thinking", action="store_true", help="enable thinking mode for supported models")
    parser.add_argument("--num_workers", type=int, default=16, help="thread pool workers for I/O tasks")
    args = parser.parse_args()

    print(f">>> RUN {args.benchmark} with {args.model_path}")

    if args.only_eval:
        # Eval-only mode: skip inference, compute metrics from existing output
        assert os.path.exists(args.output_path), f"Output path {args.output_path} does not exist"
        adapt_for_benchmark(args)
    else:
        # Full pipeline: inference + evaluation
        assert args.model_path is not None and args.output_path is not None, "model_path and output_path are required"
        
        adapt_for_benchmark(args)
        adapt_for_model(args)
        if args.custom_data_path is not None:
            args.data_path = args.custom_data_path
        parent = os.path.dirname(args.output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if DEBUG:
            print("Debug mode enabled: Running locally without Ray.")
            results = debug_inference(args)
        else:
            inference(args)

    # Reload output file and recompute rewards uniformly for consistency
    final_output = []
    with open(args.output_path, "r") as f:
        for line in f:
            item = json.loads(line)
            final_output.append(item)
    
    for row_dict in final_output:
        cleaned_gt_answer = clean_text(row_dict["answer"])
        if type(row_dict["response"]) != list:
            response_list = [row_dict["response"]]
        else:
            response_list = row_dict["response"]
        cleaned_response_list = [clean_text(x) for x in response_list]
        
        reward_list = []
        for cleaned_response in cleaned_response_list:
            try:
                reward = args.calculate_reward(cleaned_gt_answer, cleaned_response, row_dict)
            except Exception as ee:
                print(f"Error calculating reward: {ee}")
                print(f"  GT: {cleaned_gt_answer}, Response: {cleaned_response}")
                reward = 0
            reward_list.append(reward)
        
        row_dict["cleaned_response"] = cleaned_response_list if len(cleaned_response_list) > 1 else cleaned_response_list[0]
        row_dict["reward"] = reward_list if len(reward_list) > 1 else reward_list[0]

    # Aggregate metrics and save *_eval.json
    try:
        final_metrics = args.calculate_metrics(final_output)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        final_metrics = {"error": "Error calculating metrics"}
    
    ext = os.path.splitext(args.output_path)[-1]
    eval_result_file = args.output_path.replace(ext, "_eval.json")
    save_results(eval_result_file, final_output, final_metrics)
