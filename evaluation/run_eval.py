"""
EgoMind Evaluation Script

Supports multiple inference backends (HuggingFace / vLLM / LMDeploy / OpenAI API)
with Ray-based multi-GPU parallel inference and automatic benchmark metric computation.

Usage:
    python run_eval.py \
        --model_path models/EgoMind-7B \
        --output_path outputs/vsibench.jsonl \
        --benchmark vsibench

Supported benchmarks: vsibench, sparbench, sitebench, spbench
"""

import json
import os
import torch
import ray
from argparse import ArgumentParser
from pathlib import Path
from transformers import set_seed
from tqdm import tqdm

from workers import (
    DEBUG, USE_RAY, SCRIPT_DIR,
    clean_text,
    get_unprocessed_data,
    HFRayWorker,
    VLLMRayWorker,
    LMDeployRayWorker,
    OpenAIRayWorker,
)

print(f"DEBUG: {DEBUG}, USE_RAY: {USE_RAY}")


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

        futures = []
        for worker_idx, chunk in data_chunks:
            future = workers[worker_idx].inference.remote(chunk)
            futures.append(future)

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
    parser.add_argument("--batch_size", type=int, default=256, help="samples per Ray dispatch batch")
    parser.add_argument("--num_frames", type=int, default=16, help="frames to sample from each video")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing output file")
    parser.add_argument("--format_prompt", type=str, default=str(SCRIPT_DIR / "prompts" / "simple_r1v_format.jinja"), help="Jinja format prompt template")
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
        assert os.path.exists(args.output_path), f"Output path {args.output_path} does not exist"
        adapt_for_benchmark(args)
    else:
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
