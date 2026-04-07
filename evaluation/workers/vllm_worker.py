"""vLLM inference worker for high-throughput batched generation."""

import torch
import ray
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from workers.common import (
    conditional_decorator, USE_RAY,
    clean_text, build_messages,
)


@conditional_decorator(USE_RAY, ray.remote(num_gpus=1))
class VLLMRayWorker:
    """
    vLLM inference worker for high-throughput batched generation.

    Supported models: Qwen2.5-VL, MIMO-VL, Qwen2-VL, Qwen3-VL
    """
    def __init__(self, model_path, args):
        from vllm import LLM, SamplingParams
        self.args = args
        from transformers import set_seed
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
