"""LMDeploy inference worker."""

import torch
import ray
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from workers.common import (
    conditional_decorator, USE_RAY,
    clean_text, build_messages,
)


@conditional_decorator(USE_RAY, ray.remote(num_gpus=1))
class LMDeployRayWorker:
    """
    LMDeploy inference worker.

    Supported models: InternVL3.5
    """
    def __init__(self, model_path, args):
        from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig as LMDeployGenerationConfig
        from transformers import set_seed
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
