"""HuggingFace native inference worker."""

import torch
import ray
from concurrent.futures import ThreadPoolExecutor
from transformers import (
    set_seed, AutoModel, AutoProcessor, AutoTokenizer,
    AutoModelForCausalLM, GenerationConfig, Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from workers.common import (
    conditional_decorator, USE_RAY,
    clean_text, build_messages,
)


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
            from keye_vl_utils import process_vision_info as keye_process_vision_info

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

                    if self.args.model_type != "Ovis2.5":
                        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]

                    output_texts = self.tokenizer.batch_decode(
                        generated_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False
                    )

                output_texts = [self.post_process(text) for text in output_texts]

                if self.args.rollout == 1:
                    output_text = output_texts[0]
                else:
                    output_text = output_texts

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
