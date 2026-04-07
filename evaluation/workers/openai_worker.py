"""OpenAI / Azure API inference worker (no GPU required)."""

import io
import base64
import time
import openai
import ray
from urllib.parse import urlparse, parse_qs, urlunparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from workers.common import (
    conditional_decorator, USE_RAY,
    clean_text, build_messages,
)


@conditional_decorator(USE_RAY, ray.remote(num_gpus=0))
class OpenAIRayWorker:
    """
    OpenAI / Azure API inference worker (no GPU required).

    model_path format: "https://<endpoint>?ak=<api_key>"
    Supported models: GPT-4o, GPT-4.1, GPT-5, Gemini, Claude, and other closed-source models
    """
    def __init__(self, model_path, args):
        self.args = args

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
