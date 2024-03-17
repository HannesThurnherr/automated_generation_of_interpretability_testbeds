import os
from openai import OpenAI
import json
from RASP_eval import evaluate_model, visualize_results
from RASP_eval import Model
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import backoff
import tiktoken
from tracr.rasp import rasp
from openai._exceptions import (
    APIConnectionError,
    RateLimitError,
)
import anthropic
import google.generativeai as genai
from replicate.client import Client

class OpenAIChatModel(Model):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model_name = model_name
        os.environ['OPENAI_API_KEY'] = input("your Openai API key: ")  # 'your_api_key'
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def generate(self, prompt: str, system_prompt: str | None, stop_strings, **kwargs) -> str:
        assert isinstance(prompt, str), "Prompt must be a string."
        if kwargs.get("prepend_completion", None):
            prompt += kwargs.pop("prepend_completion")
        params = {
            "prompt": prompt,
            "stop_strings": stop_strings,
        }
        if system_prompt is not None:
            params["system_prompt"] = system_prompt
        completion = self._call_gpt_api(**params, **kwargs)
        return completion

    @backoff.on_exception(
        backoff.expo,
        exception=(
            RateLimitError,
            APIConnectionError
        ),
        max_value=60,
        factor=1.5,
    )
    def _call_gpt_api(self, prompt, stop_strings, system_prompt="", **kwargs) -> str:
        messages = [
            {"role": "user", "content": prompt},
        ]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        response = OpenAI().chat.completions.create(model=self.model_name, messages=messages, **kwargs)
        return response.choices[0].message.content


class AnthropicChatModel(Model):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model_name = model_name
        api_key = input("your anthropic API key: ")
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, system_prompt: str, stop_strings: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 4096,
            "stop_sequences": [],
        }
        if system_prompt is not None:
            params["system"] = system_prompt
        completion = self.client.messages.create(**params)
        if len(completion.content) == 0:
            return ""
        return completion.content[0].text


class GeminiChatModel:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        genai.configure(api_key=input("your Gemini API key: "))
        self.client = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, system_prompt: str = None, stop_strings: list = None, **kwargs) -> str:
        # Prepare the messages structure as required by the API.
        messages = [{"role": "user", "content": prompt}]

        # Configure additional parameters if provided.
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if not stop_strings:
            stop_strings = []

        # Prepare the request parameters.
        params = {
            "prompt": messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "stop_sequences": stop_strings,
            # Add other parameters here as needed.
        }

        # Execute the request and handle the response.
        response = self.client.generate_content(**params)

        # Extract the text from the response if available.
        if response and hasattr(response, 'text'):
            return response.text
        else:
            return ""




class LlamaChat:
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = Client(os.getenv("REPLICATE_API_KEY"))

    @backoff.on_exception(backoff.expo, Exception, max_value=60, factor=1.5)
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop_strings: list[str] | None,
        system_prompt: Optional[str] = None,
        prepend_completion: Optional[str] = None,
    ):
        if temperature == 0:
            temperature = 0.01

        # if you ask for 1 token, you get 0 back, but if you ask for 2, you get 1 back
        max_tokens += 1

        # referred to these links for the below code:
        # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L324-L362
        # https://github.com/replicate/cog-llama-template/blob/main/predict.py#L20-L23
        if system_prompt:
            prompt_template = (
                f"{self.B_INST} {self.B_SYS}{{system_prompt}}{self.E_SYS}{{prompt}} {self.E_INST}"
            )
        else:
            prompt_template = f"{self.B_INST} {{prompt}} {self.E_INST}"

        if prepend_completion:
            prompt_template = f"{prompt_template}{prepend_completion}"

        inputs = {
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "prompt_template": prompt_template,
        }
        if system_prompt is not None:
            inputs["system_prompt"] = system_prompt
        completion = self.model.run(self.model_name, input=inputs)
        completion = "".join(completion)

        # handle stop strings
        if stop_strings is not None:
            for stop_token in stop_strings:
                completion = completion.split(stop_token)[0]

        return completion






#model = OpenAIChatModel("gpt-4-0125-preview")
model = OpenAIChatModel("gpt-3.5-turbo-0125")
#model = AnthropicChatModel("claude-3-haiku-20240307")
results = evaluate_model(model)
visualize_results(results)