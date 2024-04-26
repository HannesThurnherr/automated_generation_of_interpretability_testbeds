import os
import time

from openai import OpenAI
from RASP_eval import evaluate_model, visualize_results
from abstract_model import Model
import backoff
import tiktoken
from tracr.rasp import rasp
from openai._exceptions import (
    APIConnectionError,
    RateLimitError,
)
import anthropic
import google.generativeai as genai


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
        try:
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
        except:
            print("pausing for 60 seconds")
            time.sleep(60)
            return self.generate(prompt, system_prompt, stop_strings, **kwargs)


class GeminiChatModel:
    def __init__(self, model_name):
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
        response = self.client.generate_content(prompt)

        # Extract the text from the response if available.
        if response and hasattr(response, 'text'):
            return response.text
        else:
            return ""






#model = OpenAIChatModel("gpt-4-turbo-2024-04-09")
#model = OpenAIChatModel("gpt-3.5-turbo-0125")
model = AnthropicChatModel("claude-3-haiku-20240307")
#model = GeminiChatModel("models/gemini-pro")
results = evaluate_model(model)
visualize_results(results)