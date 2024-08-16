import time
from appollo_research_project.repo.automated_generation_of_interpretability_testbeds.abstract_model import Model
import anthropic

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
