from appollo_research_project.repo.automated_generation_of_interpretability_testbeds.OpenAIChatModel import OpenAIChatModel
from main import evaluate_model, visualize_results






model = OpenAIChatModel("gpt-4-turbo-2024-04-09")
#model = OpenAIChatModel("gpt-3.5-turbo-0125")
#model = AnthropicChatModel("claude-3-haiku-20240307")
#model = GeminiChatModel("models/gemini-pro")
results = evaluate_model(model)
#with open("evaluation_results_gpt-3.5-turbo-0125_20240501.json") as f:
    #results = json.load(f)
visualize_results(results)