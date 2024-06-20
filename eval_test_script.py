from models.OpenAIChatModel import OpenAIChatModel
from main import evaluate_model, visualize_results


model = OpenAIChatModel("gpt-4-turbo-2024-04-09")
results = evaluate_model(model)

visualize_results(results)