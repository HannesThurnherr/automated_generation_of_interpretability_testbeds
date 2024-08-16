from models.AnthropicChatModel import AnthropicChatModel
from main import evaluate_model, visualize_results


model = AnthropicChatModel("claude-3-5-sonnet-20240620")
results = evaluate_model(model)

visualize_results(results)