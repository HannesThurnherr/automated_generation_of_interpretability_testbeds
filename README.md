### Installing rasp eval

To use RASP_eval you will need to install the tracr library (https://github.com/google-deepmind/tracr) with the following commands

```
git clone https://github.com/deepmind/tracr
cd tracr
pip3 install .
```
the rest of the required packages can be installed using

```
pip install requirements
```
### Using RASP_eval

To use RASP_eval you must define a model class with a generate method. To do this you can let your class inherit from the abstract model class in the model.py file.

For example:
```python
from abstract_model import Model
class OpenAIChatModel(Model):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model_name = model_name
        os.environ['OPENAI_API_KEY'] = input("your Openai API key: ")  # 'your_api_key'
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def generate(self, prompt: str, system_prompt: str | None, stop_strings, **kwargs) -> str:
        #rest of the method
```

you can then import the methods evaluate_model() and visualize_results() from RASP_eval

```python
from RASP_eval import evaluate_model, visualize_results

model = OpenAIChatModel("gpt-3.5-turbo-0125")
results = evaluate_model(model)
visualize_results(results)
```

The evaluate_model() method will save the results of the evaluation to a JSON file and a continuous log of all the processes to a txt file (if set to verbose)
The visualize_results() methods will take those results and both display and illustrate them using matplotlib and save the visualisation as a pdf file.
