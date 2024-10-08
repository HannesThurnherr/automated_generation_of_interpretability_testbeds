### Installing Tracrbench

To use Tracrbench you will need to install the tracr library (https://github.com/google-deepmind/tracr) with the following commands

```
git clone https://github.com/deepmind/tracr
cd tracr
pip3 install .
cd ..
```

the rest of the requirements can be installed using
```
git clone https://github.com/HannesThurnherr/automated_generation_of_interpretability_testbeds
cd automated_generation_of_interpretability_testbeds
pip install -r requirements.txt
```
### Using Tracrbench

To use Tracrbench you must define a model class with a generate method. To do this you can let your class inherit from the abstract model class in the model.py file.

For example:
```python
from abstract_model import Model
class OpenAIChatModel(Model):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: str | None, stop_strings, **kwargs) -> str:
        #rest of the method
```

you can then import the methods evaluate_model() and visualize_results() from main.py

```python
from models.OpenAIChatModel import OpenAIChatModel
from main import evaluate_model, visualize_results

model = OpenAIChatModel("gpt-3.5-turbo-0125")
results = evaluate_model(model)
visualize_results(results)
```

The evaluate_model() method will save the results of the evaluation to a JSON file and a continuous log of all the processes to a txt file (if set to verbose)
The visualize_results() methods will take those results and both display and illustrate them using matplotlib and save the visualisation as a pdf file.


### Using TracrBench models in TransformerLens
You can use TraceBench models directly in TransformerLens. Do so with the following code:
```python
from transformerlens_models import load_tracr_model, load_model_from_hf

#load model from local dataset
tracr_model = load_tracr_model("dataset/make_hist")
#...or directly from huggingface
tracr_model = load_model_from_hf("make_hist")

#run the model on some input
input = ["BOS", 1, 2, 3]
out = tracr_model.run_with_cache(input)
print("Original Decoding:", out)

#or interact directly with the encapsulated hooked_transformer:
hooked_tf = tracr_model.tl_model
```


