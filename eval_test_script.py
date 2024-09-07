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