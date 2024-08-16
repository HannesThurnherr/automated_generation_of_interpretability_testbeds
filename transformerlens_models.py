import pickle

from transformer_lens import HookedTransformer, HookedTransformerConfig
import einops
import torch
import numpy as np
import json
from tracr.rasp import rasp
from tracr.compiler import compiling
import sys
from tracr.compiler.assemble import AssembledTransformerModel
from tracr.transformer import encoder

class tracr_model(torch.nn.Module):
    def __init__(self, input_encoder, output_encoder, tl_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tl_model = tl_model
        self.input_encoder = input_encoder
        self.output_encoder = output_encoder

    def forward(self, input):
        encoding = self.input_encoder.encode(input)
        input_tokens_tensor = torch.tensor(encoding).unsqueeze(dim=0)
        logits = self.tl_model(input_tokens_tensor)
        max_output_indices = logits.squeeze(dim=0).argmax(dim=-1)
        decoded_output = self.output_encoder.decode(max_output_indices.tolist())
        decoded_output_with_bos = [self.input_encoder.bos_token] + decoded_output[1:]
        return decoded_output_with_bos

    def run_with_cache(self, input):
        encoding = self.input_encoder.encode(input)
        input_tokens_tensor = torch.tensor(encoding).unsqueeze(dim=0)
        logits, cache = self.tl_model.run_with_cache(input_tokens_tensor)
        max_output_indices = logits.squeeze(dim=0).argmax(dim=-1)
        decoded_output = self.output_encoder.decode(max_output_indices.tolist())
        decoded_output_with_bos = [self.input_encoder.bos_token] + decoded_output[1:]
        return decoded_output_with_bos, logits, cache


def tracr_to_tl(model: AssembledTransformerModel):
    n_heads = model.model_config.num_heads
    n_layers = model.model_config.num_layers
    d_head = model.model_config.key_size
    d_mlp = model.model_config.mlp_hidden_size
    act_fn = "relu"
    normalization_type = "LN" if model.model_config.layer_norm else None
    attention_type = "causal" if model.model_config.causal else "bidirectional"
    n_ctx = model.params["pos_embed"]['embeddings'].shape[0]
    # Equivalent to length of vocab, with BOS and PAD at the end
    d_vocab = model.params["token_embed"]['embeddings'].shape[0]
    # Residual stream width, I don't know of an easy way to infer it from the above config.
    d_model = model.params["token_embed"]['embeddings'].shape[1]
    # Equivalent to length of vocab, WITHOUT BOS and PAD at the end because we never care about these outputs
    # In practice, we always feed the logits into an argmax
    d_vocab_out = model.params["token_embed"]['embeddings'].shape[0] - 2
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_head=d_head,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        d_vocab_out=d_vocab_out,
        d_mlp=d_mlp,
        n_heads=n_heads,
        act_fn=act_fn,
        attention_dir=attention_type,
        normalization_type=normalization_type,
    )
    tl_model = HookedTransformer(cfg)
    sd = {}
    sd["pos_embed.W_pos"] = model.params["pos_embed"]['embeddings']
    sd["embed.W_E"] = model.params["token_embed"]['embeddings']
    # Equivalent to max_seq_len plus one, for the BOS
    # The unembed is just a projection onto the first few elements of the residual stream, these store output tokens
    # This is a NumPy array, the rest are Jax Arrays, but w/e it's fine.
    sd["unembed.W_U"] = np.eye(d_model, d_vocab_out)
    for l in range(n_layers):
        sd[f"blocks.{l}.attn.W_K"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/key"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_K"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/key"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/query"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/query"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.W_V"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/value"]["w"],
            "d_model (n_heads d_head) -> n_heads d_model d_head",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_V"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/value"]["b"],
            "(n_heads d_head) -> n_heads d_head",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.W_O"] = einops.rearrange(
            model.params[f"transformer/layer_{l}/attn/linear"]["w"],
            "(n_heads d_head) d_model -> n_heads d_head d_model",
            d_head=d_head,
            n_heads=n_heads
        )
        sd[f"blocks.{l}.attn.b_O"] = model.params[f"transformer/layer_{l}/attn/linear"]["b"]
        sd[f"blocks.{l}.mlp.W_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["w"]
        sd[f"blocks.{l}.mlp.b_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["b"]
        sd[f"blocks.{l}.mlp.W_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["w"]
        sd[f"blocks.{l}.mlp.b_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["b"]
    for k, v in sd.items():
        # I cannot figure out a neater way to go from a Jax array to a numpy array lol
        sd[k] = torch.tensor(np.array(v))
    tl_model.load_state_dict(sd, strict=False)
    return tl_model

def save_tracr_model_as_tl_model(model: AssembledTransformerModel, path):
    tl_model = tracr_to_tl(model)
    torch.save(tl_model, path+'tl_model.pt')
    in_out_encoder = {"input_encoder": model.input_encoder, "output_encoder": model.output_encoder}
    with open(path+"/input&output_encoder.pkl", "wb") as f:
        pickle.dump(in_out_encoder, f)


def load_tracr_model(path):
    tl_model = torch.load(path+"/tl_model.pt")
    with open(path+"/input&output_encoder.pkl", "rb") as f:
        in_out_encoder = pickle.load(f)
    return tracr_model(in_out_encoder["input_encoder"], in_out_encoder["output_encoder"], tl_model)


