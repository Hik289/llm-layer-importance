import argparse
import os

import torch
from transformers import LlamaForCausalLM


def remove_layers_and_save(model_path, output_dir, layers_to_remove):
    """Remove selected LLaMA layers, renumber them, and save the checkpoint."""
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    layer_count = len(model.model.layers)
    invalid = sorted({index for index in layers_to_remove if not 0 <= index < layer_count})
    if invalid:
        raise IndexError(f"layer indices out of range for {layer_count} layers: {invalid}")

    os.makedirs(output_dir, exist_ok=True)
    for layer_idx in sorted(set(layers_to_remove), reverse=True):
        del model.model.layers[layer_idx]

    for layer_idx, module in enumerate(model.model.layers):
        if hasattr(module.self_attn, "layer_idx"):
            module.self_attn.layer_idx = layer_idx

    model.config.num_hidden_layers = len(model.model.layers)
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove layers from LLaMA model and save the modified version.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained LLaMA model")
    parser.add_argument("--layer_index", type=int, required=True, help="Index of the layer to remove")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the modified model")

    args = parser.parse_args()

    remove_layers_and_save(args.model_path, args.save_path, [args.layer_index])
