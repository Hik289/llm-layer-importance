"""Hugging Face wrapper for layer-importance and pruning experiments."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

from .metrics import block_influence


class ShortHFModel:
    """Wrap a Hugging Face model and expose its transformer layers."""

    def __init__(
        self,
        model_name: str,
        layers_path: str,
        n_prune_layers: Optional[int] = None,
        mode: str = "hf",
        *,
        tokenizer_name: Optional[str] = None,
        device: Optional[str] = None,
        local_files_only: bool = False,
        cache_dir: Optional[str] = None,
    ):
        if mode not in {"hf", "diy", "glm"}:
            raise ValueError(f"unsupported model mode: {mode}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        tokenizer_source = tokenizer_name or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

        load_options = {
            "torch_dtype": dtype,
            "local_files_only": local_files_only,
            "cache_dir": cache_dir,
        }
        if mode == "diy":
            self.model = LlamaForCausalLM.from_pretrained(model_name, **load_options)
        elif mode == "glm":
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                **load_options,
            )
        elif "bert" in model_name.lower():
            self.model = AutoModel.from_pretrained(model_name, **load_options)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_options)
        self.model.to(self.device)

        layers = self.model
        for component in layers_path.split("."):
            layers = getattr(layers, component)
        self.layers = layers
        self.n_prune_layers = n_prune_layers
        self.importances = [0.0 for _ in self.layers]

    def remove_layers(
        self,
        layers_to_remove: Optional[List[int]] = None,
        angular: bool = False,
    ) -> List[int]:
        """Remove explicitly selected layers or the least-important layers."""
        selected = list(layers_to_remove or [])
        if angular:
            if not self.n_prune_layers:
                raise ValueError("n_prune_layers is required for angular pruning")
            scores = np.asarray(self.importances[: -self.n_prune_layers + 1])
            start_layer = int(np.argmin(scores))
            selected = list(range(start_layer, start_layer + self.n_prune_layers))
        elif not selected and self.n_prune_layers:
            selected = np.argsort(np.asarray(self.importances))[: self.n_prune_layers].tolist()

        invalid = [index for index in selected if not 0 <= index < len(self.layers)]
        if invalid:
            raise IndexError(f"layer indices out of range: {sorted(set(invalid))}")
        for layer_idx in sorted(set(selected), reverse=True):
            del self.layers[layer_idx]
        return sorted(set(selected))

    def compute_bi(self, hiddens: List[torch.Tensor], angular: bool, n: int) -> None:
        """Accumulate block influence between states separated by ``n`` layers."""
        if n < 1 or n >= len(hiddens):
            raise ValueError(f"n must be in [1, {len(hiddens) - 1}], got {n}")
        for index in range(len(hiddens) - n):
            input_hidden = hiddens[index]
            output_hidden = hiddens[index + n]
            if angular:
                input_hidden = input_hidden[:, -1:]
                output_hidden = output_hidden[:, -1:]
            self.importances[index] += block_influence(
                input_hidden,
                output_hidden,
                angular=angular,
            ).mean().cpu().item()

    @torch.inference_mode()
    def eval_importance(
        self,
        prompts: List[str],
        max_seq_len: int,
        stride: int = 256,
        max_gen_len: int = 0,
        temperature: float = 0.6,
        top_p: float = 0.9,
        angular: bool = False,
        n: int = 1,
    ) -> int:
        """Accumulate layer importance over sliding windows and return their count."""
        prompt_tokens = self.tokenizer(
            prompts,
            padding=True,
            max_length=max_seq_len,
            return_attention_mask=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = prompt_tokens.input_ids
        attention_mask = prompt_tokens.attention_mask
        max_prompt_len = input_ids.shape[1]
        calculation_count = 0

        for start in range(0, max_prompt_len, stride):
            active_rows = (attention_mask.sum(dim=-1) > start).nonzero(as_tuple=True)[0]
            if active_rows.numel() == 0:
                continue
            inputs = input_ids[active_rows, start : start + max_seq_len].to(self.device)
            attention = attention_mask[active_rows, start : start + max_seq_len].to(self.device)
            if max_gen_len == 0:
                outputs = self.model(
                    input_ids=inputs,
                    attention_mask=attention,
                    output_hidden_states=True,
                )
            else:
                outputs = self.model.generate(
                    input_ids=inputs,
                    attention_mask=attention,
                    max_new_tokens=max_gen_len,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
            self.compute_bi(outputs.hidden_states, angular=angular, n=n)
            calculation_count += 1
        return calculation_count
