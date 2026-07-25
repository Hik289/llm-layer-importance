import torch
import torch.nn.functional as F


def block_influence(
    input_hidden_state: torch.Tensor,
    output_hidden_state: torch.Tensor,
    angular=False,
):
    """Return token-wise block influence for hidden states shaped ``(B, S, D)``."""
    if input_hidden_state.shape != output_hidden_state.shape:
        raise ValueError("input and output hidden states must have the same shape")

    d = input_hidden_state.shape[-1]
    input_flat = input_hidden_state.reshape(-1, d)
    output_flat = output_hidden_state.reshape(-1, d)
    sim = F.cosine_similarity(input_flat, output_flat, dim=-1, eps=1e-8)
    sim = sim.nan_to_num(nan=0.0).clamp(-1.0, 1.0)

    if angular:
        return torch.arccos(sim) / torch.pi

    return 1 - sim
