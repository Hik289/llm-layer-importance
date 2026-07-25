import pytest
import torch

from peft_pretraining.utils.metrics import block_influence


def test_block_influence_matches_pairwise_cosine():
    inputs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    outputs = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    result = block_influence(inputs, outputs)
    assert torch.allclose(result, torch.tensor([0.0, 1.0]))


def test_angular_distance_is_finite_for_zero_vectors():
    states = torch.zeros(1, 2, 3)
    result = block_influence(states, states, angular=True)
    assert torch.isfinite(result).all()
    assert torch.allclose(result, torch.full((2,), 0.5))


def test_block_influence_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        block_influence(torch.zeros(1, 2, 3), torch.zeros(1, 3, 3))
