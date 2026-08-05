import torch

from jetson_app.model import AnomalyGRU


def test_forward_shapes_with_mixed_tags():
    model = AnomalyGRU(
        num_tags=3,
        continuous_indices=[0, 2],
        binary_indices=[1],
        hidden_size=8,
        num_layers=1,
    )
    x = torch.randn(4, 5, 3)  # batch=4, window=5, tags=3
    continuous_out, binary_logits = model(x)
    assert continuous_out.shape == (4, 2)
    assert binary_logits.shape == (4, 1)


def test_forward_all_continuous_binary_logits_none():
    model = AnomalyGRU(
        num_tags=2,
        continuous_indices=[0, 1],
        binary_indices=[],
        hidden_size=4,
        num_layers=1,
    )
    x = torch.randn(2, 3, 2)
    continuous_out, binary_logits = model(x)
    assert continuous_out.shape == (2, 2)
    assert binary_logits is None


def test_forward_all_binary_continuous_out_none():
    model = AnomalyGRU(
        num_tags=2,
        continuous_indices=[],
        binary_indices=[0, 1],
        hidden_size=4,
        num_layers=1,
    )
    x = torch.randn(2, 3, 2)
    continuous_out, binary_logits = model(x)
    assert continuous_out is None
    assert binary_logits.shape == (2, 2)
