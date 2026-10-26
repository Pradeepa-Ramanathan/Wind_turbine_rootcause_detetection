import numpy as np
import torch

from src.model import GraphTemporalForecaster, make_windows


def test_model_output_shape_and_finite_values():
    nodes, window = 5, 24
    graph = np.ones((nodes, nodes), dtype=np.float32) - np.eye(nodes, dtype=np.float32)
    model = GraphTemporalForecaster(nodes, explicit_graphs=[graph], window=window, hidden=8, layers=1).eval()
    x = torch.randn(2, window, nodes)
    with torch.no_grad():
        prediction = model(x)
    assert prediction.shape == (2, nodes)
    assert torch.isfinite(prediction).all()


def test_inference_is_deterministic():
    nodes, window = 5, 24
    graph = np.ones((nodes, nodes), dtype=np.float32) - np.eye(nodes, dtype=np.float32)
    model = GraphTemporalForecaster(nodes, explicit_graphs=[graph], window=window, hidden=8, layers=1).eval()
    x = torch.randn(2, window, nodes)
    with torch.no_grad():
        first = model(x)
        second = model(x)
    assert torch.allclose(first, second)


def test_make_windows_alignment():
    values = np.arange(40, dtype=np.float32).reshape(10, 4)
    x, y = make_windows(values, window=3)
    assert x.shape == (7, 3, 4)
    assert y.shape == (7, 4)
    np.testing.assert_array_equal(y[0], values[3])
