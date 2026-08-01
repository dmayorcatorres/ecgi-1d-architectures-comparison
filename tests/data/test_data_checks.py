import pytest
import torch

from utils.data_checks import assert_no_nan_batch


def test_assert_no_nan_batch_passes_on_clean_data():
    xb = torch.randn(4, 12, 256)
    yb = torch.randn(4, 256)
    assert_no_nan_batch(xb, yb, tag="clean")  # no debe lanzar


def test_assert_no_nan_batch_raises_on_nan_in_input():
    xb = torch.randn(4, 12, 256)
    xb[0, 0, 0] = float("nan")
    yb = torch.randn(4, 256)
    with pytest.raises(ValueError, match="NaN/Inf"):
        assert_no_nan_batch(xb, yb, tag="nan_input")


def test_assert_no_nan_batch_raises_on_inf_in_target():
    xb = torch.randn(4, 12, 256)
    yb = torch.randn(4, 256)
    yb[1, 5] = float("inf")
    with pytest.raises(ValueError, match="NaN/Inf"):
        assert_no_nan_batch(xb, yb, tag="inf_target")
