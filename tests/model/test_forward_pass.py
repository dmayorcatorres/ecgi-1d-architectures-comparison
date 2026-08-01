import pytest
import torch

from config import DataConfig
from models.factory import ModelFactory

# Un representante por familia (ver docs/architecture.md) para mantener la
# suite rápida; test_model_registry.py ya cubre las 36 variantes a nivel
# de registro/tipo.
REPRESENTATIVE_VARIANTS = ["mlp", "rnn_gru", "cnnlstm", "alstm"]

BATCH_SIZE = 2
TIME_STEPS = 64
INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 64


@pytest.fixture
def factory():
    cfg = DataConfig(
        time_steps=TIME_STEPS,
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
    )
    return ModelFactory(cfg)


@pytest.mark.parametrize("variant", REPRESENTATIVE_VARIANTS)
def test_forward_pass_produces_finite_output_of_expected_batch_size(factory, variant):
    model = factory.create(variant, {})
    model.eval()

    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, TIME_STEPS)
    with torch.no_grad():
        out = model(x)

    assert out.shape[0] == BATCH_SIZE, (
        f"{variant}: dimensión de batch inesperada {out.shape}"
    )
    assert torch.isfinite(out).all(), f"{variant}: salida contiene NaN/Inf"


def test_model_factory_filters_unknown_kwargs_or_absorbs_via_var_kwargs(factory):
    # MLP no acepta un kwarg "unknown_param" explícito ni **kwargs — el
    # factory debe filtrarlo silenciosamente (ver models/factory.py) en
    # vez de fallar con TypeError.
    model = factory.create("mlp", {"stage1_out": 16, "unknown_param": 999})
    assert model is not None
