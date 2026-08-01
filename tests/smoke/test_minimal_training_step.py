import pytest
import torch
import torch.nn as nn

from config import DataConfig
from models.factory import ModelFactory
from utils.runtime import set_seed


@pytest.mark.smoke
def test_minimal_training_loop_reduces_loss_on_synthetic_data():
    """
    Réplica mínima del ciclo de entrenamiento real de run_bohb.py
    (MSELoss + AdamW, ver docs/model_card.md) sobre datos sintéticos:
    no valida resultados científicos, solo que el ciclo
    forward -> loss -> backward -> step es funcional y reduce la pérdida.
    """
    set_seed(42)

    cfg = DataConfig(time_steps=32, input_channels=4, output_channels=32)
    model = ModelFactory(cfg).create("mlp", {"stage1_out": 16, "stage2_out": 16, "stage3_out": 16})

    x = torch.randn(8, 4, 32)
    y = torch.randn(8, 32, 32)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    model.train()
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0], "La pérdida no disminuyó tras 20 pasos de entrenamiento sintético"
