import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requiere GPU (CUDA)")
    config.addinivalue_line("markers", "smoke: prueba de humo end-to-end, más lenta")


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA no disponible en este entorno")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture
def synthetic_signal_batch():
    """Batch sintético (B, C, T) representativo de una señal ECG/EMG."""
    torch.manual_seed(0)
    batch_size, channels, time_steps = 4, 12, 256
    return torch.randn(batch_size, channels, time_steps)
