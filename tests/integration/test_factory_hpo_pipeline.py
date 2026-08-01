import torch
from ray import tune

from config import DataConfig
from hpo.adapters import space_to_tune
from models.factory import ModelFactory
from models.register import SPACE_REGISTRY


def test_sampled_hyperparameters_instantiate_a_working_model():
    """
    Integra ModelFactory + SPACE_REGISTRY + hpo/adapters.py: toma el
    espacio de búsqueda real de una arquitectura, extrae un valor concreto
    de cada hiperparámetro (sin depender de ray.tune.run), construye el
    modelo con esos valores y verifica que corre un forward pass.
    """
    variant = "mlp"
    spec = SPACE_REGISTRY[variant]
    spec = spec() if callable(spec) else spec
    tune_space = space_to_tune(spec)

    # Toma un valor determinista de cada distribución para no depender de
    # Ray Tune en tiempo de ejecución del trial.
    params = {}
    for k, v in tune_space.items():
        if isinstance(v, tune.search.sample.Categorical):
            params[k] = v.categories[0]
        elif isinstance(v, tune.search.sample.Float):
            params[k] = float(v.lower)
        else:
            params[k] = v  # valor "fixed"

    cfg = DataConfig(time_steps=64, input_channels=4, output_channels=64)
    factory = ModelFactory(cfg)
    model = factory.create(variant, params)

    x = torch.randn(2, 4, 64)
    out = model(x)
    assert out.shape[0] == 2
    assert torch.isfinite(out).all()
