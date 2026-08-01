import pytest

from config import DataConfig
from configs.base_config import TrainingConfig


def test_data_config_requires_all_fields():
    cfg = DataConfig(time_steps=256, input_channels=12, output_channels=256)
    assert cfg.time_steps == 256
    assert cfg.input_channels == 12
    assert cfg.output_channels == 256


def test_data_config_missing_required_field_raises():
    with pytest.raises(TypeError):
        DataConfig(time_steps=256, input_channels=12)  # falta output_channels


def test_training_config_defaults_match_documented_values():
    cfg = TrainingConfig(data_path="src/data/kitpvc/kitpvc.mat")
    # Valores verificados en src/configs/base_config.py y documentados en
    # configs/training/default.yaml — si cambian aquí, actualizar también
    # ese YAML (ver docs/adr/0003-estrategia-configuracion.md).
    assert cfg.hpo_backend == "ray"
    assert cfg.n_trials == 30
    assert cfg.n_splits == 5
    assert cfg.final_epochs == 200
    assert cfg.time_steps == 256
    assert len(cfg.variants) > 0


def test_training_config_variants_are_registered_models():
    from models.register import MODEL_REGISTRY

    cfg = TrainingConfig(data_path="src/data/kitpvc/kitpvc.mat")
    unknown = [v for v in cfg.variants if v not in MODEL_REGISTRY]
    assert not unknown, (
        f"TrainingConfig.variants contiene variantes no registradas en "
        f"MODEL_REGISTRY: {unknown}"
    )


def test_training_config_variants_come_from_yaml_and_match_fallback():
    """
    Ver docs/adr/0003-estrategia-configuracion.md: TrainingConfig.variants
    se lee de configs/training/default.yaml::active_default_variants, con
    un fallback hardcodeado que debe coincidir exactamente (para no
    alterar el comportamiento por defecto si el YAML llega a faltar).
    """
    from configs.base_config import _FALLBACK_VARIANTS

    cfg = TrainingConfig(data_path="src/data/kitpvc/kitpvc.mat")
    assert cfg.variants == _FALLBACK_VARIANTS


def test_training_config_falls_back_when_yaml_unreadable(monkeypatch):
    import builtins

    real_open = builtins.open

    def _raise_on_yaml(path, *args, **kwargs):
        if str(path).endswith("configs/training/default.yaml"):
            raise FileNotFoundError(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _raise_on_yaml)

    from configs.base_config import _FALLBACK_VARIANTS, _load_default_variants

    assert _load_default_variants() == _FALLBACK_VARIANTS
