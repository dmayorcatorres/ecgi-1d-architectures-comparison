# src/configs/base_config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# Fallback si configs/training/default.yaml no existe o no es legible.
# Debe mantenerse igual al valor de "active_default_variants" en ese YAML
# (ver docs/adr/0003-estrategia-configuracion.md) — es el mismo
# subconjunto de arquitecturas CNN que este archivo tenía hardcodeado
# antes de externalizar la configuración.
_FALLBACK_VARIANTS: Tuple[str, ...] = (
    "dilatedcnn", "disjointcnn", "disjointresnet", "fcn",
    "resnet", "resnetcnn", "tcn", "tfrcnn", "zhaocnn",
)


def _load_default_variants() -> Tuple[str, ...]:
    """Lee configs/training/default.yaml::active_default_variants.

    Si el archivo no existe, no es legible, o PyYAML no está instalado,
    devuelve _FALLBACK_VARIANTS (comportamiento idéntico al que tenía
    este módulo antes de externalizar la configuración a YAML).
    """
    yaml_path = Path(__file__).resolve().parents[2] / "configs" / "training" / "default.yaml"
    try:
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return tuple(data["active_default_variants"])
    except Exception:
        return _FALLBACK_VARIANTS


@dataclass
class TrainingConfig:
    # ─── Obligatorios (sin default) ──────────────────────────────────────
    data_path: str                       # 1º: va antes de cualquier campo con default

    # ─── HPO ─────────────────────────────────────────────────────────────
    hpo_backend: str = "ray"             # "ray" | "optuna"
    n_trials:    int  = 30              # nº de muestras por variante

    # ─── Entrenamiento / CV ─────────────────────────────────────────────
    batch_size:     int = 1
    n_splits:       int = 5
    final_epochs:   int = 200


    optuna_epochs:  int = 5
    optuna_trials:  int = 5

    # ─── Otros parámetros ───────────────────────────────────────────────
    verbose:         bool = False
    input_channels:  int  = 12
    output_channels: int  = 256
    time_steps:      int  = 256


    # ─── Variantes a evaluar ────────────────────────────────────────────
    # Leído desde configs/training/default.yaml::active_default_variants.
    # Para cambiar qué familia/subconjunto se evalúa por defecto, edita
    # ese YAML (no este archivo) — ver docs/adr/0003. El catálogo
    # completo por familia (CNN/RNN/híbridas/atención) está documentado
    # en configs/training/default.yaml::variants.
    variants: Tuple[str, ...] = field(default_factory=_load_default_variants)

