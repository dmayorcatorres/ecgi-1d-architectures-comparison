# models/factory.py

import inspect
from typing import Dict, Any
from config import DataConfig
from models.register import MODEL_REGISTRY

class ModelFactory:
    def __init__(self, data_cfg: DataConfig):
        # data_cfg tiene los atributos time_steps, input_channels, output_channels
        self.common: Dict[str, Any] = {
            "time_steps":    data_cfg.time_steps,
            "input_channels":data_cfg.input_channels,
            "output_channels":data_cfg.output_channels,
        }

    def create(self, variant: str, params: Dict[str, Any]):
        """Fusiona common + params, filtra args válidos y construye el modelo."""
        cls = MODEL_REGISTRY[variant.lower()]
        # combina parámetros
        init_args = {**self.common, **params}
        # filtrar según la firma de __init__
        sig = inspect.signature(cls.__init__)
        filtered = {k: v for k, v in init_args.items()
                    if k in sig.parameters}
        return cls(**filtered)
