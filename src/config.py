from dataclasses import dataclass

@dataclass
class DataConfig:
    """Parámetros comunes a *todas* las arquitecturas."""
    time_steps: int
    input_channels: int
    output_channels: int
