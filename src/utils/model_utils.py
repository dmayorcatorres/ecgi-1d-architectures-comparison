import time
import torch
import logging
from typing import Dict, Any
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

class ModelAnalyzer:
    """
    Analiza características de modelos: parámetros, FLOPs, latencia, memoria
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.has_thop = self._check_thop_availability()

    def analyze_model(self, model: torch.nn.Module, input_shape: tuple) -> Dict[str, Any]:
        """Analiza todas las características del modelo"""
        stats = {
            'parameters': self.count_parameters(model),
            'latency': self.measure_latency(model, input_shape),
            'gpu_memory': self.get_gpu_memory_usage()
        }
        
        if self.has_thop:
            stats.update(self.calculate_flops(model, input_shape))
        
        return stats

    def count_parameters(self, model: torch.nn.Module) -> int:
        """Cuenta los parámetros entrenables"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def measure_latency(self, model: torch.nn.Module, input_shape: tuple) -> float:
        """Mide latencia de inferencia en milisegundos"""
        dummy_input = torch.randn(input_shape).to(self.device)
        model.to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        
        return (time.time() - start_time) * 1000  # ms

    def calculate_flops(self, model: torch.nn.Module, input_shape: tuple) -> Dict[str, int]:
        """Calcula FLOPs y MACs (requiere thop)"""
        if not self.has_thop:
            return {}
            
        from thop import profile
        dummy_input = torch.randn(input_shape).to(self.device)
        flops, macs = profile(model, inputs=(dummy_input,), verbose=False)
        return {'flops': flops, 'macs': macs}

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Obtiene uso de memoria de GPU en MB"""
        return {
            'allocated': torch.cuda.memory_allocated() / 1e6,
            'reserved': torch.cuda.memory_reserved() / 1e6
        }

    def _check_thop_availability(self) -> bool:
        """Verifica si thop está instalado"""
        try:
            import thop  # noqa
            return True
        except ImportError:
            logger.warning("thop no instalado. FLOPs/MACs no disponibles.")
            return False