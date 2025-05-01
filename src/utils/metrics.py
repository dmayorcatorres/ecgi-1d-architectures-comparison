import torch
import logging
import math
from typing import Optional, Union, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class ECGMetricsCalculator:
    """
    Calcula métricas de evaluación para señales ECG/EMG
    """
    
    @staticmethod
    def compute_mae(targets: torch.Tensor, preds: torch.Tensor) -> float:
        """Mean Absolute Error (MAE) con desviación estándar"""
        errors = torch.abs(targets - preds)
        return errors.mean().item(), errors.std().item()

    @staticmethod
    def compute_mse(targets: torch.Tensor, preds: torch.Tensor) -> float:
        """Mean Squared Error (MSE) con desviación estándar"""
        squared_errors = (targets - preds)**2
        return squared_errors.mean().item(), squared_errors.std().item()

    @staticmethod
    def compute_rmse(targets: torch.Tensor, preds: torch.Tensor) -> float:
        """Root Mean Squared Error (RMSE)"""
        mse, _ = ECGMetricsCalculator.compute_mse(targets, preds)
        return math.sqrt(mse)

    @staticmethod
    def compute_cc(targets: torch.Tensor, preds: torch.Tensor) -> float:
        """Coeficiente de Correlación (CC)"""
        vx = targets - torch.mean(targets)
        vy = preds - torch.mean(preds)
        return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))).item()

    @staticmethod
    def complete_report(targets: torch.Tensor, preds: torch.Tensor) -> dict:
        """Reporte completo de métricas principales"""
        return {
            'MAE': ECGMetricsCalculator.compute_mae(targets, preds),
            'MSE': ECGMetricsCalculator.compute_mse(targets, preds),
            'RMSE': ECGMetricsCalculator.compute_rmse(targets, preds),
            'CC': ECGMetricsCalculator.compute_cc(targets, preds)
        }