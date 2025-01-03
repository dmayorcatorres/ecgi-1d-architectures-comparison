import numpy as np

class EarlyStopping:
    """
    Clase para aplicar Early Stopping en GANs.
    Detiene el entrenamiento si no hay mejoras en las métricas.
    """
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, current_loss):
        """
        Verifica si se cumple la condición de Early Stopping.

        Args:
            current_loss (float): Valor actual de la pérdida.

        Returns:
            bool: True si se debe detener el entrenamiento, False en caso contrario.
        """
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            if self.verbose:
                print(f"✅ EarlyStopping: Mejora detectada. Nueva mejor pérdida: {self.best_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠️ EarlyStopping: Sin mejora. Contador: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("🛑 EarlyStopping: Se ha detenido el entrenamiento por falta de mejora.")

        return self.early_stop