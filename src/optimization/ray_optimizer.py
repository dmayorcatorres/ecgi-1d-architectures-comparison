# src/optimization/ray_optimizer.py
import logging
import sys
import types
from typing import Dict, Any, List

import torch
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ignite.engine import Events                          # ← 1 línea nueva
from src.training.engines import TrainingEngine           # ← 1 línea nueva

from src.optimization.hpo_spaces import ray_search_space

logger = logging.getLogger(__name__)


class RayOptimizer:
    """
    Igual interfaz que `OptunaOptimizer`, pero con Ray Tune.
    Hace un mini-entrenamiento por trial y usa ASHA para quedarnos
    con la configuración que minimiza la loss.
    """
    # ───────────────────────────── init ─────────────────────────────
    # ───────────────────────────── init ─────────────────────────────
    def __init__(self, config, data_loaders, model_factory):
        # Guardar referencias para usarlas después  ←──  NUEVO
        self.cfg          = config
        self.data_loaders = data_loaders          # [train_ld, val_ld]
        self.factory      = model_factory
        self.device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Ray Tune arrastra dependencias opcionales ―nos las inventamos si faltan
        for missing in ("gymnasium", "tree"):
            if missing not in sys.modules:
                sys.modules[missing] = types.ModuleType(missing)

        # ⚠️  Arranca Ray si aún no está inicializado
        if not ray.is_initialized():                       # <── NUEVO ⬇︎
            ray.init(ignore_reinit_error=True,
                     log_to_driver=False)                  # ← sin runtime_env
            logger.info(f"Ray iniciado – recursos: {ray.available_resources()}")


    # src/optimization/ray_optimizer.py
    # ───────────────────────── _make_trainable ─────────────────────────
    def _make_trainable(self, variant: str):
        """
        Función objetivo para Ray Tune.
        - Decide el dispositivo (GPU/CPU) en tiempo de ejecución dentro del worker.
        - Usa la API moderna `ray.train.report`.
        - Envía DataLoaders y ModelFactory al Object Store con `ray.put()` para
        que los trials sólo pasen un handle y no serialicen 10 + MiB.
        - Entrena una sola época para estimar la loss.
        """
        # Objetos grandes → se ponen en el Object Store UNA sola vez
        train_ref   = ray.put(self.data_loaders[0])   # train_ld
        val_ref     = ray.put(self.data_loaders[1])   # val_ld
        factory_ref = ray.put(self.factory)           # ModelFactory

        loss_fn = torch.nn.MSELoss()

        def trainable(cfg):
            # Import local para la nueva API
            from ray import train

            # Recuperamos referencias compartidas
            train_ld = ray.get(train_ref)
            val_ld   = ray.get(val_ref)
            factory  = ray.get(factory_ref)

            # Dispositivo elegido en el worker
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Hiper-parámetros: lr separado del resto
            hp = {k: v for k, v in cfg.items() if k != "lr"}
            lr = cfg["lr"]

            # Modelo y optimizador
            model     = factory.create_model(variant, hp).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Entrenamiento express (1 epoch) con tu TrainingEngine
            trainer = TrainingEngine(model, optimizer, device, loss_fn)
            tr_eng, val_eng = trainer.create_engines()

            @tr_eng.on(Events.EPOCH_COMPLETED)
            def _validate(_):
                val_eng.run(val_ld, max_epochs=1)

            tr_eng.run(train_ld, max_epochs=1)

            # Reporte de la métrica con la nueva API
            train.report({"loss": val_eng.state.metrics["loss"]})

        return trainable



    # ───────────────────────── optimize ─────────────────────────
    def optimize(self, variants: List[str]) -> Dict[str, Dict[str, Any]]:
        best: Dict[str, Dict[str, Any]] = {}

        for v in variants:
            logger.info(f"[Ray] Buscando hiper-parámetros para «{v}» …")
            reporter  = CLIReporter(metric_columns=["loss", "training_iteration"])
            analysis = tune.run(
                self._make_trainable(v),
                config=ray_search_space(v),
                num_samples=self.cfg.n_trials,
                scheduler=ASHAScheduler(metric="loss", mode="min", max_t=10),
                resources_per_trial={"cpu": 1, "gpu": 0},
                verbose=0,
            )

            cfg            = analysis.get_best_config(metric="loss", mode="min")
            cfg[f"lr_{v}"] = cfg.pop("lr")
            cfg["Best Trial"] = ", ".join(f"{k}={v}" for k, v in cfg.items())
            best[v] = cfg
            logger.info(f"[Ray] mejor configuración para {v}: {cfg}")

        return best
