import os
import random
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

try:
    import hydra
    from hydra import utils as hydra_utils
    from omegaconf import OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from ignite.engine import Engine, Events
from ignite.metrics import Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from torch.cuda.amp import GradScaler, autocast

# Optional advanced precision
try:
    from apex import amp
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False

# Optional experiment tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
    model: Dict[str, Any]
    train_loader: Dict[str, Any]
    val_loader: Dict[str, Any]
    optimizer: Dict[str, Any]
    scheduler: Optional[Dict[str, Any]] = None
    criterion: str = "MSELoss"
    batch_size: int = 32
    accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    use_amp: bool = True
    use_apex: bool = False
    metrics: list = field(default_factory=lambda: ["loss"])
    early_stopping: Dict[str, Any] = field(default_factory=lambda: {"patience":5, "min_delta":0.0})
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    seed: int = 42
    distributed: bool = False
    resume_from: Optional[str] = None
    epochs: int = 10

class TrainingEngine:
    def __init__(self, cfg_or_model, *args):
        """
        Inicializa el motor de entrenamiento. Soporta dos firmas:
        1. Legacy: (model, optimizer, device, criterion)
        2. Nuevo: (cfg: TrainConfig)
        """
        # LEGACY SIGNATURE
        if not isinstance(cfg_or_model, TrainConfig):
            model = cfg_or_model
            optimizer = args[0] if len(args) > 0 else None
            device = args[1] if len(args) > 1 else torch.device('cpu')
            criterion = args[2] if len(args) > 2 else nn.MSELoss()
            self.cfg = None
            self.model = model.to(device)
            self.optimizer = optimizer
            self.device = device
            self.criterion = criterion
            self.scheduler = None
            self.use_amp = False
            self.use_apex = False
            self.scaler = None
            return
        # NEW SIGNATURE WITH TrainConfig
        cfg: TrainConfig = cfg_or_model
        self.cfg = cfg
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if cfg.distributed:
            torch.distributed.init_process_group(backend='nccl')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if HYDRA_AVAILABLE:
            self.model = hydra_utils.instantiate(cfg.model).to(self.device)
        else:
            raise ImportError("Hydra required for TrainConfig instantiation")
        if cfg.distributed:
            self.model = DDP(self.model, device_ids=[torch.cuda.current_device()])
        if HYDRA_AVAILABLE:
            self.optimizer = hydra_utils.instantiate(cfg.optimizer, params=self.model.parameters())
            self.scheduler = hydra_utils.instantiate(cfg.scheduler) if cfg.scheduler else None
        self.use_amp = cfg.use_amp and torch.cuda.is_available()
        self.use_apex = cfg.use_apex and APEX_AVAILABLE
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_apex:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')

    def create_engines(self):
        """
        Crea y devuelve los Ignite engines de entrenamiento y evaluación.
        Adjunta Loss en modo legacy.
        """
        train_engine = Engine(self._train_step)
        val_engine = Engine(self._eval_step)
        if self.cfg is None:
            Loss(self.criterion).attach(val_engine, 'loss')
        return train_engine, val_engine

    def _train_step(self, engine, batch):
        state = engine.state
        self.model.train()
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        with autocast(enabled=self.use_amp):
            outputs = self.model(inputs)
            if self.cfg is None:
                loss = self.criterion(outputs, targets).mean()
            else:
                loss_fn = getattr(nn, self.cfg.criterion)()
                loss = loss_fn(outputs, targets).sum() / inputs.size(0)
        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as sl:
                sl.backward()
        else:
            loss.backward()
        if state.iteration % (self.cfg.accumulation_steps if self.cfg else 1) == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm if self.cfg else 1.0)
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.cfg and self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
        return {'loss': loss.item()}

    def _eval_step(self, engine, batch):
        """
        Evaluación por batch.
        En legacy y nuevo pipeline, devuelve (y_pred, y).
        """
        self.model.eval()
        with torch.no_grad():
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
        # Devuelve predicción y targets para métrica Loss
        return outputs, targets

    def setup(self):
        if self.cfg is None:
            return
        def worker_init_fn(wid):
            np.random.seed(self.cfg.seed + wid)
            random.seed(self.cfg.seed + wid)
        self.train_loader = hydra_utils.instantiate(self.cfg.train_loader, batch_size=self.cfg.batch_size, worker_init_fn=worker_init_fn)
        self.val_loader = hydra_utils.instantiate(self.cfg.val_loader, batch_size=self.cfg.batch_size, worker_init_fn=worker_init_fn)
        self.trainer, self.evaluator = self.create_engines()
        RunningAverage(output_transform=lambda out: out['loss']).attach(self.trainer, 'avg_loss')
        Loss(getattr(nn, self.cfg.criterion)()).attach(self.evaluator, 'val_loss')
        tb = TensorboardLogger(log_dir=self.cfg.log_dir)
        tb.attach(
            self.trainer,
            log_handler=TensorboardLogger.OutputHandler(
                tag='train', metric_names=['avg_loss'],
                global_step_transform=global_step_from_engine(self.trainer)
            ),
            event_name=Events.ITERATION_COMPLETED
        )
        tb.attach(
            self.evaluator,
            log_handler=TensorboardLogger.OutputHandler(
                tag='val', metric_names=['val_loss'],
                global_step_transform=global_step_from_engine(self.trainer)
            ),
            event_name=Events.COMPLETED
        )
        cp = ModelCheckpoint(self.cfg.checkpoint_dir, 'chkpt', n_saved=5, require_empty=False, save_as_state_dict=True)
        self.trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            cp,
            {'model': self.model, 'optimizer': self.optimizer, 'scheduler': self.scheduler, 'scaler': self.scaler}
        )
        if self.cfg.resume_from:
            cp.load_objects(
                load_path=self.cfg.resume_from,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler
            )
        def score_fn(engine):
            return -engine.state.metrics['val_loss']
        es = EarlyStopping(
            patience=self.cfg.early_stopping['patience'],
            score_function=score_fn,
            trainer=self.trainer,
            min_delta=self.cfg.early_stopping['min_delta']
        )
        self.evaluator.add_event_handler(Events.COMPLETED, es)

    def run(self):
        """
        Ejecuta el pipeline completo: setup y entrenamiento con validación.
        """
        if self.cfg is None:
            raise RuntimeError("Run requires TrainConfig for full pipeline")
        self.setup()
        @self.trainer.on(Events.EPOCH_COMPLETED)
        def do_eval(engine):
            self.evaluator.run(self.val_loader)
        self.trainer.run(self.train_loader, max_epochs=self.cfg.epochs)

# CLI entrypoint
if __name__ == '__main__':
    if HYDRA_AVAILABLE:import os
import random
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

try:
    import hydra
    from hydra import utils as hydra_utils
    from omegaconf import OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from ignite.engine import Engine, Events
from ignite.metrics import Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from torch.cuda.amp import GradScaler, autocast

# Optional advanced precision
try:
    from apex import amp
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False

# Optional experiment tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
    model: Dict[str, Any]
    train_loader: Dict[str, Any]
    val_loader: Dict[str, Any]
    optimizer: Dict[str, Any]
    scheduler: Optional[Dict[str, Any]] = None
    criterion: str = "MSELoss"
    batch_size: int = 32
    accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    use_amp: bool = True
    use_apex: bool = False
    metrics: list = field(default_factory=lambda: ["loss"])
    early_stopping: Dict[str, Any] = field(default_factory=lambda: {"patience":5, "min_delta":0.0})
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    seed: int = 42
    distributed: bool = False
    resume_from: Optional[str] = None
    epochs: int = 10

class TrainingEngine:
    def __init__(self, cfg_or_model, *args):
        """
        Inicializa el motor de entrenamiento. Soporta dos firmas:
        1. Legacy: (model, optimizer, device, criterion)
        2. Nuevo: (cfg: TrainConfig)
        """
        # LEGACY SIGNATURE
        if not isinstance(cfg_or_model, TrainConfig):
            model = cfg_or_model
            optimizer = args[0] if len(args) > 0 else None
            device = args[1] if len(args) > 1 else torch.device('cpu')
            criterion = args[2] if len(args) > 2 else nn.MSELoss()
            self.cfg = None
            self.model = model.to(device)
            self.optimizer = optimizer
            self.device = device
            self.criterion = criterion
            self.scheduler = None
            self.use_amp = False
            self.use_apex = False
            self.scaler = None
            return
        # NEW SIGNATURE WITH TrainConfig
        cfg: TrainConfig = cfg_or_model
        self.cfg = cfg
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if cfg.distributed:
            torch.distributed.init_process_group(backend='nccl')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if HYDRA_AVAILABLE:
            self.model = hydra_utils.instantiate(cfg.model).to(self.device)
        else:
            raise ImportError("Hydra required for TrainConfig instantiation")
        if cfg.distributed:
            self.model = DDP(self.model, device_ids=[torch.cuda.current_device()])
        if HYDRA_AVAILABLE:
            self.optimizer = hydra_utils.instantiate(cfg.optimizer, params=self.model.parameters())
            self.scheduler = hydra_utils.instantiate(cfg.scheduler) if cfg.scheduler else None
        self.use_amp = cfg.use_amp and torch.cuda.is_available()
        self.use_apex = cfg.use_apex and APEX_AVAILABLE
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_apex:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')

    def create_engines(self):
        """
        Crea y devuelve los Ignite engines de entrenamiento y evaluación.
        Adjunta Loss en modo legacy.
        """
        train_engine = Engine(self._train_step)
        val_engine = Engine(self._eval_step)
        if self.cfg is None:
            Loss(self.criterion).attach(val_engine, 'loss')
        return train_engine, val_engine

    def _train_step(self, engine, batch):
        state = engine.state
        self.model.train()
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()

        with autocast(enabled=self.use_amp):
            outputs = self.model(inputs)

            # ——> parche: recortamos a la longitud mínima
            if outputs.size(-1) != targets.size(-1):
                M = min(outputs.size(-1), targets.size(-1))
                outputs = outputs[..., :M]
                targets = targets[..., :M]

            # cálculo de la pérdida
            if self.cfg is None:
                loss = self.criterion(outputs, targets).mean()
            else:
                loss_fn = getattr(nn, self.cfg.criterion)()
                loss = loss_fn(outputs, targets).sum() / inputs.size(0)

        # backward + gradient accumulation / clipping / step
        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as sl:
                sl.backward()
        else:
            loss.backward()

        if state.iteration % (self.cfg.accumulation_steps if self.cfg else 1) == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.cfg.max_grad_norm if self.cfg else 1.0)
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.cfg and self.scheduler and not isinstance(self.scheduler,
                                                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()

        return {'loss': loss.item()}

    def _eval_step(self, engine, batch):
        """
        Evaluación por batch.
        En legacy y nuevo pipeline, devuelve (y_pred, y), recortados si hiciera falta.
        """
        self.model.eval()
        with torch.no_grad():
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)

            # ——> parche: recortamos a la longitud mínima
            if outputs.size(-1) != targets.size(-1):
                M = min(outputs.size(-1), targets.size(-1))
                outputs = outputs[..., :M]
                targets = targets[..., :M]

        return outputs, targets

    def setup(self):
        if self.cfg is None:
            return
        def worker_init_fn(wid):
            np.random.seed(self.cfg.seed + wid)
            random.seed(self.cfg.seed + wid)
        self.train_loader = hydra_utils.instantiate(self.cfg.train_loader, batch_size=self.cfg.batch_size, worker_init_fn=worker_init_fn)
        self.val_loader = hydra_utils.instantiate(self.cfg.val_loader, batch_size=self.cfg.batch_size, worker_init_fn=worker_init_fn)
        self.trainer, self.evaluator = self.create_engines()
        RunningAverage(output_transform=lambda out: out['loss']).attach(self.trainer, 'avg_loss')
        Loss(getattr(nn, self.cfg.criterion)()).attach(self.evaluator, 'val_loss')
        tb = TensorboardLogger(log_dir=self.cfg.log_dir)
        tb.attach(
            self.trainer,
            log_handler=TensorboardLogger.OutputHandler(
                tag='train', metric_names=['avg_loss'],
                global_step_transform=global_step_from_engine(self.trainer)
            ),
            event_name=Events.ITERATION_COMPLETED
        )
        tb.attach(
            self.evaluator,
            log_handler=TensorboardLogger.OutputHandler(
                tag='val', metric_names=['val_loss'],
                global_step_transform=global_step_from_engine(self.trainer)
            ),
            event_name=Events.COMPLETED
        )
        cp = ModelCheckpoint(self.cfg.checkpoint_dir, 'chkpt', n_saved=5, require_empty=False, save_as_state_dict=True)
        self.trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            cp,
            {'model': self.model, 'optimizer': self.optimizer, 'scheduler': self.scheduler, 'scaler': self.scaler}
        )
        if self.cfg.resume_from:
            cp.load_objects(
                load_path=self.cfg.resume_from,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler
            )
        def score_fn(engine):
            return -engine.state.metrics['val_loss']
        es = EarlyStopping(
            patience=self.cfg.early_stopping['patience'],
            score_function=score_fn,
            trainer=self.trainer,
            min_delta=self.cfg.early_stopping['min_delta']
        )
        self.evaluator.add_event_handler(Events.COMPLETED, es)

    def run(self):
        """
        Ejecuta el pipeline completo: setup y entrenamiento con validación.
        """
        if self.cfg is None:
            raise RuntimeError("Run requires TrainConfig for full pipeline")
        self.setup()
        @self.trainer.on(Events.EPOCH_COMPLETED)
        def do_eval(engine):
            self.evaluator.run(self.val_loader)
        self.trainer.run(self.train_loader, max_epochs=self.cfg.epochs)
        # Verificar early stopping
        logger.info(f"Training finished at epoch {self.trainer.state.epoch}")
        if self.trainer.state.epoch < self.cfg.epochs:
            logger.info("EarlyStopping triggered: stopped before max epochs")