# ecgdl/data/datamodule.py
import pytorch_lightning as pl, torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, random_split

class ECGDataModule(pl.LightningDataModule):
    def __init__(self, mat_path, batch_size=128, time_steps=256, num_workers=8,
                 train_frac=0.7, cache=True):
        super().__init__()
        self.save_hyperparameters()

    # ───── carga única en CPU ──────────────────────────────
    def prepare_data(self):
        from .preprocessing import load_mat_cached

        self.ecg, self.emg = load_mat_cached(
            Path(self.hparams.mat_path),
            self.hparams.time_steps
        )

    # ───── se ejecuta en cada proceso DDP ─────────────────
    def setup(self, stage=None):
        ds = TensorDataset(self.ecg, self.emg)
        n_train = int(len(ds) * self.hparams.train_frac)
        n_val   = len(ds) - n_train
        self.ds_train, self.ds_val = random_split(ds, [n_train, n_val])

    # ───── DataLoaders con afinidad DDP / pin - mem ───────
    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
