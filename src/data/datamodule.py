# data/datamodule.py
import pytorch_lightning as pl
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, Subset

from utils.split import stratified_split_indices_approx


class ECGDataModule(pl.LightningDataModule):
    """
    direction:
      - "ecg2emg" => xb=ecg, yb=emg
      - "emg2ecg" => xb=emg, yb=ecg   (INVERSO real)
    """
    def __init__(
        self,
        mat_path,
        batch_size=128,
        time_steps=256,
        num_workers=8,
        train_frac=0.7,
        cache=True,
        key_x="emg",
        key_y="tag",
        key_bspm="ecg",
        split_seed: int = 42,
        direction: str = "ecg2emg",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.split_seed = split_seed
        self.direction = direction

        self.labels = None
        self.ecg = None
        self.emg = None

        self.x = None
        self.y = None

    def prepare_data(self):
        from .preprocessing import load_mat_cached

        # Ojo: respetamos tu loader actual
        self.ecg, self.emg, self.labels = load_mat_cached(
            Path(self.hparams.mat_path),
            self.hparams.time_steps,
            key_x=self.hparams.key_x,
            key_y=self.hparams.key_y,
            key_bspm=self.hparams.key_bspm,
        )

        # define dirección real del aprendizaje
        if self.direction.lower() == "ecg2emg":
            self.x = self.ecg
            self.y = self.emg
        elif self.direction.lower() == "emg2ecg":
            self.x = self.emg
            self.y = self.ecg
        else:
            raise ValueError(f"direction inválida: {self.direction} (usa 'ecg2emg' o 'emg2ecg')")

        # DEBUG / sanity
        print("\n[ECGDataModule] Loaded tensors:")
        print("  ecg   :", tuple(self.ecg.shape), self.ecg.dtype)
        print("  emg   :", tuple(self.emg.shape), self.emg.dtype)
        print("  labels:", tuple(self.labels.shape), self.labels.dtype)
        print(f"  direction={self.direction} -> x={tuple(self.x.shape)} | y={tuple(self.y.shape)}")

        # checks rápidos
        assert self.x.ndim == 3 and self.y.ndim == 3, "x/y deben ser 3D (N,C,T)"
        assert self.labels.ndim in (1, 2), "labels debe ser (N,) o (N,1)"
        assert self.x.shape[0] == self.y.shape[0], "N beats no coincide entre x e y"

    def setup(self, stage=None):
        ds = TensorDataset(self.x, self.y, self.labels)

        if self.hparams.train_frac == 1.0:
            self.ds_train = ds
            self.ds_val = Subset(ds, [])
            return

        train_idx, val_idx = stratified_split_indices_approx(
            self.labels,
            self.hparams.train_frac,
            seed=self.split_seed,
        )
        self.ds_train = Subset(ds, train_idx)
        self.ds_val   = Subset(ds, val_idx)

    def train_dataloader(self):
        nw = self.hparams.num_workers
        return DataLoader(
            self.ds_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw > 0),
            prefetch_factor=4 if nw > 0 else None,
        )

    def val_dataloader(self):
        nw = self.hparams.num_workers
        return DataLoader(
            self.ds_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw > 0),
            prefetch_factor=4 if nw > 0 else None,
        )

    def get_all_labels(self):
        # verify_holdout() espera esto
        return self.labels

    def get_full_dataset(self):
        # por si lo usas en otras partes / debugging
        return TensorDataset(self.x, self.y, self.labels)

