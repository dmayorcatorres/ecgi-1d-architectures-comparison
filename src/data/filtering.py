# data/filtering.py
#
# Filtro Butterworth (HP -> notch 50/60 Hz -> LP) para PhysMixFormer1D
# ("physformer"), portado de aux/physformer_standalone/
# physmixformer_standalone_v3.py::ButterFilter. Se aplica a la señal CRUDA
# (fs real del dataset), antes de la reducción PAA que hace el resto del
# pipeline -- filtrar después de PAA no tendría sentido físico (el
# Nyquist ya cambió). Solo se usa para "physformer"; el resto de las 36
# arquitecturas siguen sin filtrar (data/preprocessing.py::load_mat_cached),
# sin cambios de comportamiento para ellas.
from pathlib import Path
from typing import Dict, Tuple

import scipy.io as sio
import scipy.signal as sps
import torch

from data.preprocessing import _ensure_nct, paa_reduce

# fs real y corte de low-pass por dataset -- mismos valores validados en
# aux/physformer_standalone/physmixformer_standalone_v3.py::DATASETS.
# Solo auckland/bordeaux por ahora (los únicos con geometría confiable
# hoy -- ver data/geometry.py).
DATASET_FS: Dict[str, float] = {
    "auckland": 2048.0,
    "bordeaux": 2048.0,
}
DATASET_LP_CUT: Dict[str, float] = {
    "auckland": 500.0,
    "bordeaux": 500.0,
}


class ButterFilter:
    """HP 0.5 Hz (4º) -> notch 50 Hz -> notch 60 Hz -> LP fijo (6º), SOS."""
    def __init__(self, fs: float, lp_cut: float, hp_cut: float = 0.5,
                 notches: Tuple[float, ...] = (50.0, 60.0), notch_Q: float = 35.0):
        self.fs = fs
        self.hp_sos = sps.butter(4, hp_cut, btype="high", fs=fs, output="sos")
        self.lp_sos = sps.butter(6, min(lp_cut, fs / 2.0 * 0.98), btype="low", fs=fs, output="sos")
        self.notch_sos = []
        for f0 in notches:
            if f0 >= fs / 2.0:
                continue
            try:
                b_no, a_no = sps.iirnotch(f0, notch_Q, fs=fs)
            except TypeError:
                w0 = f0 / (fs / 2.0)
                b_no, a_no = sps.iirnotch(w0, notch_Q)
            self.notch_sos.append(sps.tf2sos(b_no, a_no))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, T)."""
        arr = x.detach().cpu().numpy()
        arr = sps.sosfiltfilt(self.hp_sos, arr, axis=-1)
        for sos in self.notch_sos:
            arr = sps.sosfiltfilt(sos, arr, axis=-1)
        arr = sps.sosfiltfilt(self.lp_sos, arr, axis=-1)
        arr = arr.copy()
        return torch.from_numpy(arr).to(x.device, dtype=x.dtype)


def load_filtered(mat_path: Path, time_steps: int, dataset_name: str,
                  key_x: str = "emg", key_y: str = "tag", key_bspm: str = "ecg"
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Espejo de data/preprocessing.py::load_mat_cached, pero filtrando la
    señal cruda ANTES de la reducción PAA. Sin cache en disco (se llama
    una sola vez por run_pipeline, no por trial BOHB -- igual que
    data/geometry.py::load_geometry)."""
    fs = DATASET_FS.get(dataset_name)
    lp_cut = DATASET_LP_CUT.get(dataset_name)
    if fs is None or lp_cut is None:
        raise KeyError(
            f"'{dataset_name}' no tiene fs/lp_cut registrados en DATASET_FS/"
            f"DATASET_LP_CUT -- agregar antes de usar el filtro para este dataset."
        )

    d = sio.loadmat(mat_path)
    ecg = torch.as_tensor(d[key_bspm]).float()
    emg = torch.as_tensor(d[key_x]).float()
    ecg = _ensure_nct(ecg)
    emg = _ensure_nct(emg)
    ecg = torch.nan_to_num(ecg)
    emg = torch.nan_to_num(emg)

    y_raw = torch.as_tensor(d[key_y]).long().squeeze()
    uniq = torch.unique(y_raw)
    remap = {int(u): i for i, u in enumerate(sorted(uniq.tolist()))}
    labels = torch.tensor([remap[int(v)] for v in y_raw])

    butter = ButterFilter(fs=fs, lp_cut=lp_cut)
    ecg = butter(ecg)
    emg = butter(emg)

    ecg = paa_reduce(ecg, time_steps)
    emg = paa_reduce(emg, time_steps)
    ecg = torch.nan_to_num(ecg)
    emg = torch.nan_to_num(emg)

    return ecg, emg, labels
