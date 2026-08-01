# utils/split.py
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, Tuple, List
from torch.utils.data import Subset

def stratified_split_indices_approx(labels: torch.Tensor, ratio: float, seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Split estratificado con 'largest remainder' para aproximar el ratio global.
    Asegura (cuando es posible) al menos 1 train y 1 val por clase.
    """
    rng = np.random.default_rng(seed)
    y = labels.cpu().numpy().astype(int)

    # Agrupa índices por clase y baraja intra-clase
    cls_to_idx = defaultdict(list)
    for i, c in enumerate(y):
        cls_to_idx[int(c)].append(i)
    for idxs in cls_to_idx.values():
        rng.shuffle(idxs)

    N = len(y)
    counts = {c: len(idxs) for c, idxs in cls_to_idx.items()}

    # Objetivo global de train
    desired_total = int(round(N * ratio))

    # Asignación inicial por clase (floor) y fracciones
    floors = {c: int(np.floor(n * ratio)) for c, n in counts.items()}
    fracs  = {c: n * ratio - floors[c]    for c, n in counts.items()}
    train_counts = floors.copy()
    total = sum(train_counts.values())

    # Garantiza al menos 1 train si la clase tiene >=2 ejemplos
    for c, n in counts.items():
        if n >= 2 and train_counts[c] == 0:
            train_counts[c] = 1
            total += 1

    # No dejes clase sin validación (train <= n-1)
    for c, n in counts.items():
        train_counts[c] = min(train_counts[c], max(n - 1, 0))

    # Reparte los +1 a las clases con mayor fracción hasta llegar al total deseado
    inc_needed = max(0, desired_total - total)
    if inc_needed > 0:
        order = sorted(fracs, key=lambda c: fracs[c], reverse=True)
        k = 0
        while inc_needed > 0 and k < len(order):
            c = order[k]
            if train_counts[c] < counts[c] - 1:  # deja al menos 1 para val
                train_counts[c] += 1
                inc_needed -= 1
            k += 1

    train_idx, val_idx = [], []
    for c, idxs in cls_to_idx.items():
        t = train_counts[c]
        train_idx.extend(idxs[:t])
        val_idx.extend(idxs[t:])

    rng.shuffle(train_idx); rng.shuffle(val_idx)
    return train_idx, val_idx


def verify_holdout(dm, expected_train_frac: float, tol: float = None) -> Dict:
    """
    Verifica que el hold-out:
      - usa EXACTAMENTE el 30% no visto de acuerdo al rango discreto alcanzable por clases
      - no hay solape entre train/val (via fingerprints)
    Si 'tol' se pasa, se usa |frac - expected| <= tol; si no, se valida contra el
    rango teórico [lower, upper] derivado de floors/ceils por clase.
    """
    # Tamaños reales
    n_train = len(dm.ds_train); n_val = len(dm.ds_val)
    N = n_train + n_val
    train_frac = n_train / max(N, 1)

    # Rango alcanzable por discreción (floors/ceils)
    labels = dm.get_all_labels().cpu().numpy()
    classes, counts = np.unique(labels, return_counts=True)
    floors = np.floor(counts * expected_train_frac).astype(int)
    ceils  = np.ceil(counts  * expected_train_frac).astype(int)

    # Asegura al menos 1 val por clase (si la clase tiene >=2)
    ceils = np.minimum(ceils, counts - 1)
    # Asegura al menos 1 train por clase (si la clase tiene >=2)
    floors = np.maximum(floors, (counts >= 2).astype(int))

    lower = floors.sum() / max(N, 1)
    upper = ceils.sum()  / max(N, 1)

    msg = (f"Split alcanzado={train_frac:.3f}, objetivo={expected_train_frac:.2f}, "
           f"rango discreto≈[{lower:.3f}, {upper:.3f}]")
    print(msg)

    if tol is None:
        assert lower - 1e-9 <= train_frac <= upper + 1e-9, \
            f"{msg} → fuera del rango alcanzable."
    else:
        assert abs(train_frac - expected_train_frac) <= tol, \
            f"{msg} con tol={tol}"

    # Fingerprints (ids) para detectar fugas por solape
    def _fp(subset):
        if isinstance(subset, Subset):
            ids = subset.indices
        else:
            # Dataset completo (no debería darse en hold-out)
            ids = list(range(len(subset)))
        return (len(ids), hash(tuple(sorted(map(int, ids)))))

    fp_train = _fp(dm.ds_train)
    fp_val   = _fp(dm.ds_val)

    # No deberían solaparse
    if isinstance(dm.ds_train, Subset) and isinstance(dm.ds_val, Subset):
        assert set(map(int, dm.ds_train.indices)).isdisjoint(
               set(map(int, dm.ds_val.indices))), "¡Solape entre train y val!"

    return dict(n_train=n_train, n_val=n_val, train_frac=train_frac,
                lower=lower, upper=upper, fp_train=fp_train, fp_val=fp_val)
