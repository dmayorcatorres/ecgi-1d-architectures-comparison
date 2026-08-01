import torch

from utils.split import stratified_split_indices_approx


def _make_labels(n_per_class=50, n_classes=3):
    return torch.cat([torch.full((n_per_class,), c) for c in range(n_classes)])


def test_split_is_reproducible_with_same_seed():
    labels = _make_labels()
    train1, val1 = stratified_split_indices_approx(labels, ratio=0.6, seed=42)
    train2, val2 = stratified_split_indices_approx(labels, ratio=0.6, seed=42)
    assert train1 == train2
    assert val1 == val2


def test_split_has_no_overlap_between_train_and_val():
    labels = _make_labels()
    train, val = stratified_split_indices_approx(labels, ratio=0.6, seed=42)
    assert set(train).isdisjoint(set(val))


def test_split_covers_all_indices_exactly_once():
    labels = _make_labels()
    train, val = stratified_split_indices_approx(labels, ratio=0.6, seed=42)
    assert sorted(train + val) == list(range(len(labels)))


def test_split_ratio_is_approximately_respected():
    labels = _make_labels(n_per_class=100, n_classes=3)  # N=300
    train, val = stratified_split_indices_approx(labels, ratio=0.6, seed=42)
    # "largest remainder" aproxima el ratio global; tolerancia de 1% (~3 muestras)
    assert abs(len(train) - 180) <= 3


def test_split_different_seeds_produce_different_partitions():
    labels = _make_labels()
    train_a, _ = stratified_split_indices_approx(labels, ratio=0.6, seed=1)
    train_b, _ = stratified_split_indices_approx(labels, ratio=0.6, seed=2)
    assert set(train_a) != set(train_b)
