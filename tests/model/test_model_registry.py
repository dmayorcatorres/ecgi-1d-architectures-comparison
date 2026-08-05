import torch.nn as nn

from models.register import MODEL_REGISTRY, SPACE_REGISTRY

# Lista verificada por auditoría (grep sobre src/models/register.py) — si
# se añade/quita una arquitectura, este test debe actualizarse junto con
# docs/architecture.md y configs/model/default.yaml.
EXPECTED_FAMILIES = {
    "cnnbased": {"dilatedcnn", "disjointcnn", "disjointresnet", "fcn",
                 "inceptiontime", "lite", "mlp", "multichanneldcnn",
                 "multiscalecnn", "resnet", "resnetcnn", "tcn", "tfrcnn",
                 "zhaocnn"},
    "rnnbased": {"esn", "indrnn", "peepholelstm", "qrnn", "tlgru",
                 "vsgrunet", "rnn_gru", "rnn_lstm"},
    "hybridbased": {"deepresidualbilstm", "cnnlstm", "cnngru", "cnnlstmgru",
                     "disjointlstm", "lstmfcn", "cnnbilstmbigrudil"},
    "atbased": {"alstm", "transformer1d", "informer1d", "autoformer1d",
                "patchtst", "vit1d", "timenet1d", "physformer"},
}


def test_model_registry_contains_all_expected_variants():
    expected = set().union(*EXPECTED_FAMILIES.values())
    missing = expected - MODEL_REGISTRY.keys()
    assert not missing, f"Variantes esperadas ausentes de MODEL_REGISTRY: {missing}"


def test_model_registry_has_exactly_37_variants():
    assert len(MODEL_REGISTRY) == 37


def test_all_registered_models_are_nn_module_subclasses():
    for name, cls in MODEL_REGISTRY.items():
        assert issubclass(cls, nn.Module), f"{name} no es un nn.Module"


def test_every_registered_model_has_a_search_space():
    missing = set(MODEL_REGISTRY) - set(SPACE_REGISTRY)
    assert not missing, (
        f"Modelos registrados sin espacio de búsqueda en SPACE_REGISTRY: {missing}"
    )


def test_search_spaces_are_dict_or_callable():
    for name, spec in SPACE_REGISTRY.items():
        resolved = spec() if callable(spec) else spec
        assert isinstance(resolved, dict), f"Espacio de '{name}' no resuelve a dict"
        assert len(resolved) > 0, f"Espacio de '{name}' está vacío"
