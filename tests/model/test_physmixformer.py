import torch

from data.geometry import GeometryPack, gaussian_knn_laplacian, pairwise_dist_3d
from models.atbased.PhysMixFormer1D import PhysMixFormer1D

BATCH_SIZE = 2
TIME_STEPS = 32
INPUT_CHANNELS = 4    # Et (torso)
OUTPUT_CHANNELS = 3   # Ee (corazón)


def _make_model(**kwargs):
    return PhysMixFormer1D(
        time_steps=TIME_STEPS,
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        d_model=16,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        **kwargs,
    )


def _make_synthetic_geometry() -> GeometryPack:
    """Malla sintética mínima (sin caras -> distancia euclídea, no
    geodésica) solo para ejercitar el camino con física en el test."""
    coords_torso = torch.randn(INPUT_CHANNELS, 3)
    coords_heart = torch.randn(OUTPUT_CHANNELS, 3)
    faces_torso = torch.zeros((0, 3), dtype=torch.long)
    faces_heart = torch.zeros((0, 3), dtype=torch.long)

    L_torso, deg_torso = gaussian_knn_laplacian(coords_torso, faces_torso, sigma=5.0, k=2)
    L_heart, deg_heart = gaussian_knn_laplacian(coords_heart, faces_heart, sigma=5.0, k=2)
    D_te = pairwise_dist_3d(coords_torso, coords_heart)

    return GeometryPack(
        coords_torso=coords_torso, faces_torso=faces_torso,
        coords_heart=coords_heart, faces_heart=faces_heart,
        L_torso=L_torso, deg_torso=deg_torso,
        L_heart=L_heart, deg_heart=deg_heart,
        D_te=D_te,
    )


def test_forward_without_geometry_returns_none_pi_and_finite_output():
    model = _make_model()
    model.eval()
    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, TIME_STEPS)

    with torch.no_grad():
        x_hat, Pi, A_eff = model(x)

    assert Pi is None and A_eff is None
    assert x_hat.shape == (BATCH_SIZE, OUTPUT_CHANNELS, TIME_STEPS)
    assert torch.isfinite(x_hat).all()


def test_forward_with_geometry_returns_row_stochastic_pi():
    model = _make_model()
    model.attach_geometry(_make_synthetic_geometry())
    model.eval()
    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, TIME_STEPS)

    with torch.no_grad():
        x_hat, Pi, A_eff = model(x)

    assert x_hat.shape == (BATCH_SIZE, OUTPUT_CHANNELS, TIME_STEPS)
    assert torch.isfinite(x_hat).all()
    assert Pi is not None and Pi.shape == (INPUT_CHANNELS, OUTPUT_CHANNELS)
    assert torch.allclose(Pi.sum(dim=-1), torch.ones(INPUT_CHANNELS), atol=1e-4), \
        "Pi debe ser fila-estocástica (softmax por fila)"
    assert A_eff is not None and torch.isfinite(A_eff).all()


def test_compute_loss_is_finite_and_differentiable_without_geometry():
    model = _make_model()
    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, TIME_STEPS)
    y = torch.randn(BATCH_SIZE, OUTPUT_CHANNELS, TIME_STEPS)

    pred, Pi, _ = model(x)
    loss = model.compute_loss(pred, y, x, Pi=Pi)

    assert torch.isfinite(loss)
    loss.backward()
    assert model.backbone.output_proj.weight.grad is not None


def test_compute_loss_is_finite_and_differentiable_with_geometry():
    model = _make_model()
    model.attach_geometry(_make_synthetic_geometry())
    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, TIME_STEPS)
    y = torch.randn(BATCH_SIZE, OUTPUT_CHANNELS, TIME_STEPS)

    pred, Pi, _ = model(x)
    loss = model.compute_loss(pred, y, x, Pi=Pi)

    assert torch.isfinite(loss)
    loss.backward()
    assert model.backbone.output_proj.weight.grad is not None
    # los 5 términos deben estar presentes en el combinador tras adjuntar geometría
    assert set(model.combiner.log_vars.keys()) == {"time", "spec", "corr", "geom", "smooth"}


def test_anneal_epoch_is_noop_without_physics():
    model = _make_model()
    model.anneal_epoch(1, 10)  # no debe lanzar excepción


def test_anneal_epoch_updates_warmup_alpha_with_physics():
    model = _make_model(physics_warmup_epochs=4)
    model.attach_geometry(_make_synthetic_geometry())

    model.anneal_epoch(1, 10)
    alpha_start = model.physics.warmup_alpha
    model.anneal_epoch(4, 10)
    alpha_end = model.physics.warmup_alpha

    assert alpha_start == 0.0
    assert alpha_end == 1.0
