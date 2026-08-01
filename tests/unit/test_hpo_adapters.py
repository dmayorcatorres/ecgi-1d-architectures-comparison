import pytest
from ray import tune

from hpo.adapters import _float_grid, space_to_tune


def test_float_grid_covers_low_and_high():
    vals = _float_grid(0.0, 1.0, 0.25)
    assert vals[0] == pytest.approx(0.0)
    assert vals[-1] == pytest.approx(1.0)
    assert len(vals) == 5


def test_float_grid_invalid_step_raises():
    with pytest.raises(ValueError):
        _float_grid(0.0, 1.0, 0.0)


def test_space_to_tune_handles_all_spec_kinds():
    spec = {
        "a": ("fixed", 3),
        "b": ("choice", [1, 2, 3]),
        "c": ("int", 1, 5, 1),
        "d": ("float", 0.0, 1.0),
        "e": ("float", 0.0, 1.0, 0.5),
        "f": ("loguniform", 1e-5, 1e-1),
    }
    result = space_to_tune(spec)

    assert result["a"] == 3
    assert isinstance(result["b"], tune.search.sample.Categorical)
    assert isinstance(result["c"], tune.search.sample.Categorical)
    assert isinstance(result["d"], tune.search.sample.Float)
    assert isinstance(result["e"], tune.search.sample.Categorical)
    assert isinstance(result["f"], tune.search.sample.Float)


def test_space_to_tune_rejects_unknown_kind():
    with pytest.raises(ValueError):
        space_to_tune({"x": ("not_a_kind", 1, 2)})
