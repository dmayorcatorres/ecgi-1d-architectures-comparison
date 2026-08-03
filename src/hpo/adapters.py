import ray
from ray import tune

def _float_grid(low: float, high: float, step: float):
    """Genera valores discretos [low, high] con paso 'step' (incluye high si cae justo)."""
    if step <= 0:
        raise ValueError(f"step debe ser > 0, llegó: {step}")
    n = int(round((high - low) / step))
    vals = [low + i * step for i in range(n + 1)]
    # recorte por si el redondeo se pasa un pelo
    vals = [v for v in vals if v <= high + 1e-12]
    return vals

def _to_tune(spec: tuple):
    """
    Convierte una especificación de espacio HPO al correspondiente parameter de Ray Tune.

    spec formatos:
      - ("choice", [opts])
      - ("int", low, high, step)
      - ("float", low, high)              # uniform continuous
      - ("float", low, high, step)        # quantized -> se discretiza con choice (BOHB-friendly)
      - ("loguniform", low, high)
      - ("fixed", v)
    """
    kind = spec[0]
    vals = spec[1:]

    if kind == "fixed":
        (v,) = vals
        return v

    if kind == "choice":
        (options,) = vals
        return tune.choice(options)

    if kind == "int":
        low, high, step = vals
        return tune.choice(list(range(low, high + 1, step)))

    if kind == "float":
        # BOHB + ConfigSpace NO se lleva bien con quniform (q=...)
        if len(vals) == 3:
            low, high, step = vals
            return tune.choice(_float_grid(low, high, step))
        elif len(vals) == 2:
            low, high = vals
            return tune.uniform(low, high)
        else:
            raise ValueError(f"Especificación float inválida: {spec}")

    if kind == "loguniform":
        low, high = vals
        return tune.loguniform(low, high)

    raise ValueError(f"Tipo de espacio HPO no soportado: {kind}")


def space_to_tune(space_dict: dict) -> dict:
    """
    Dado un diccionario de espacios HPO con tuplas, devuelve un diccionario
    con los Hyperparameters listos para Ray Tune.
    """
    return {hp_name: _to_tune(spec) for hp_name, spec in space_dict.items()}
