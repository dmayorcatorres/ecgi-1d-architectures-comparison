import ray
from ray import tune

def _to_tune(spec: tuple):
    """
    Convierte una especificación de espacio HPO al correspondiente parameter de Ray Tune.

    spec formatos:
      - ("choice", [opts])
      - ("int", low, high, step)
      - ("float", low, high)              # uniform continuous
      - ("float", low, high, step)       # quantized uniform
      - ("loguniform", low, high)
    """
    kind = spec[0]
    vals = spec[1:]
    if kind == "choice":
        (options,) = vals
        return tune.choice(options)

    if kind == "int":
        # tunable ints with step
        low, high, step = vals
        # Ray Tune no soporta step directo en rand(); usamos choice
        return tune.choice(list(range(low, high + 1, step)))

    if kind == "float":
        # si viene step, usamos quniform, sino uniform
        if len(vals) == 3:
            low, high, step = vals
            return tune.quniform(low, high, step)
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
