# utils/ray_helpers.py
import os
import json

def bytes_from_shm(target_frac: float = 0.75,
                   reserve_bytes: int = 256 * 1024 * 1024,
                   min_bytes: int = 256 * 1024 * 1024) -> int:
    """Calcula object_store_memory como fracción de /dev/shm."""
    try:
        st = os.statvfs("/dev/shm")
        shm_bytes = st.f_frsize * st.f_blocks
    except Exception:
        shm_bytes = 2 * 1024**3  # fallback 2 GiB
    max_usable = max(0, shm_bytes - reserve_bytes)
    target = int(shm_bytes * target_frac)
    return max(min_bytes, min(target, max_usable))

def spilling_config() -> dict:
    """Configura spilling a disco si hay ruta disponible."""
    candidates = []
    env_dir = os.environ.get("RAY_SPILL_DIR")
    if env_dir: candidates.append(env_dir)
    candidates += ["/ray_spill", "/tmp/ray_spill"]

    dirs = []
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            if os.path.isdir(d):
                dirs.append(d)
        except Exception:
            continue

    if not dirs:
        return {}
    return {
        "object_spilling_config": json.dumps({
            "type": "filesystem",
            "params": {"directory_path": dirs}
        })
    }
