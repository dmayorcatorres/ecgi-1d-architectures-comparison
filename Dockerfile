# Imagen base: se conserva la misma que ya usaba el proyecto (no se cambia
# la versión de CUDA/PyTorch sin validación explícita del equipo, ya que
# afecta reproducibilidad de resultados en GPU). Ver docs/adr/0001-framework.md
# y docs/reproducibility.md.
# TODO: evaluar y documentar una migración a una versión soportada más
# reciente de la imagen NGC de PyTorch (22.05 ya no recibe actualizaciones
# de seguridad).
FROM nvcr.io/nvidia/pytorch:22.05-py3

# Metadatos de la imagen (no reemplazan LICENSE ni CITATION.cff).
LABEL org.opencontainers.image.title="ecgi-1d-architectures-comparison" \
      org.opencontainers.image.description="Comparación de arquitecturas 1D de deep learning para regresión ECG<->EMG en ECGi" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /workspace

# Instala primero las dependencias declaradas en requirements.txt para
# aprovechar la caché de capas de Docker (evita reinstalar todo el entorno
# cada vez que cambia el código fuente en src/).
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia el resto del proyecto. Los datos, checkpoints y artefactos pesados
# quedan fuera de la imagen (ver .dockerignore) — deben montarse como
# volumen en tiempo de ejecución (ver compose.yaml).
COPY . /workspace

# Usuario no root: mejora de seguridad recomendada para contenedores que
# procesan datos y ejecutan código de terceros (modelos/checkpoints).
RUN groupadd --gid 1000 ecgi && \
    useradd --uid 1000 --gid ecgi --shell /bin/bash --create-home ecgi && \
    chown -R ecgi:ecgi /workspace
USER ecgi

# Comprobación de salud mínima: valida que el entorno Python puede importar
# las dependencias críticas del pipeline (no requiere GPU ni datos).
HEALTHCHECK --interval=5m --timeout=30s --start-period=30s \
    CMD python -c "import torch, ray, numpy; print('ok')" || exit 1

CMD ["/bin/bash"]
