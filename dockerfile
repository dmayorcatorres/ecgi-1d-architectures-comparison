# Usa la imagen de PyTorch como base
FROM nvcr.io/nvidia/pytorch:22.05-py3

# Instala paquetes adicionales, si es necesario
RUN pip install --upgrade pip && \
    pip install numpy matplotlib

# Copia archivos de tu proyecto al contenedor
COPY . /workspace

# Configura el contenedor para que arranque con Bash
CMD ["/bin/bash"]
