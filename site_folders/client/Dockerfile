# No-apt Dockerfile for rootless Podman builds (DGX-friendly)
# Uses a PyTorch CUDA base that already includes Python + conda.
FROM docker.io/pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Non-interactive + sane Python defaults
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Workspace + PYTHONPATH
WORKDIR /workspace
ENV PYTHONPATH=/workspace:${PYTHONPATH}

# --------------------------------------------------------------------
# 1) System-like tools without apt: use conda-forge (no root needed)
# --------------------------------------------------------------------
# Note: build-essential isn't needed unless you compile native extensions.
# Many wheels exist, but pybind11 + some packages *may* compile if wheels missing.
# conda-forge provides compilers as: compilers / gxx_linux-64 / gcc_linux-64.
RUN conda config --set channel_priority strict && \
    conda install -y -c conda-forge \
      # tmux \
      git \
      curl \
      wget \
      ca-certificates \
      openssl \
      # optional: compilers (safer on HPC where some wheels may not match)
      compilers \
      make \
      cmake \
      && \
    conda clean -a -y

# --------------------------------------------------------------------
# 2) Pip tooling (still no apt)
# --------------------------------------------------------------------
RUN python -m pip install --upgrade pip setuptools wheel

# --------------------------------------------------------------------
# 3) Your Python deps
# --------------------------------------------------------------------
RUN python -m pip install --no-cache-dir \
      numpy \
      pybind11 \
      opencv-python-headless \
      pillow \
      scipy \
      h5py \
      imageio \
      pandas \
      tqdm \
      matplotlib \
      jupyter \
      "nvflare==2.6.2" \
      scikit-learn

# --------------------------------------------------------------------
# 4) Sanity check
# --------------------------------------------------------------------
RUN python -c "import numpy as np, torch, torchvision, torchaudio, PIL, cv2; \
print('numpy', np.__version__); \
print('torch', torch.__version__, 'cuda', getattr(torch.version, 'cuda', None), 'GPU?', torch.cuda.is_available()); \
print('torchvision', torchvision.__version__); \
print('torchaudio', torchaudio.__version__); \
print('Pillow', PIL.__version__); \
print('cv2', cv2.__version__); \
t = torch.zeros(2, 2); \
assert isinstance(t.numpy(), np.ndarray)"

# --------------------------------------------------------------------
# 5) Startup script
# --------------------------------------------------------------------
# COPY start_fl_client.sh /usr/local/bin/start_fl_client.sh
# RUN chmod +x /usr/local/bin/start_fl_client.sh

# CMD ["/usr/local/bin/start_fl_client.sh"]

CMD ["bash", "-lc", "echo 'Container ready. Exec in and start NVFLARE manually.'; tail -f /dev/null"]