# Use a CUDA-enabled base image compatible with Ubuntu 22.04
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the PYTHONPATH for your workspace structure
ENV PYTHONPATH=/workspace:${PYTHONPATH}

# Minimal OS deps (plus libgl/glib just in case)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-dev python3-pip python3-venv build-essential tmux \
      git curl wget ca-certificates \
      libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install the PyTorch stack first from the matching CUDA 12.1 index
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
RUN python -m pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} \
      torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Then install the rest from PyPI
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

WORKDIR /workspace

# Optional: build-time sanity check so bad combos fail fast
RUN python - <<'PY'
import numpy, torch, torchvision, torchaudio, PIL, cv2
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", getattr(torch.version,'cuda',None), "GPU?", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("torchaudio", torchaudio.__version__)
print("Pillow", PIL.__version__)
print("cv2", cv2.__version__)
import numpy as np
t = torch.zeros(2,2)
assert isinstance(t.numpy(), np.ndarray)
PY

CMD ["/bin/bash"]