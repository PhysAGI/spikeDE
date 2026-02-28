# Installation

spikeDE supports :simple-python: **Python 3.9 through 3.13**.

!!! note "Use a virtual environment"
    We **strongly recommend** installing spikeDE in an isolated virtual environment to avoid conflicts with system-wide packages. You can create one using [venv](https://docs.python.org/3/library/venv.html), :simple-anaconda: [Conda](https://docs.conda.io/), or :simple-docker: [Docker](https://www.docker.com/).

---

## Install Dependencies

spikeDE relies on several third-party libraries. Please follow the instructions below to install them.

### Install PyTorch

The primary dependency is :simple-pytorch: PyTorch. Since installation commands vary by platform and hardware, please refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for the most accurate instructions.

!!! tip "GPU Acceleration"
    To leverage **GPU acceleration** with spikeDE, ensure you install the **CUDA-enabled version** of PyTorch that matches your NVIDIA driver. The CPU-only version will work but will not utilize GPU resources.

### Additional Dependencies

spikeDE utilizes neural differential equation solvers requiring the following specific packages:

- **`torchdiffeq`** (for Ordinary Differential Equations - ODEs):
  ```sh
  pip install torchdiffeq
  ```

- **`torchfde`** (for Fractional Differential Equations - FDEs):
  ```sh
  pip install git+https://github.com/kangqiyu/torchfde.git
  ```

---

## Install spikeDE

Once the dependencies are ready, install spikeDE directly from the source repository:

```sh
pip install git+https://github.com/PhysAGI/spikeDE.git
```

!!! success "GPU Support Enabled"
    spikeDE automatically detects and utilizes **GPU acceleration** if a CUDA-enabled PyTorch installation and a compatible NVIDIA GPU are present. **No additional configuration is required**.