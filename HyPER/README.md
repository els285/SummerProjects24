# HyPER setup 

Solution 18th July

Build a conda environment using the exact specification below.
This only works on a CPU node, not the login node, so you need to access such a
node:

```bash
qrsh -l short
conda env create -f new_env.yaml
```
where `new_env.yaml` contains:
```yaml
name: HyPER
channels:
  - pytorch
  - nvidia
dependencies:
  - python=3.10.*
  - pip
  - pytorch=2.3.*
  - pytorch-cuda=11.8
  - torchvision
  - torchaudio
  - cudatoolkit=11.8
  - pip:
    - torch_geometric
    - https://data.pyg.org/whl/torch-2.3.0%2Bcu118/torch_scatter-2.1.2%2Bpt23cu118-cp310-cp310-linux_x86_64.whl
    - numpy==1.26.4
    - pandas
    - lightning
    - pytorch-lightning
    - tensorboard
    - jupyterlab
    - uproot
    - awkward
    - h5py
    - torch-hep==0.0.4
    - hydra-core
    - rich
    - onnx
    - onnxruntime
    - onnxscript
```
Note that you may already have an environment named `HyPER` created already,
which you will need to delete, or change the name of this one.