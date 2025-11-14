# PhysNet-TF2
TensorFlow 2 PhysNet implementation

Patching of PhysNet for Tensorflow 2.

Git repository for original (TensorFlow 1) version of PhysNet: [https://github.com/MMunibas/PhysNet](https://github.com/MMunibas/PhysNet).


## Requirements:

- Tensorflow 2.20
- Python 3.13
- CUDA 12 (for running on GPUs only)

## Installation instructions:

1. Install Anaconda or Miniconda on the system. Link: [www.anaconda.com](https://www.anaconda.com).
2. In terminal, initialize the conda (`base`) environment. 
3. Create a environment of PhysNet with Tensorflow2.
```bash
    conda create -n physnet-tf2 python=3.13
    conda activate physnet-tf2
```
For system with GPU:
```bash
    pip install tensorflow[and-gpu] ase tensorboard
```
For system without GPU:
```bash
    pip install tensorflow ase tensorboard
```

## Training PhysNet:

1. Create a file `all.npz`, which contains the following information about the set of moleculs (for a set containing `n` configurations each with `m` atoms):

| Property | Symbol | Dimension of array | Data type | Units |  Remarks |
|---------|-----|-----|------|--------|---|
| Geometry index | N | n | int ||
| Energy | E | n | float | Hartree |  Individual atomic contributions removed |
| Charge | Q | n | float | e | Total charge on molecule |
| Dipole moment | D | n x 3 | float | Debye | Total dipole moment of molecule|
| Atomic number | Z | n x m | int |||
| Positions | R | n x m x 3 |float| Angstrom ||
| Forces | F | n x m x 3 | float | Hartree/Bohr || 

2. Modify the `config.txt` file according to specification. Ponder on the following:
   - `num_train` and `num_valid`: Number of points in training and validation sets respectively. The rest are used for test set. A general rule of thumb for split can be 80%, 10% and 10% for training, validation and testing.
   - `seed`: The initialization seed for random number. It is essential to change this if multiple models are trained on the same dataset.

### For local computer:

3. Load the ``physnet-tf2`` conda environment, and then run using:
    
```bash
python3 train.py @config.txt
```