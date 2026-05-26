# PhysNet-TF2
TensorFlow 2 PhysNet implementation

Patching of PhysNet for Tensorflow 2.

Git repository for original (TensorFlow 1) version of PhysNet: [https://github.com/MMunibas/PhysNet](https://github.com/MMunibas/PhysNet).


## Requirements:

- Tensorflow 2.20
- Python 3.13
- CUDA 12 (for running on GPUs only)

## Pre-requisite setup instructions:

1. Install Anaconda or Miniconda on the system. Link: [www.anaconda.com](https://www.anaconda.com).
2. In terminal, initialize the conda (`base`) environment. 
3. Create a environment of PhysNet with Tensorflow2.
```bash
conda create -n physnet-tf2 python=3.13
conda activate physnet-tf2
```
For system with GPU:
```bash
pip install tensorflow[and-cuda] ase tensorboard
```
For system without GPU:
```bash
pip install tensorflow ase tensorboard
```

## Training PhysNet:

1. Create a file `all.npz`, which contains the following information about the set of moleculs (for a set containing `n` configurations each with `m` atoms):

| Property | Symbol | Dimension of array | Data type | Units |  Remarks |
|---------|-----|-----|------|--------|---|
| No. of atoms | N | n | int ||
| Energy | E | n | float | eV |  Individual atomic contributions removed |
| Charge | Q | n | float | e | Total charge on molecule |
| Dipole moment | D | n x 3 | float | a.u. | Total dipole moment of molecule|
| Atomic number | Z | n x m | int |||
| Positions | R | n x m x 3 |float| Angstrom ||
| Forces | F | n x m x 3 | float | eV/Angstrom || 

2. Modify the `config.txt` file according to specification. Ponder on the following:
   - `num_train` and `num_valid`: Number of points in training and validation sets respectively. The rest are used for test set. A general rule of thumb for split can be 80%, 10% and 10% for training, validation and testing.
   - `seed`: The initialization seed for random number. It is essential to change this if multiple models are trained on the same dataset.

### For local computer:

3. Load the ``physnet-tf2`` conda environment, and then run using:
    
```bash
python3 train.py @config.txt
```

### For running on cluster (slurm queuing system):

3. Modify the submit.run file according to cluster specifications.
4. Submit using the following command:
```bash
sbatch submit.run
```

## GPU Compatibility:

> [!WARNING]
> Although the latest CUDA version released is CUDA 13, Tensorflow 2 currently supports only until CUDA 12. Therefore, please ensure that **CUDA 12 is enabled in your system during runtime.**

> [!TIP]
> To check the runtime CUDA version of the local computer (or node), one can use `nvidia-smi` command.

> [!NOTE]
> This version has been tested on NVIDIA RTX 2080Ti, RTX 3080, RTX 3090, RTX 3090 Ti and RTX 4090 GPU cards, along with their submodels.

## Utilities:

Some utility scripts are also bundled with this software, which are as follows:

- **predict_dataset.py**: Predict the energy, and error for energy and forces in training, validation and test sets.
- **energy-predict.py**: Predict the energy of a xyz file (multiple frames) using a trained model.
- **property-predict.py**: Predicts all properties (energy, forces and dipole moments) for a xyz file (multiple frames) using a trained model and store as a npz file, which can be used for further training.
- **md-sim.py**: Carry out MD simulation using PhysNet model.
- **dmc.py**: Carry out diffusion Monte Carlo sampling. The file _dmc_template.txt_ contains parameters for the same.
- **minimum_dynamic_path.py**: Compute the minimum dynamic path for a transition state, for both forward and reverse directions.
- **adpative-sampling.py**: Compute the adaptive sampling trajectory for multiple models, identify cases with large mismatch in energy between models.
- **normal-mode-npz.py**: Compute normal mode and store the data as a npz file.

## Additional functions:

### Gaussian 16 interface

- **opt_hcooh.com**: Optimize with Gaussian using PhysNet MLFF.
- **nnvpt2_hcooh.com**: Compute anharmonic frequencies with Gaussian using PhysNet MLFF.


## Citation:

```bibtex
@article{unke2019physnet,
  title={PhysNet: A neural network for predicting energies, forces, dipole moments, and partial charges},
  author={Unke, Oliver T and Meuwly, Markus},
  journal={Journal of chemical theory and computation},
  volume={15},
  number={6},
  pages={3678--3693},
  year={2019},
  publisher={ACS Publications}
}
```
