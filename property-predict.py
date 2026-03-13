#!/usr/bin/env python3
import argparse
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.visualize import view
from ase.optimize import *
from NNCalculator.NNCalculator import *

'''
Script to calculate the potential energy of a structure.
Usage: python predic_mol.py -i structure.xyz
'''

#parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input",   type=str,   help="input xyz",  required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("--charge",  type=int, help="total charge", default=0)

args = parser.parse_args()
#print("input ", args.input)

#read input file (molecular structure to predict) and create
#an atoms object
traj = read(args.input, index = ":")
atoms = traj[0]

#setup calculator object, which in this case is the NN calculator
#it is important that it is setup with the settings as used in the
#training procedure.
calc = NNCalculator(
    checkpoint="./models/XXXXX/YYYYY", #load the model you want to used
    atoms=atoms,
    charge=args.charge,
    F=128,
    K=64,
    num_blocks=5,
    num_residual_atomic=2,
    num_residual_interaction=3,
    num_residual_output=1,
    sr_cut=10.0,
    use_electrostatic=True,
    use_dispersion=False,
    s6=1.0000,                    #s6 coefficient for d3 dispersion, by default is learned
    s8=2.3550,                    #s8 coefficient for d3 dispersion, by default is learned
    a1=0.5238,                    #a1 coefficient for d3 dispersion, by default is learned
    a2=3.5016)                   #a2 coefficient for d3 dispersion, by default is learned)


num=len(traj)
Nmax=atoms.get_number_of_atoms()

#Initialize the matrices
N = np.zeros([num], dtype=int)
E = np.zeros([num], dtype=float)
Q = np.zeros([num], dtype=float)
D = np.zeros([num, 3], dtype=float)
Z = np.zeros([num, Nmax], dtype=int)
R = np.zeros([num, Nmax, 3], dtype=float)
F = np.zeros([num, Nmax, 3], dtype=float)

for ii in range(num):
    atoms=traj[ii]

    #attach the calculator object to the atoms object
    atoms.set_calculator(calc)

    #calculate properties of the molecule/geometry
    #[for an extensive list see https://wiki.fysik.dtu.dk/ase/ase/atoms.html]
    epot = atoms.get_potential_energy().item()
    f = atoms.get_forces()
    mu = np.dot(atoms.get_charges(), atoms.get_positions()) #where atoms.get_charges() calculates
    #the atomic partial charges (in elementary charges)

    N[ii]=Nmax
    E[ii]=epot
    Q[ii]=args.charge
    D[ii,:]=mu
    Z[ii,:]=atoms.get_atomic_numbers()
    R[ii,:,:]=atoms.get_positions()
    F[ii,:,:]=f
    
    if ii%100==0:
        print(ii," processed")

np.savez_compressed(f"data_XXXXX.npz",N=N, E=E, Q=Q, D=D, Z=Z, R=R, F=F)

quit()
