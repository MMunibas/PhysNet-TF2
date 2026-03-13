#!/usr/bin/env python3

# imports
import argparse
from ase import Atoms
from ase.io import read, write
from ase.optimize import *
from ase.visualize import view
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory
from os.path import splitext
from HessianNNCalculator.NNCalculator import *

#parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
required.add_argument("-i", "--input",   type=str,   help="input xyz",  required=True)
required.add_argument("-o", "--output",   type=str,   help="output traj",  required=True)
required.add_argument("-d", "--direction",   type=str,   help="forward (f) or reverse (r)",  required=True)

optional = parser.add_argument_group("optional arguments")
optional.add_argument("--charge",  type=float, help="total charge", default=0.0)

optional.add_argument("--timestep",  type=float, help="timestep for Langevin algorithm", default=0.1)
optional.add_argument("--friction",  type=float, help="friction coeff for Langevin algorithm", default=0.02)
optional.add_argument("--interval",  type=float, help="interval", default=1)


args = parser.parse_args()


if args.direction.lower()=="forward" or args.direction.lower()=="f":
    eps=0.00005
elif args.direction.lower()=="reverse" or args.direction.lower()=="r":
    eps=-0.00005
else:
    print("Only accepted options: forward (f) or reverse (r) for -d or --direction.")
    exit(0)


#read input file
atoms = read(args.input)
n_atom = len(atoms)
m = atoms.get_masses()

calc = NNCalculator(
    checkpoint=["./models/model-1/best_model.ckpt-696000"],
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
    use_dispersion=True,
    s6=1.0000,                    #s6 coefficient for d3 dispersion, by default is learned
    s8=2.3550,                    #s8 coefficient for d3 dispersion, by default is learned
    a1=0.5238,                    #a1 coefficient for d3 dispersion, by default is learned
    a2=3.5016)                   #a2 coefficient for d3 dispersion, by default is learned)


#setup calculator (which will be used to describe the atomic interactions)
atoms.set_calculator(calc)



#calculate gradients and hessians.
grad = -atoms.get_forces()
hessian = np.reshape(calc.get_hessian(atoms), (3*n_atom, 3*n_atom))
#obtain eigenvalues and vectors
w, v = np.linalg.eigh(hessian) # v[0] is the 'transition state vector'with imaginary freq, move along that direction.
v = v.T


# define the algorithm for MD:
dyn = VelocityVerlet(atoms, args.timestep * units.fs)


# save the positions of all atoms after every 100th time step.
traj = Trajectory(args.output + '.traj', 'w', atoms)
traj.write() #save current geometry



#set momenta corresponding to norm. displacement vector
atoms.set_momenta(eps*-np.reshape(v[0],(n_atom,3)))

calc.calc_hessian = False


# run the dynamics

for i in range(2000):
    dyn.run(1)
    if i%args.interval == 0:
        epot = atoms.get_potential_energy() / len(atoms)
        ekin = atoms.get_kinetic_energy() / len(atoms)
        print(i, epot, ekin, epot+ekin, calc.energy_stdev)
        traj.write()


