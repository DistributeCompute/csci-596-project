#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:00:59
#SBATCH --output=dist_comp.out
#SBATCH --account=anakano_429

mpirun -bind-to none -n 2 ./dist_comp
