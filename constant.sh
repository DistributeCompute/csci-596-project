#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:02:59
#SBATCH --output=<CONSTANT_NAME>.out
#SBATCH --account=anakano_429

mpirun -bind-to none -n 2 ./<CONSTANT_NAME> 
