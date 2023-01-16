#!/bin/bash
#SBATCH -J tenpy_simulation
#SBATCH -p bigmem
#SBATCH -N 1 	# Number of nodes
#SBATCH -n 1  	# Number of process
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time 48:00:00
#SBATCH --mail-user=lehoanganh1112@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --comment python     # See Application SBATCH options name table's

module purge
module load python/3.7.1 
module load intel/19.1.2

python -u ./run_DMRG_w4.py

