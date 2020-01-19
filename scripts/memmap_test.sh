#!/bin/bash
#SBATCH -J mp_memmap_test  					# Job name
#SBATCH -p serial_requeue 						# partition (queue)
#SBATCH -n 5 									# Number of cores requested
#SBATCH -N 1 									# Ensure that all cores are on one machine
#SBATCH --mem 5000 							# memory requested per node
#SBATCH -t 1-00:00 								# time (D-HH:MM)
#SBATCH --output=/n/home10/cjou/projects 		# file output location
#SBATCH -o ../../../outpt_mp_memmap_test.txt 		# File that STDOUT writes to
#SBATCH -e ../../../err_mp_memmap_test.txt		# File that STDERR writes to
#SBATCH --open-mode=append
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate virtualenv

python ../convert_np_to_memmap.py