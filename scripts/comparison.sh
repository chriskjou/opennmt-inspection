#!/bin/bash
#SBATCH -J comparison 					# Job name
#SBATCH -p seas_dgx1 							# partition (queue)
#SBATCH --mem 2000 							# memory pool for all cores
#SBATCH -t 0-2:00 								# time (D-HH:MM)
#SBATCH --output=/n/home10/cjou/projects 		# file output location
#SBATCH -o ../../../outpt_comparison.txt 		# File that STDOUT writes to
#SBATCH -e ../../../err_comparison.txt		# File that STDERR writes to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate virtualenv

python ../compare_layers_and_subjects.py --cross_validation --model_to_brain --total_batches 100 --layer1 1 --layer2 2