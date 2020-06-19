#!/bin/bash
#SBATCH -J anova  								# Job name
#SBATCH -p serial_requeue 						# partition (queue)
#SBATCH --mem 2000 							# memory pool for all cores
#SBATCH -t 0-24:00 								# time (D-HH:MM)
#SBATCH --output=/n/home10/cjou/projects 		# file output location
#SBATCH -o ../../../outpt_anova.txt 			# File that STDOUT writes to
#SBATCH -e ../../../err_anova.txt				# File that STDERR writes to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate virtualenv

python ../calculate_slope_maps.py -argmax -anova