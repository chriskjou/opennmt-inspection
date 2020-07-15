#!/bin/bash
#SBATCH -J nested_anova  								# Job name
#SBATCH -p seas_dgx1 						# partition (queue)
#SBATCH --mem 3000 							# memory pool for all cores
#SBATCH -t 0-3:00 								# time (D-HH:MM)
#SBATCH --output=/n/home10/cjou/projects 		# file output location
#SBATCH -o ../../../outpt_nested_anova.txt 			# File that STDOUT writes to
#SBATCH -e ../../../err_nested_anova.txt				# File that STDERR writes to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate virtualenv

python ../nested_cv_significance.py -aal -avg