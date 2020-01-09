#!/bin/bash
#SBATCH -J convert_pickle_pred  				# Job name
#SBATCH -p seas_dgx1 							# partition (queue)
#SBATCH --mem 5000 								# memory pool for all cores
#SBATCH -t 0-24:00 								# time (D-HH:MM)
#SBATCH --output=/n/home10/cjou/projects 		# file output location
#SBATCH -o ../../outpt_convert_pickle_pred.txt 		# File that STDOUT writes to
#SBATCH -e ../../err_convert_pickle_pred.txt			# File that STDERR writes to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate virtualenv

python pickle_to_json.py