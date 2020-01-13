#!/bin/bash
#SBATCH -J fdr_test  							# Job name
#SBATCH -p seas_dgx1 							# partition (queue)
#SBATCH --mem 5000 								# memory pool for all cores
#SBATCH -t 0-24:00 								# time (D-HH:MM)
#SBATCH --output=/n/home10/cjou/projects 		# file output location
#SBATCH -o ../../outpt_fdr_test.txt 		# File that STDOUT writes to
#SBATCH -e ../../err_fdr_test.txt			# File that STDERR writes to
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate virtualenv

python significance_threshold.py --embedding_layer /n/shieber_lab/Lab/users/cjou/embeddings/parallel/spanish/2layer-brnn/avg/parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg.mat