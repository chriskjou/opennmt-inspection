#!/bin/bash
#SBATCH -J only_voxel_count  					# Job name
#SBATCH -p seas_dgx1 							# partition (queue)
#SBATCH --mem 10000 							# memory pool for all cores
#SBATCH -t 0-24:00 								# time (D-HH:MM)
#SBATCH --output=/n/home10/cjou/projects 		# file output location
#SBATCH -o ../../rank_logs/outpt_only_voxel_count.txt 		# File that STDOUT writes to
#SBATCH -e ../../rank_logs/err_only_voxel_count.txt			# File that STDERR writes to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate virtualenv

python calculate_average_rank.py --embedding_layer /n/shieber_lab/Lab/users/cjou/embeddings/parallel/spanish/2layer-brnn/avg/parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg.mat --batch_num 0 --model_to_brain --total_batches 100