#!/bin/bash
#SBATCH -J multi_script_test  					# Job name
#SBATCH -p seas_dgx1 							# partition (queue)
#SBATCH --gres=gpu:1							# for GPU
#SBATCH --mem 5000 								# memory pool for all cores
#SBATCH -t 0-24:00 								# time (D-HH:MM)
#SBATCH --output=/n/home10/cjou/projects 		# file output location
#SBATCH -o ../../outpt_multi_script_test.txt 			# File that STDOUT writes to
#SBATCH -e ../../err_multi_script_test.txt				# File that STDERR writes to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
module load cuda/9.0-fasrc02
source activate virtualenv

python batch_calculate_average_rank_multi.py --embedding_layer /n/shieber_lab/Lab/users/cjou/embeddings/parallel/spanish/2layer-brnn/avg/parallel-english-to-spanish-model-2layer-brnn-pred-layer1-avg.mat  --model_to_brain  --subject_number 1 --batch_num 0 --total_batches 100 --sub_batch_num 0 --total_sub_batches 10       
