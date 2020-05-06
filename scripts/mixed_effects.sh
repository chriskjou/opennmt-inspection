#!/bin/bash
#SBATCH -J mixed_effects 						# Job name
#SBATCH -p serial_requeue 						# partition (queue)
#SBATCH --mem 35000 							# memory pool for all cores
#SBATCH -t 0-24:00 								# time (D-HH:MM)
#SBATCH --output=/n/home10/cjou/projects 		# file output location
#SBATCH -o ../../../err_mixed_effects.txt 		# File that STDOUT writes to
#SBATCH -e ../../../outpt_mixed_effects.txt		# File that STDERR writes to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckjou@college.harvard.edu

module load Anaconda3/5.0.1-fasrc02
source activate virtualenv

python ../mixed_effects.py --embedding_layer /n/shieber_lab/Lab/users/cjou/embeddings/bert/layer1/avg.p  --model_to_brain   --cross_validation  --which_layer 1    --bert  --batch_num 0  