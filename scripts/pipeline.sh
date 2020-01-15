#!/bin/bash

### CREATE SCRIPTS
# run `python pipeline.py`

### DECODING
decoding_ID = $(sbatch --parsable ../../decoding_scripts/foldername/foldername.sh)
echo $decoding_ID

# -> merge rmses
merge_rmse_ID =$(sbatch --parsable --dependency=afterok:${decoding_ID} ../get_residuals.sh)
echo $merge_rmse_ID

### RANK
rank_ID = $(sbatch --parsable ../calculate_average_rank_test.sh)
echo $rank_ID

# -> merge rank

### FDR
fdr_ID = $(sbatch --parsable ../fdr_test.sh)
echo $fdr_ID

# -> merge FDR

### LLH

# -> merge LLH