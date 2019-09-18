#!/bin/bash
validation_num_lines=5000
mkdir "../multiparallelize/validation"
mkdir "../multiparallelize/training"
for path in ../multiparallelize/*.txt; do
	total_num_lines=$(wc -l $path | awk '{print $1}')
	filename=$(basename $path)
	filename=${filename%.txt}
 	training_lines=$(expr $total_num_lines - $validation_num_lines)
	validation_filename=$"../multiparallelize/validation/${filename}_validation.txt"
	training_filename=$"../multiparallelize/training/${filename}_training.txt"
	touch $training_filename && head -n $training_lines $path > $training_filename
	touch $validation_filename && tail -n $validation_num_lines $path > $validation_filename
	echo "${filename} setup complete"
done
echo "done"
