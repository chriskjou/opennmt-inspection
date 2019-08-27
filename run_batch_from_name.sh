#!/bin/bash

while IFS= read filename; do
  echo "\"sbatch ${filename} -H\""
done
