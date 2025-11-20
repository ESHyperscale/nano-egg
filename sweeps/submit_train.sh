#!/bin/bash

list1=("64" "512" "4096" "32768" "262144")


for A in "${list1[@]}"; do
    echo "${A}"
    sbatch --job-name="egg2_${A}" --output="outputs/egg_${A}.txt" --gpus=1 --time=12:00:00 --wrap "~/data/nano-egg/sweeps/do_train.sh $A"
done
