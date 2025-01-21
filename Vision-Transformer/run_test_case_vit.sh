#!/bin/bash

# Define arrays for tau and lr values
tau_values="0.15"  # Add more values if needed
lr_values="1e-3"   # Add more values if needed
num_runs="1 2 3 4 5"
test_case="cifar10" # or cifar100
wandb_act="0" # enter 1 to run with wandb logging

# Loop over tau and lr values
for run in $num_runs
do
    for tau in $tau_values 
        do
        for lr in $lr_values 
            do
            # Execute the Python script with the current values of tau and lr
            python main_VIT_psi_lora.py --dataset_name $test_case \
                                        --net_name vit_lora \
                                        --wandb 1 \
                                        --batch_size 256 \
                                        --epochs 5 \
                                        --lr $lr \
                                        --init_r 32 \
                                        --coeff_steps 1 \
                                        --wandb $wandb_act \
                                        --tau $tau
        done
    done
done
