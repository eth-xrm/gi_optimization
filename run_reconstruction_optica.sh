#!/bin/bash


#SBATCH --mem=150G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --array=0-72

# Copyright (c) 2025, ETH Zurich

# Declare parameters as arrays
declare -a sample_sizes=(9 12 14 16)
declare -a dose_ratios=(1.0 0.81534387 0.71952235 0.64572291)

declare -a doses=(0.2 0.3 0.4 0.5 0.6 0.7 1 1.5 2 2.5 3 4 5 6 7 8 9 10)


# Get and echo the length of the arrays
echo "Length of sample_sizes array: ${#sample_sizes[@]}"
echo "Length of dose_ratios array: ${#dose_ratios[@]}"

num_sample_sizes=${#sample_sizes[@]}  # Number of sample_sizes
num_doses=${#doses[@]}               # Number of doses
total_combinations=$((num_sample_sizes * num_doses))

# Determine the indices for sample_size, dose_ratio, and dose
sample_index=$((SLURM_ARRAY_TASK_ID / num_doses))  # Integer division for sample_sizes
dose_index=$((SLURM_ARRAY_TASK_ID % num_doses))   # Modulus for doses



# Dynamically assign parameters based on task ID
sample_size_cm=${sample_sizes[$sample_index]}
dose_ratio_to_rawlik=${dose_ratios[$sample_index]}
dose=${doses[$dose_index]}

# Fixed parameter
measurement_name='Static_Cone'
visibility_penalty=0.49

# GI Parameters
grating_height=180e-6
source_sample_distance=1.018
sample_g2_distance=0.718
pitch=4.2e-6
talbot_order=5
design_energy=46000
kvp=70
pixel=200
visibility=0.11
#vis_spectrum='None'
vis_spectrum='utils/vis_spectrum/vis_spectrum_static.npy'
energy_spectrum='utils/vis_spectrum/energies_spectrum_static.npy'

echo "Running run_reconstruction_optica.sh with parameters:"
echo "    measurement_name:" ${measurement_name}
echo "    sample_size_cm:" ${sample_size_cm}
echo "    dose_ratio_to_rawlik:" ${dose_ratio_to_rawlik}
echo "    Dose:" ${dose}
echo "Running reconstructions..."

python reconstruction_simulation.py \
    --measurementname ${measurement_name} \
    --samplesize ${sample_size_cm} \
    --doseratio ${dose_ratio_to_rawlik} \
    --vp ${visibility_penalty} \
    --doses ${dose} \
    --gratingheight ${grating_height} \
    --ss ${source_sample_distance} \
    --sd ${sample_g2_distance} \
    --pitch ${pitch} \
    --to ${talbot_order} \
    --ed ${design_energy} \
    --kvp ${kvp} \
    --pixel ${pixel} \
    --visibility ${visibility} \
    --vis_spectrum ${vis_spectrum}\
    --energy_spectrum ${energy_spectrum}\
    --vishardening \
    || exit 1