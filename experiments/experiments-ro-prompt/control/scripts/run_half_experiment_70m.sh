#!/bin/bash
# Usage example: 
# ./scripts/run_properties.sh EleutherAI/pythia-70m 100
# # ./scripts/run_properties.sh EleutherAI/pythia-1.4b 100
# ----------------------------------------------------------
# Warning: Don't forget to use SLURM
# srun --pty --partition=ava_s.p --nodelist=ava-s5 --gpus=1 --mem=20G bash
# conda activate py39
# cd $HOME/projects/constrained-decoding/experiment-ro-prompts/control
model_name=$1
num_samples=130000

echo MODEL_NAME=$model_name
echo NUM_SAMPLES $num_samples
echo 3!; sleep 1s; echo 2!; sleep 1s; echo 1!; sleep 1s; echo Starting...

echo ===================== MULTINOMIAL ===========================
python -m run_properties --config ./configs/default_properties.yml --model_name $model_name --num_samples $num_samples -dec multinomial

for temp in 0.1 0.3 0.5 0.85 1.15 1.5;
do
    echo "===================== TEMPERATURE ($temp) ==========================="
    python -m run_properties --config ./configs/default_properties.yml --model_name $model_name --num_samples $num_samples -dec temperature_$temp
done

for topk in 2 10 40 100;
do
    echo "===================== TOP-K ($topk) ==========================="
    python -m run_properties --config ./configs/default_properties.yml --model_name $model_name --num_samples $num_samples -dec top_k_$topk
done

for topp in 0.1 0.3 0.5 0.7 0.8 0.9;
do
    echo "===================== TOP-P ($topp) ==========================="
    python -m run_properties --config ./configs/default_properties.yml --model_name $model_name --num_samples $num_samples -dec top_p_$topp
done
