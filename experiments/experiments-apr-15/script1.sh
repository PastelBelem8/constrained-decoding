# conda activate py39

for target_word in "muslim", "jewish", "buddhist", "christian", "doctor", "nurse", "mother", "father", "bird"
do
    python -m collect_data_script --target-word $target_word --num-samples 2500 &
done
