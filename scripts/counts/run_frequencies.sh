
#!/bin/bash

# run in ava-s5
PILE_URL="/extra/ucinlp1/PILE/train"
for i in {00..29}
do
        python /home/cbelem/projects/freq_count/compute_unigrams_in_disk.py --file $PILE_URL/$i.jsonl.zst --output-dir $PILE_URL/temp/PILE-30-unigrams -n 3 &
done