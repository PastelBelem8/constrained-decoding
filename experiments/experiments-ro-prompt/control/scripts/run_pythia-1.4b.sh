#!/bin/bash
python run_decoder.py --c ./configs/multinomial.yaml -m EleutherAI/pythia-1.4b -bs 32
python run_decoder.py --c ./configs/temp.yaml -m EleutherAI/pythia-1.4b -bs 32
python run_decoder.py --c ./configs/top_k.yaml -m EleutherAI/pythia-1.4b -bs 32
python run_decoder.py --c ./configs/top_p.yaml -m EleutherAI/pythia-1.4b -bs 32


python run_decoder.py --c ./configs/multinomial.yaml -m EleutherAI/pythia-1.4b-deduped -bs 32
python run_decoder.py --c ./configs/temp.yaml -m EleutherAI/pythia-1.4b-deduped -bs 32
python run_decoder.py --c ./configs/top_k.yaml -m EleutherAI/pythia-1.4b-deduped -bs 32
python run_decoder.py --c ./configs/top_p.yaml -m EleutherAI/pythia-1.4b-deduped -bs 32