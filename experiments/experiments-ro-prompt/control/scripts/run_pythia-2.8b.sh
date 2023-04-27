#!/bin/bash
python run_decoder.py --c ./configs/multinomial.yaml -m EleutherAI/pythia-2.8b -bs 16
python run_decoder.py --c ./configs/temp.yaml -m EleutherAI/pythia-2.8b -bs 16
python run_decoder.py --c ./configs/top_k.yaml -m EleutherAI/pythia-2.8b -bs 16
python run_decoder.py --c ./configs/top_p.yaml -m EleutherAI/pythia-2.8b -bs 16


python run_decoder.py --c ./configs/multinomial.yaml -m EleutherAI/pythia-2.8b-deduped -bs 16
python run_decoder.py --c ./configs/temp.yaml -m EleutherAI/pythia-2.8b-deduped -bs 16
python run_decoder.py --c ./configs/top_k.yaml -m EleutherAI/pythia-2.8b-deduped -bs 16
python run_decoder.py --c ./configs/top_p.yaml -m EleutherAI/pythia-2.8b-deduped -bs 16
