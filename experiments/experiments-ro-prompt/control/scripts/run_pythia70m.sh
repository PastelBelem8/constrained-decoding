#!/bin/bash
python run_decoder.py --c ./configs/multinomial.yaml -m EleutherAI/pythia-70m -d cuda:0 -bs 128
python run_decoder.py --c ./configs/temp.yaml -m EleutherAI/pythia-70m -d cuda:0 -bs 128
python run_decoder.py --c ./configs/top_k.yaml -m EleutherAI/pythia-70m -d cuda:0 -bs 128
python run_decoder.py --c ./configs/top_p.yaml -m EleutherAI/pythia-70m -d cuda:0 -bs 128


python run_decoder.py --c ./configs/multinomial.yaml -m EleutherAI/pythia-70m-deduped -d cuda:0 -bs 128
python run_decoder.py --c ./configs/temp.yaml -m EleutherAI/pythia-70m-deduped -d cuda:0 -bs 128
python run_decoder.py --c ./configs/top_k.yaml -m EleutherAI/pythia-70m-deduped -d cuda:0 -bs 128
python run_decoder.py --c ./configs/top_p.yaml -m EleutherAI/pythia-70m-deduped -d cuda:0 -bs 128
