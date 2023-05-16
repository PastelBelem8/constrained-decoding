cd ../code

python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_data/PILE/raw_data.csv --colname sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_data/PILE/top_k_2.csv --colname sampled_sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_data/PILE/top_k_10.csv --colname sampled_sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_data/PILE/top_k_40.csv --colname sampled_sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_data/PILE/top_k_100.csv --colname sampled_sequence --batch_size 32 --device cuda:0

python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_model/raw_data.csv --colname sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_model/top_k_2.csv --colname sampled_sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_model/top_k_10.csv --colname sampled_sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_model/top_k_40.csv --colname sampled_sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_model/top_k_100.csv --colname sampled_sequence --batch_size 32 --device cuda:0

python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/toxic_words/raw_data.csv --colname sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/toxic_words/top_k_2.csv --colname sampled_sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/toxic_words/top_k_10.csv --colname sampled_sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/toxic_words/top_k_40.csv --colname sampled_sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/toxic_words/top_k_100.csv --colname sampled_sequence --batch_size 32 --device cuda:0

python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/religions/raw_data.csv --colname sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/religions/top_k_2.csv --colname sampled_sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/religions/top_k_10.csv --colname sampled_sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/religions/top_k_40.csv --colname sampled_sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/religions/top_k_100.csv --colname sampled_sequence --batch_size 32 --device cuda:0

python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_good/raw_data.csv --colname sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_good/top_k_2.csv --colname sampled_sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_good/top_k_10.csv --colname sampled_sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_good/top_k_40.csv --colname sampled_sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_good/top_k_100.csv --colname sampled_sequence --batch_size 32 --device cuda:0

python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_bad/raw_data.csv --colname sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_bad/top_k_2.csv --colname sampled_sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_bad/top_k_10.csv --colname sampled_sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_bad/top_k_40.csv --colname sampled_sequence --batch_size 32 --device cuda:0
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_bad/top_k_100.csv --colname sampled_sequence --batch_size 32 --device cuda:0
