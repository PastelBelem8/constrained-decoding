cd ../code

python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_data/PILE/top_p_0.1.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_data/PILE/top_p_0.3.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_data/PILE/top_p_0.5.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_data/PILE/top_p_0.7.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_data/PILE/top_p_0.8.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_data/PILE/top_p_0.9.csv --colname sampled_sequence --batch_size 32 --device cuda:3

python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_model/top_p_0.1.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_model/top_p_0.3.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_model/top_p_0.5.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_model/top_p_0.7.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_model/top_p_0.8.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_model/top_p_0.9.csv --colname sampled_sequence --batch_size 32 --device cuda:3

python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/toxic_words/top_p_0.1.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/toxic_words/top_p_0.3.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/toxic_words/top_p_0.5.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/toxic_words/top_p_0.7.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/toxic_words/top_p_0.8.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/toxic_words/top_p_0.9.csv --colname sampled_sequence --batch_size 32 --device cuda:3

python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/religions/top_p_0.1.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/religions/top_p_0.3.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/religions/top_p_0.5.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/religions/top_p_0.7.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/religions/top_p_0.8.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/religions/top_p_0.9.csv --colname sampled_sequence --batch_size 32 --device cuda:3

python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_good/top_p_0.1.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_good/top_p_0.3.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_good/top_p_0.5.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_good/top_p_0.7.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_good/top_p_0.8.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_good/top_p_0.9.csv --colname sampled_sequence --batch_size 32 --device cuda:3

python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_bad/top_p_0.1.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_bad/top_p_0.3.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_bad/top_p_0.5.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_bad/top_p_0.7.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_bad/top_p_0.8.csv --colname sampled_sequence --batch_size 32 --device cuda:3
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_bad/top_p_0.9.csv --colname sampled_sequence --batch_size 32 --device cuda:3
