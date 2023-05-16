cd ../code

python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_data/PILE/temperature_0.1.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_data/PILE/temperature_0.3.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_data/PILE/temperature_0.5.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_data/PILE/temperature_0.95.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_data/PILE/temperature_1.15.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_data/PILE/temperature_1.5.csv --colname sampled_sequence --batch_size 32 --device cuda:1

python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_model/temperature_0.1.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_model/temperature_0.3.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_model/temperature_0.5.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_model/temperature_0.95.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_model/temperature_1.15.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/seed_model/temperature_1.5.csv --colname sampled_sequence --batch_size 32 --device cuda:1

python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/toxic_words/temperature_0.1.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/toxic_words/temperature_0.3.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/toxic_words/temperature_0.5.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/toxic_words/temperature_0.95.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/toxic_words/temperature_1.15.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/toxic_words/temperature_1.5.csv --colname sampled_sequence --batch_size 32 --device cuda:1

python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/religions/temperature_0.1.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/religions/temperature_0.3.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/religions/temperature_0.5.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/religions/temperature_0.95.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/religions/temperature_1.15.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/religions/temperature_1.5.csv --colname sampled_sequence --batch_size 32 --device cuda:1

python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_good/temperature_0.1.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_good/temperature_0.3.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_good/temperature_0.5.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_good/temperature_0.95.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_good/temperature_1.15.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_good/temperature_1.5.csv --colname sampled_sequence --batch_size 32 --device cuda:1

python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_bad/temperature_0.1.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_bad/temperature_0.3.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_bad/temperature_0.5.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_bad/temperature_0.95.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_bad/temperature_1.15.csv --colname sampled_sequence --batch_size 32 --device cuda:1
python -m run_properties --input_filepath ../data/EleutherAI__pythia-1.4b/constr_seed_data/adjectives_bad/temperature_1.5.csv --colname sampled_sequence --batch_size 32 --device cuda:1
