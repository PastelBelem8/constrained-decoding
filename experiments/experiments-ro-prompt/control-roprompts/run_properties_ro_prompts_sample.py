

import pandas as pd
import os, time, tqdm
import spacy

import sys, pprint
sys.path.append(f"{__file__.rpartition('/')[0]}/../control")
from run_decoder import get_model_filename
from collect_properties_half_mult import load_toxicity_model, collect

NLP = spacy.load("en_core_web_sm", disable=["NER"])
TOXIC_MODEL_PATH = '/extra/ucinlp1/cbelem/experiment-ro-prompts/toxic_model.pth'



def parse_arguments() -> dict:
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--model_revision", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=None, type=int)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # CLI specified arguments get precedence over the config ones
    for arg in ("batch_size", "model_name", "model_revision"):
        if getattr(args, arg) is not None:
            config[arg] = getattr(args, arg)

    return config


if __name__ == "__main__":
    # Running this script
    # srun --pty --partition=ava_s.p --nodelist=ava-s5 --gpus=1 --mem=20G bash
    # python -m run_properties_ro_prompts --config ./configs/properties-roprompt.yaml --model_name EleutherAI/pythia-70m --batch_size 32
    # python -m run_properties_ro_prompts --config ./configs/properties-raw-roprompt.yaml --batch_size 32

    configs = parse_arguments()
    base_dir = configs["base_dir"]
    output_dir = configs["output_dir"]

    print("="*60)
    print(f"Collecting properties")
    print(f"[Experiment] Configs:\n{configs}")
    print("="*60)

    if configs.get("model_name") is not None:
        model_name  = get_model_filename(configs["model_name"], configs.get("model_revision"))
        base_dir = f"{base_dir}/{model_name}"
        output_dir = f"{output_dir}/{model_name}"
        decoding_filenames = os.listdir(base_dir)
    else:
        decoding_filenames = ["raw_data.csv"]

    print("\n\n")
    print("-> 1. Creating results directory at", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Found the following filenames under:")
    print("--> Base_dir:", base_dir)
    print("--> Filenames:", decoding_filenames)

    print("--> Loading toxicity model...")
    start = time.time()
    tmodel, tdevice = load_toxicity_model()
    duration = (time.time() - start) / 60
    print(f"\tDuration (min): {duration}")

    for filename in tqdm.tqdm(sorted(decoding_filenames)):
        colname = configs["colname"]
        print("-"*60)
        print(filename)
        print("-"*60)
        print("--> 1. Reading file:", filename)
        data = pd.read_csv(f"{base_dir}/{filename}", index_col=0)
        sample = data.sample(n=configs.get("num_samples", 15_000), replace=False, random_state=configs["seed"])
        sample = collect(sample, colname, tmodel, tdevice, configs["batch_size"])

        print(f"--> 4. Persisting results at {output_dir}/{filename}")
        print(sample.columns)

        sample.to_csv(f"{output_dir}/{filename}")