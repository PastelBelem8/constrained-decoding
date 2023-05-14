from run_decoder import get_model_filename
from toxic_clf_model import ToxicCommentTagger

import pandas as pd
import os, time, tqdm
from typing import Tuple
import spacy


NLP = spacy.load("en_core_web_sm", disable=["NER"])
TOXIC_MODEL_PATH = '/extra/ucinlp1/cbelem/experiment-ro-prompts/toxic_model.pth'


def parse_arguments() -> dict:
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--model_revision", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--decoding", default=None, type=str)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # CLI specified arguments get precedence over the config ones
    for arg in ("batch_size", "decoding", "model_name", "model_revision"):
        if getattr(args, arg) is not None:
            config[arg] = getattr(args, arg)

    return config


def load_toxicity_model(device: str=None) -> Tuple[ToxicCommentTagger, str]:
    import torch
    model = ToxicCommentTagger(n_classes=6)
    model.load_state_dict(torch.load(TOXIC_MODEL_PATH))
    model.eval();

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device);
    return model, device


def get_toxic_predictions(df: pd.DataFrame, colname: str, model: ToxicCommentTagger, batch_size: int, device: str) -> pd.DataFrame:
    df = df.copy()
    result = model.predict(df[colname].values, batch_size=batch_size, device=device)

    df[f"{colname}_toxicity"]     = result[:,0]
    df[f"{colname}_severe_toxic"] = result[:,1]
    df[f"{colname}_obscene"]      = result[:,2]
    df[f"{colname}_threat"]       = result[:,3]
    df[f"{colname}_insult"]       = result[:,4]
    df[f"{colname}_identity_hate"]= result[:,5]

    return df


def compute_diversity(text: spacy.tokens.doc.Doc) -> tuple:
    assert isinstance(text, spacy.tokens.doc.Doc)
    text_words = [tok.text for tok in text]
    nunique = len(set(text_words))

    return nunique, len(text_words)


def get_statistics(data: pd.DataFrame, colname) -> pd.DataFrame:
    data = data.copy()

    data[f"{colname}_num_chars"] = data[colname].apply(len)

    # Apply spacy pipeline
    sampled_texts = data[colname].values
    sampled_texts = [NLP(t) for t in sampled_texts]

    # Add num_sentences and num_puncts
    num_puncts, num_sents, num_words, num_unique, diversity = [], [], [], [], []
    for text in sampled_texts:
        num_puncts.append(len([t for t in text if t.is_punct]))
        num_sents.append(len(list(text.sents)))

        # Compute diversity
        nunique, n_words = compute_diversity(text)
        div = nunique/n_words if n_words != 0 else 0
        num_words.append(n_words)
        num_unique.append(nunique)
        diversity.append(div)

    data[f"{colname}_num_sentences"] = num_sents
    data[f"{colname}_num_punct"] = num_puncts
    data[f"{colname}_num_words"] = num_words
    data[f"{colname}_unique_words"] = num_unique
    data[f"{colname}_diversity"] = diversity

    return data


def collect(data: pd.DataFrame, colname, tmodel, tdevice, batch_size):
    print(f"--> 2. Get statistics from data with {len(data)} examples for colname", colname)
    start = time.time()
    data = get_statistics(data, colname=colname)
    duration = (time.time() - start) / 60
    print(f"       Finished in (min): {duration}")

    print("--> 3. Compute toxic predictions for", colname)
    start = time.time()
    data = get_toxic_predictions(
        data,
        colname,
        model=tmodel,
        batch_size=batch_size,
        device=tdevice,
    )

    duration = (time.time() - start) / 60
    print(f"       Finished in (min): {duration}")
    return data


if __name__ == "__main__":
    # Running this script
    # $ srun --pty --partition=ava_s.p --nodelist=ava-s5 --gpus=1 --mem=20G bash
    # $ python collect_properties_half_mult.py --config ./configs/collect_properties_half_mult_exp.yaml --model_name EleutherAI/pythia-70m --batch_size 32
    configs = parse_arguments()
    output_dir = configs["output_dir"]

    print("="*60)
    print(f"Collecting properties")
    print("[Experiment] Configs:\n{configs}")
    print("="*60)
    model_name  = get_model_filename(configs["model_name"], configs.get("model_revision"))
    output_dir = f"{output_dir}/{model_name}"

    print("\t-> 1. Creating results directory at", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    base_dir = configs["base_dir"]
    base_dir = f"{base_dir}/{model_name}"
    decoding_filenames = os.listdir(base_dir)
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

        if filename == "raw_data.csv":
            colname = "prefix"
            data = collect(data, "prefix", tmodel, tdevice, configs["batch_size"])
            data = collect(data, "sequence", tmodel, tdevice, configs["batch_size"])
        else:
            data = collect(data, colname, tmodel, tdevice, configs["batch_size"])

        print(f"--> 4. Persisting results at {output_dir}")
        data.to_csv(f"{output_dir}/{filename}")