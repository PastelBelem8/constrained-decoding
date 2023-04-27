from run_decoder import get_model_filename
from toxic_clf_model import ToxicCommentTagger

import pandas as pd
import os, time
from typing import Tuple
import spacy


NLP = spacy.load("en_core_web_sm", disable=["NER"])
TOXIC_MODEL_PATH = '/extra/ucinlp1/cbelem/experiment-ro-prompts/toxic_model.pth'


def parse_arguments() -> dict:
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",  "--config", required=True, type=str)
    parser.add_argument("-m",  "--model_name", required=True, type=str)
    parser.add_argument("-r",  "--model_revision", default=None, type=str)
    parser.add_argument("-d",  "--device", default="cuda", type=str)
    parser.add_argument("-bs", "--batch_size", default=None, type=int)
    parser.add_argument("-n",  "--num_samples", default=None, type=int)
    parser.add_argument("-dec",  "--decoding", default=None, type=str)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # CLI specified arguments get precedence over the config ones
    for arg in ("batch_size", "num_samples", "decoding", "model_name", "model_revision"):
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

    df[f"toxicity"]     = result[:,0]
    df[f"severe_toxic"] = result[:,1]
    df[f"obscene"]      = result[:,2]
    df[f"threat"]       = result[:,3]
    df[f"insult"]       = result[:,4]
    df[f"identity_hate"]= result[:,5]

    return df


def get_sample(base_dir: str, model_name: str, decoding: str, num_samples: int, seed: int) -> pd.DataFrame:
    data_fp = f"{base_dir}/{model_name}-{decoding}.csv"

    print("Reading original data from", data_fp)
    data = pd.read_csv(data_fp, index_col=0)

    sample = data.sample(n=num_samples, replace=False, random_state=seed)
    return sample


def compute_diversity(text: spacy.tokens.doc.Doc) -> tuple:
    assert isinstance(text, spacy.tokens.doc.Doc)
    text_words = [tok.text for tok in text]
    unique_words = len(set(text_words))

    return unique_words / len(text_words), len(text_words)


def get_statistics(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data["num_chars"] = data["sequence"].apply(len)

    # Apply spacy pipeline
    sampled_texts = data["sequence"].values
    sampled_texts = [NLP(t) for t in sampled_texts]

    # Add num_sentences and num_puncts
    num_puncts, num_sents, num_words, diversity = [], [], [], []
    for text in sampled_texts:
        num_puncts.append(len([t for t in text if t.is_punct]))
        num_sents.append(len(list(text.sents)))

        # Compute diversity
        div, n_words = compute_diversity(text)
        num_words.append(n_words)
        diversity.append(div)

    data["num_sentences"] = num_sents
    data["num_punct"] = num_puncts
    data["num_words"] = num_words
    data["diversity"] = diversity

    return data


if __name__ == "__main__":
    # Running this script
    # $ srun --pty --partition=ava_s.p --nodelist=ava-s5 --gpus=1 --mem=20G bash
    # $ python -m run_properties --config ./configs/default_properties.yml --model_name EleutherAI/pythia-70m --num_samples 10000 -dec multinomial

    configs = parse_arguments()
    output_dir = configs["output_dir"]

    print(f"Collecting properties \n[Experiment] Configs: {configs}")
    model_name  = get_model_filename(configs["model_name"], configs.get("model_revision"))
    output_dir = f"{output_dir}/{model_name}"

    print("\t-> 1. Creating results directory at", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()
    print("\t-> 2. Sampling sequences")
    decoding = configs["decoding"]
    samples = get_sample(
        base_dir=configs["base_dir"],
        model_name=model_name,
        decoding=decoding,
        num_samples=configs["num_samples"],
        seed=configs["seed"],
    )

    print(f"\t-> 3. Get statistics from the {len(samples)} sampled sequences")
    samples = get_statistics(samples)

    print("\t-> 4. Loading toxicity model")
    tmodel, tdevice = load_toxicity_model()

    print("\t-> 5. Compute predictions")
    samples = get_toxic_predictions(samples, "sequence", model=tmodel, batch_size=configs["batch_size"], device=tdevice)

    print(f"\t-> 6. Persisting results at {output_dir}")
    samples.to_csv(f"{output_dir}/{decoding}.csv")

    duration = (time.time() - start) / 60
    print(f"Finished!\n Ellapsed time (min): {duration}")
