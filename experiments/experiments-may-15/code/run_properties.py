from toxic_clf_model import ToxicCommentTagger

from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import Tuple

import argparse
import pandas as pd
import numpy as np


import torch


TOXIC_MODEL_PATH = '/extra/ucinlp1/cbelem/experiment-ro-prompts/toxic_model.pth'


def load_toxicity_model(device: str=None) -> Tuple[ToxicCommentTagger, str]:
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
    # df[f"severe_toxic"] = result[:,1]
    # df[f"obscene"]      = result[:,2]
    # df[f"threat"]       = result[:,3]
    # df[f"insult"]       = result[:,4]
    # df[f"identity_hate"]= result[:,5]
    return df


def load_sa_model(device: str=None) -> Tuple[T5ForConditionalGeneration, T5Tokenizer, str]:
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def get_templates_sa(dataset="glue", subset="sst2"):
    # Load prompts for this dataset
    from promptsource.templates import DatasetTemplates
    sst2prompts = DatasetTemplates(dataset, subset)

    templates_names = [
        'positive negative after',
        "review",
        "said",
        "following positive negative",
        "happy or mad",
    ]

    templates = [sst2prompts[t] for t in templates_names]
    return templates, templates_names


def get_sa_predictions(df: pd.DataFrame, colname: str, template, model: T5ForConditionalGeneration, tokenizer, batch_size: int, device: str) -> pd.DataFrame:
    df = df.copy()
    sequences = df[colname].values
    answer_choices = template.get_answer_choices_list({"sentence": sequences[0]})

    results = []
    for bstart in range(0, len(sequences), batch_size):
        bend = min(len(sequences), bstart + batch_size)
        batch = sequences[bstart:bend]

        # ---------------------------------------------------
        # Get labels
        # ---------------------------------------------------
        answer_choices_ids = tokenizer.batch_encode_plus(
            [answer_choices] * len(batch), return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        # ---------------------------------------------------
        # Get example
        # ---------------------------------------------------
        seq_templates = [template.apply({"sentence": ex})[0] for ex in batch]
        sequences_ids = tokenizer.batch_encode_plus(
            seq_templates, return_tensors="pt", truncation=True, padding=True, add_special_tokens=False,
        )

        decoder_input_ids = torch.zeros((len(batch), 1), dtype=int, device=device)

        # Get next word prediction
        # note this code assumes answer choice ids is a single token
        out = model(input_ids=sequences_ids["input_ids"].to(device),
             attention_mask=sequences_ids["attention_mask"].to(device),
             decoder_input_ids=decoder_input_ids,
        ).logits

        out = torch.nn.functional.log_softmax(out, dim=-1)
        proba = torch.exp(torch.gather(out, 2, answer_choices_ids.unsqueeze(dim=1))).squeeze()
        proba = torch.divide(proba, proba.sum(dim=1).unsqueeze(dim=1))[:,1]

        results.extend(proba.cpu().detach().numpy().tolist())

    df["pos_sentiment"] = results
    return df




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filepath", required=True, type=str)
    parser.add_argument("--colname", required=True, type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    args = parser.parse_args()

    df = pd.read_csv(args.input_filepath, index_col=0)


    TOXIC_MODEL, TOXIC_DEVICE = load_toxicity_model(args.device)
    df = get_toxic_predictions(df, args.colname, TOXIC_MODEL, args.batch_size, TOXIC_DEVICE)
    TOXIC_MODEL = None

    SA_MODEL, SA_TOKENIZER, SA_DEVICE = load_sa_model(args.device)
    sa_templates, _ = get_templates_sa()
    df = get_sa_predictions(df, args.colname, sa_templates[0], SA_MODEL, SA_TOKENIZER, args.batch_size, SA_DEVICE)
    df.to_csv(args.input_filepath + ".properties")