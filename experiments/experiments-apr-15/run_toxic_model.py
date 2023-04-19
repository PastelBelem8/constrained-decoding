import pandas as pd
import numpy as np

import os

DEVICE = 2
TARGET_WORD = "muslim" #s2-2
# TARGET_WORD = "buddhist" #s2-3

BASE_DIR = "/extra/ucinlp1/cbelem/experiments-apr-15"

OUTPUT_DIR = f"{BASE_DIR}/toxicity_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from toxic_clf_model import ToxicCommentTagger
import torch

TOXIC_MODEL_DEVICE = f"cuda:{DEVICE}"
# It is a multilabel model (we will consider only the "toxic label", index 0)
# the other labels are as follow: toxic, severe_toxic, obscene, threat, insult, identity_hate
toxicity_model = ToxicCommentTagger(n_classes=6)
toxicity_model.load_state_dict(torch.load('./model.pth'))
toxicity_model.eval();
toxicity_model.to(TOXIC_MODEL_DEVICE)

def add_toxic_prediction(df: pd.DataFrame, sequences, colname: str, model, device, batch_size: int) -> pd.DataFrame:
    df = df.copy()
    result = model.predict(sequences, device=device, batch_size=batch_size)

    df[f"{colname}toxicity"]     = result[:,0]
    df[f"{colname}severe_toxic"] = result[:,1]
    df[f"{colname}obscene"]      = result[:,2]
    df[f"{colname}threat"]       = result[:,3]
    df[f"{colname}insult"]       = result[:,4]
    df[f"{colname}identity_hate"]= result[:,5]

    return df

print(TARGET_WORD)

MODEL_DIR = f"{BASE_DIR}/models/EleutherAI__pythia-70m"
MODEL_OUT_DIR = f"{OUTPUT_DIR}/models/EleutherAI__pythia-70m"
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

BASE_PATH = f"{MODEL_DIR}/{TARGET_WORD}_min_prefix.csv"
BASE_DATA = pd.read_csv(BASE_PATH, index_col=0)
print("Read data from", BASE_PATH, len(BASE_DATA))

top_p_mask = BASE_DATA["sampling_kwargs"].apply(lambda x: "top_p" in x)
BASE_DATA.loc[top_p_mask, "sampling"] = ["top-p"] * sum(top_p_mask)

# Let us process the toxicity by parts
sampling_types = sorted(BASE_DATA["sampling"].unique())

for sampling in sampling_types:
    data = BASE_DATA[BASE_DATA["sampling"] == sampling]
    data = data.dropna()
    sequences = (data["sequence"]).values.tolist()
    sequences_preds = add_toxic_prediction(data, sequences, "", toxicity_model, batch_size=32, device=TOXIC_MODEL_DEVICE)
    sequences_preds.to_csv(f"{MODEL_OUT_DIR}/{TARGET_WORD}_{sampling}.csv")
    print("Wrote file:", f"{MODEL_OUT_DIR}/{TARGET_WORD}_{sampling}.csv")
