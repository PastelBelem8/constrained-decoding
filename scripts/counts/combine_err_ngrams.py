"""Combine ngrams counts per subset"""
from tqdm import tqdm
from typing import List

import argparse, joblib, os


def update_counts(new_counts, vals):
    for ngram, pos_counts in vals.counts.items():
        for pos, count in pos_counts.items():
            new_counts.add(ngram, pos, count)


if __name__ == "__main__":
    base_dir = "/srv/nvme0/ucinlp/cbelem/PILE-30-bigrams"
    subset_name = "Pile-CC"

    destination_file =  f"{base_dir}-agg/{subset_name}.pkl"

    err_files = [
        "05.jsonl.zst_Pile-CC_2-counts.pkl",
        "23.jsonl.zst_Pile-CC_2-counts.pkl"
    ]

    counts = joblib.load(f"{base_dir}-agg/{subset_name}.pkl")
    counts_before = counts.total_tokens
    for filename in tqdm(err_files):
        try:
            file_counts = joblib.load(f"{base_dir}/{filename}")
            update_counts(counts, file_counts)
            print(counts_before, "+", file_counts.total_tokens, "=", counts_before + file_counts.total_tokens, "vs.", counts.total_tokens)
            counts_before = counts.total_tokens
        except:
            print("Error in file:", filename)

    joblib.dump(counts, f"{base_dir}-agg/{subset_name}-corrected")
    print("Done!")
