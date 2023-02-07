"""Combine ngrams counts per subset"""
from tqdm import tqdm
from typing import List

import argparse, joblib, os


def group_by_subset(filenames: List[str]):
    filenames_per_subset = {}

    for filename in filenames:
        # filename is structured as follows:
        # <json.std_filename>_<subset>_<ngram-size>-<tokens-counts>
        subset = filename.split("_")[1]

        subset_filenames = filenames_per_subset.setdefault(subset, [])
        subset_filenames.append(filename)

    return filenames_per_subset

def update_counts(new_counts, vals):
    for ngram, pos_counts in vals.counts.items():
        for pos, count in pos_counts.items():
            new_counts.add(ngram, pos, count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subset", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/srv/nvme0/ucinlp/cbelem/PILE-30-bigrams"
    filenames_in_base_dir = os.listdir(base_dir)
    assert filenames_in_base_dir != [], f"Empty directory: {base_dir}"

    filenames_per_subset = group_by_subset(filenames_in_base_dir)
    subset = args.subset
    filenames = sorted(filenames_per_subset[subset])
    print("Aggregating", subset)
    counts = joblib.load(f"{base_dir}/{filenames[0]}")

    for filename in tqdm(filenames[1:]):
        try:
            with open(f"{base_dir}/{filename}", "rb") as f:
                file_counts = joblib.load(f)
            update_counts(counts, file_counts)
        except:
            print("Error in file:", filename)

    joblib.dump(counts, f"{base_dir}-agg/{subset}.pkl")
