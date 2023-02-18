"""Combine ngrams counts per subset"""
from collections import defaultdict
from pathlib import Path

import argparse, joblib, jsonlines, os


PILE_SUBSETS = (
    "ArXiv",
    "BookCorpus2",
    "Books3",
    "DM Mathematics",
    "Enron Emails",
    "EuroParl",
    "FreeLaw",
    "Github",
    "Gutenberg (PG-19)",
    "HackerNews",
    "NIH ExPorter",
    "OpenSubtitles",
    "OpenWebText2",
    "PhilPapers",
    "Pile-CC",
    "PubMed Abstracts",
    "PubMed Central",
    "StackExchange",
    "USPTO Backgrounds",
    "Ubuntu IRC",
    "Wikipedia (en)",
    "YoutubeSubtitles",
)
SUBSET2ID = {subset: i for i, subset in enumerate(PILE_SUBSETS)}


def read_jsonl_gz(filepath: str):
    import gzip, json

    with gzip.open(filepath, mode="rt", encoding="utf-8") as f:
        for row in f:
            data = json.loads(row)
            yield data
    yield None




def default_0():
    return 0

def default_init():
    return defaultdict(default_0)


class Counts:
    def __init__(self, n: int, max_ngrams: int = 10_000, out_dir: Path = "./temp"):
        # Tracks the total number of ngrams processed
        self.all_ngrams = 0

        # Maximum number of ngrams to keep in memory
        self.max_ngrams = max_ngrams

        # Size of the ngram being processed
        self.ngram_size = n

        # dict<tuple, dict<subset_id, counts>>
        self.ngram2counts = defaultdict(default_init)

        # Output dir
        self.out_dir = out_dir
        self.counts_filepath = f"{self.out_dir}/{n}-gram.jsonl.gz"

    def add(self, ngram: tuple, subset: str, incr: int=1):
        subset_id = SUBSET2ID[subset]
        self.ngram2counts[ngram]["total_counts"] += incr
        self.ngram2counts[ngram][subset_id] += incr
        self.all_ngrams += 0

    def dump(self):
        """Dump cache"""
        def marshal(ngram, counts):
            ngram_str = ",".join((str(g) for g in ngram))
            result = {"ngram": ngram_str}
            result.update(**counts)
            return result

        def unmarshal(obj) -> tuple:
            if obj is not None:
                return tuple(map(int, obj["ngram"].split(","))), obj
            else:
                return None, None

        temp_file = f"{self.counts_filepath}.temp"
        # Load previous file version (assume it's lexicographically ordered)
        prev_count_version = iter(read_jsonl_gz(self.counts_filepath))

        sorted_ngrams = sorted(self.ngram2counts.keys())

        with open(temp_file, "wb") as f_out:
            with jsonlines.Writer(f_out, sort_keys=True) as writer:

                # Write down the ngrams in lexicographic order
                buffer = []
                while (data := next(prev_count_version)) != None:
                    prev_ngram, prev_ngram_obj = unmarshal(data)

                    if len(sorted_ngrams) == 0:

                    while len(sorted_ngrams) > 0 and (curr_ngram := sorted_ngrams.pop(0)) < prev_ngram:
                        curr_ngram_obj = self.ngram2counts[curr_ngram]
                        buffer.append(marshal(curr_ngram, curr_ngram_obj))

                    # if ran out of sorted_ngrams


                    # if current ngram matches prev ngram, then update the counts
                    if curr_ngram == prev_ngram:
                        curr_ngram_obj = self.ngram2counts[curr_ngram]

                        updated_ngram = prev_ngram_obj

                        for key, counts in curr_ngram_obj.items():
                            updated_ngram[key] += counts

                        buffer.append(marshal(ngram, updated_ngram))







                        buffer.extend([
                            marshal(ngram, self.ngram2counts[ngram]) for ngram in sorted_ngrams[:id]])



                        writer.write_all(buffer)

                        buffer = []
                        sorted_ngrams = sorted_ngrams[id+1:]




        # todo: rename temp_file to self.counts_filepath










                for ngram in sorted_ngrams:
                    if prev_ngram is None or prev_ngram > ngram:
                        ngram_obj = self.ngram2counts[ngram]
                        buffer.append(marshal(ngram, ngram_obj))
                    elif prev_ngram < ngram:
                        buffer.append(prev_ngram_obj)
                        prev_ngram, prev_ngram_obj = unmarshal(next(prev_count_version))
                    else:
                        # update counts and dump buffer
                        updated_ngram = prev_ngram

                        for key, counts in self.ngram2counts[ngram].items():
                            updated_ngram[key] += counts

                        buffer.append(updated_ngram)


                IF
                if len(buffer) > 0:
                    writer.write_all(buffer)
                    buffer = []


        # ---------------------------------------------------------------------------------
        # Store information in jsonlines, compressed
        # ---------------------------------------------------------------------------------
        # {ngram_i: (t1,...,tn), total_counts: N, subset_id_0: counts_subset_id, ... subset_id_21: counts_subset_id}
        # ...
        # {ngram_k: (t1,...,tn), total_counts: N, subset_id_0: counts_subset_id, ... subset_id_21: counts_subset_id}
        # ---------------------------------------------------------------------------------







if not self.out_filepath.parent.exists():
    os.makedirs(self.out_filepath.parent)
else:
    print(self.out_filepath.parent, "already exists...")






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument
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
