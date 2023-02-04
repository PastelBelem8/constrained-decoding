from functools import partial
from frequencies import PositionalFrequencies
from collections import defaultdict

import argparse, joblib, json, os


def read_file(file):
    import zstandard as zstd
    key = 0
    with zstd.open(open(file, "rb"), "rt", encoding="utf-8") as f:
        for row in f:
            data = json.loads(row)
            yield key, data
            key += 1

    yield None, None

def load_tokenizer(model_name: str, num_tokens: int) -> callable:
    if "gpt-neo" in model_name or "gpt2" in model_name:
        # reference: https://huggingface.co/docs/transformers/model_doc/gpt_neo
        from transformers import GPT2TokenizerFast
        tokenizer_class = GPT2TokenizerFast
    else:
        raise NotImplemented

    # Load models
    tokenizer = tokenizer_class.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    tokenize = partial(
        tokenizer.batch_encode_plus,
        max_length=num_tokens,
        truncation=True,
        padding="max_length",
        add_special_tokens=False,
    )

    return tokenize


def filter_tokenizer_results(results: dict):
    """Filter tokens based on the masks."""
    batch_tokens = []

    for tokens, tokens_mask in zip(results["input_ids"], results["attention_mask"]):
        valid_tokens = [token for token, mask in zip(tokens, tokens_mask) if mask]
        batch_tokens.append(valid_tokens)

    # Returns: list[list[int]]
    # The first list constitutes batch_size list of tokens (list[int]).
    # Each list of tokens constitutes up to 15 ints.
    return batch_tokens


def compute_frequencies(frequencies: PositionalFrequencies, tokens: list, n: int):
    n_tokens = len(tokens)
    for i in range(0, n_tokens - n + 1, 1):
        ngram = tuple(tokens[i:i+n])
        frequencies.add(ngram, i)

def compute_ngrams(args, n: int=2, slice: int=300):
    tokenize = load_tokenizer(args.model_name, args.num_tokens)
    frequencies = defaultdict(PositionalFrequencies)

    while (data := read_file(args.file)) is not (None, None):
        num_file, doc = data
        subset = doc["meta"]["pile_set_name"]
        text = [doc["text"][:slice]] # Creating list to avoid breaking code

        tokenized_text = tokenize(text)
        batch_tokens = filter_tokenizer_results(tokenized_text)

        for tokens in batch_tokens:
            compute_frequencies(frequencies[subset], tokens, n=n)

        if num_file % 10_000 == 0:
            print(f"Processed {num_file}")
            break

    print(len(frequencies))

    for pile_subset, counts in frequencies:
        joblib.dump(counts, f"{args.output_dir}/{args.file}_{pile_subset}_{n}-counts.pkl")


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config-path",
        default="./scripts/configs/elastic-search.yml",
        help="Filepath for Elastic search configuration file.",
    )
    parser.add_argument(
        "-f", "--pile-file", type=str,
        required=True,
        help="Name of the PILE file to be counted"
    )
    parser.add_argument(
        "-ix",
        "--index",
        default="re_pile",
        help="Index in Elastic Search",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        default="EleutherAI/gpt-neo-125M",
        help="Model name",
    )
    parser.add_argument("-n", "--n-jobs", type=int, default=16, help="Number of CPUs.")
    parser.add_argument(
        "-K",
        "--num-tokens",
        type=int,
        default=30,
        help="Number of tokens to process from each document.",
    )
    parser.add_argument(
        "-f",
        "--freq",
        type=int,
        default=128,
        help="Number of documents to process at a time.",
    )

    parser.add_argument(
        "-n", "--ngram-size",
        type=int,
        default=2
    )

    parser.add_argument(
        "-o", "--output-dir", default="./temp", help="Path to a temporary directory."
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    compute_ngrams(args, args.ngram_size)