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
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    tokenize = partial(
        tokenizer.batch_encode_plus,
        max_length=num_tokens,
        truncation=True,
        padding="max_length",
        add_special_tokens=False,
    )

    print("Tokenizer.pad_token:",tokenizer.pad_token)
    print("Tokenizer.pad_token_id:",tokenizer.pad_token_id)

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

def compute_ngrams_by_subset(args, n: int=2, slice: int=300):
    tokenize = load_tokenizer(args.model_name, args.num_tokens)
    frequencies = defaultdict(PositionalFrequencies)
    data_iter = iter(read_file(args.file))

    while (data := next(data_iter)) != (None, None):
        num_file, doc = data
        subset = doc["meta"]["pile_set_name"]
        if args.subset == "*" or args.subset == subset:
            text = [doc["text"][:slice]] # Creating list to avoid breaking code

            tokenized_text = tokenize(text)
            batch_tokens = filter_tokenizer_results(tokenized_text)

            for tokens in batch_tokens:
                compute_frequencies(frequencies[subset], tokens, n=n)

            if num_file % 1_000_000 == 0:
                print(f"{args.file} Computed {n}-grams in {num_file} documents")

    filename = args.file.rpartition("/")[-1]
    for pile_subset, counts in frequencies.items():
        with open(f"{args.output_dir}/{filename}_{pile_subset}_{n}-counts.pkl", "wb") as f:
            joblib.dump(counts, f)

def compute_ngrams(args, n: int=2, slice: int=300):
    tokenize = load_tokenizer(args.model_name, args.num_tokens)
    frequencies = PositionalFrequencies()
    data_iter = iter(read_file(args.file))

    while (data := next(data_iter)) != (None, None):
        num_file, doc = data
        text = [doc["text"][:slice]] # Creating list to avoid breaking code

        tokenized_text = tokenize(text)
        batch_tokens = filter_tokenizer_results(tokenized_text)

        for tokens in batch_tokens:
            compute_frequencies(frequencies, tokens, n=n)

        if num_file % 1_000_000 == 0:
            print(f"{args.file}: Computed {n}-grams in {num_file} documents")

    print(args.file, len(frequencies))

    filename = args.file.rpartition("/")[-1]
    filepath = f"{args.output_dir}/{filename}_all_subsets_{n}-counts.pkl"

    print("About to dump counts into", filepath)
    with open(filepath, "wb") as f:
        joblib.dump(frequencies, f)

    print("Done! Counts can be found at", filepath)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-file", "--file", type=str,
        required=True,
        help="Path to the PILE file to be indexed"
    )
    parser.add_argument(
        "-m",
        "--model-name",
        default="EleutherAI/gpt-neo-125M",
        help="Model name",
    )
    parser.add_argument(
        "-sub",
        "--subset",
        default="*",  # process all
        help="Subset name to count the frequencies or * if all should be accounted for.",
    )
    parser.add_argument(
        "-K",
        "--num-tokens",
        type=int,
        default=30,
        help="Number of tokens to process from each document.",
    )

    parser.add_argument(
        "-n", "--ngram-size",
        type=int,
        default=2
    )
    parser.add_argument(
        "-s", "--slice",
        type=int,
        default=300
    )

    parser.add_argument(
        "-o", "--output-dir", default="./temp", help="Path to a temporary directory."
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    print(args)
    compute_ngrams_by_subset(args, args.ngram_size, args.slice)