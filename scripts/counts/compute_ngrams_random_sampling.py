"""Combines ngrams counts

Algorithm uses a random sampling approach to avoid
exploding up the memory and taking so much time.

Ideally, we'll get a representative sample of the
most common expressions.
"""
from collections import defaultdict
from pathlib import Path

import argparse, jsonlines, os
import zstandard as zstd
import numpy as np
import logging


logger: logging.Logger = logging.getLogger(__name__)


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

    def drop_tail(self):
        self.save(head=False)

    def head_tail(self):
        # Ascending order
        ord_counts = sorted(
            self.ngram2counts.items(),
            key=lambda ngram: ngram[1]["total_counts"],
        )

        # Keep
        head = ord_counts[-self.max_ngrams:]
        tail = ord_counts[:-self.max_ngrams]
        return head, tail

    def save(self, head: bool=True):
        def __save_aux__(filepath, data, keep_in_memory=True):
            outputs = []
            for ngram, counts in data:
                # Serialize ngram tuple representation: (1, 2) --> "1, 2"
                ngram_str = ",".join((str(g) for g in ngram))

                # Create json object with counts
                result = {"ngram": ngram_str}
                result.update(counts)
                outputs.append(result)

                # Remove from memory
                if not keep_in_memory:
                    self.ngram2counts.pop(ngram)

            logger.info(f"Logging counts to file: {filepath}")
            with open(filepath, "wb") as f_out:
               with jsonlines.Writer(f_out, sort_keys=True) as writer:
                   writer.write_all(outputs)

        if len(self.ngram2counts) < self.max_ngrams:
            logger.warn(f"Haven't reached max_ngrams yet: {len(self.ngram2counts)} < {self.max_ngrams}")
            return

        filepath = self.counts_filepath
        head, tail = self.head_tail()

        # By default save head
        if head:
            __save_aux__(filepath, head, keep_in_memory=True)

        # keep an idea of how many ngrams have been processed and drop tail
        tail_filepath = f"{filepath}.tail_at_{self.all_ngrams}"
        __save_aux__(tail_filepath, tail, keep_in_memory=False)

        assert len(self.ngram2counts) == self.max_ngrams, \
            f"Total ngrams {len(self.ngram2counts)} but expected {self.max_ngrams} after save."


def read_file(filepath: str):
    import json

    key = 0
    with open(filepath, "rb") as f1:
        with zstd.open(f1, "rt", encoding="utf-8") as f:
            for row in f:
                data = json.loads(row)
                yield key, data
                key += 1

    yield None, None


def parse_ngrams(tokens, n: int):
    tokens = tokens["input_ids"]

    for i in range(0, len(tokens) - n + 1, 1):
        yield tuple(tokens[i:i+n])

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-file", "--pile-file", type=str,
        required=True,
        help="Path to the PILE file to be indexed"
    )
    parser.add_argument(
        "-model",
        "--model-name",
        default="EleutherAI/gpt-neo-125M",
        help="Model name",
    )

    parser.add_argument(
        "-p",
        "--docs-pct",
        type=float,
        default=0.1,
        help="Probability of tokenizing a document.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=874812,
        help="Probability of tokenizing a document.",
    )

    parser.add_argument(
        "-n", "--ngram-size",
        type=int,
        default=3
    )

    parser.add_argument(
        "-c", "--max-ngrams-in-memory",
        type=int,
        default=10_000,
        help="Number of ngrams to keep in memory at all times"
    )

    parser.add_argument(
        "-dtf", "--drop-tail-freq",
        type=int,
        default=500_000,
        help="How often to call drop tail in terms of documents processed",
    )

    parser.add_argument(
        "-o", "--output-dir", default="./temp", help="Path to a temporary directory."
    )
    return parser


def load_tokenizer(model_name: str) -> callable:
    model_name_lwr = model_name.lower()
    if "gpt-neo" in model_name_lwr or "gpt2" in model_name_lwr:
        # reference: https://huggingface.co/docs/transformers/model_doc/gpt_neo
        from transformers import GPT2TokenizerFast
        tokenizer_class = GPT2TokenizerFast
    else:
        raise NotImplemented

    # Load models
    logger.info(f"Loading {tokenizer_class.__name__}")
    tokenizer = tokenizer_class.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    return tokenizer

def setup_logger(path):
    global logger

    # Create handlers
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    c_format = logging.Formatter('[%(asctime)s][%(levelname)s]: %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    c_handler.setFormatter(c_format)

    f_handler = logging.FileHandler(path)
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter('[%(asctime)s][%(levelname)s]: %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    assert 0 < args.docs_pct <= 1

    # Create directory
    filename = args.pile_file.rpartition("/")[-1]
    filename = filename.split(".")[0]
    output_dir = f"{args.output_dir}/{filename}"
    os.makedirs(output_dir, exist_ok=True)

    setup_logger(f"{output_dir}/compute_ngrams.log")
    logger.debug(args)

    # Tokenization
    tokenizer: callable = load_tokenizer(args.model_name)

    # Data structure with the counts per subset
    counts = Counts(
        n=args.ngram_size,
        max_ngrams=args.max_ngrams_in_memory,
        out_dir=args.output_dir,
    )

    rand = np.random.default_rng(args.seed)
    preprocess_prob = args.docs_pct
    # Documents
    data_iter = iter(read_file(args.pile_file))

    processed_docs = 0
    while (data := next(data_iter)) != (None, None):
        num_file, doc = data

        # Randomly sample the chances of processing this document
        r: float = rand.random(1)[0]

        try:
            if r <= preprocess_prob:
                logger.info(f"Processing file id='{num_file}'")

                processed_docs += 1
                subset = doc["meta"]["pile_set_name"]

                # FIXME - not supporting longer sequences
                tokenized_text = tokenizer(doc["text"], truncation=True, max_length=tokenizer.max_len_single_sentence)
                ngrams = parse_ngrams(tokenized_text, n=args.ngram_size)
                for ngram in ngrams:
                    counts.add(ngram, subset, 1)

                if processed_docs % args.drop_tail_freq == 0:
                    logger.info(f"Dropping tail after {processed_docs} documents")
                    counts.drop_tail()
                    break

        except Exception as e:
              logger.error("Exception occurred", exc_info=True)
    counts.save()