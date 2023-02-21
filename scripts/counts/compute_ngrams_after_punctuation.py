"""Combines ngrams counts

Algorithm uses a random sampling approach to avoid
exploding up the memory and taking so much time.

Ideally, we'll get a representative sample of the
most common expressions.
"""
from collections import defaultdict

import argparse, jsonlines, os, sys
import zstandard as zstd
import gzip
import logging


logger: logging.Logger = None

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "[%(asctime)s][%(levelname)s] - %(lineno)d: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        print(record.levelno)
        formatter = logging.Formatter(log_fmt,  datefmt="%d-%b-%y %H:%M:%S")
        return formatter.format(record)


# Add whitespace after the token to capture the "individual" words
TOKENS_OF_INTEREST = list(map(str.lower, map(lambda s: f"{s} ",  (
    # ----------------------------------------------
    # Punctuation
    # ----------------------------------------------
    "\\.", ",", "-", ":", "!", "\\?", ";",

    # ----------------------------------------------
    # Common words (https://www.collocates.info/iweb.asp)
    # ----------------------------------------------
    "Hello", "Here", "While", "For", "and",
    "The", "There", "This", "That",
    "What", "How", "Where",
    "Once", "end",
    "Thank",
    "If", "With", "By",
    # Verbs
    "can", "may", "want", "have", "like", "born",

    # ----------------------------------------------
    # Common words according to 14 billion word corpus
    # ----------------------------------------------
    "time", "some", "part", "not",
    # ----------------------------------------------
    # pronouns
    # ----------------------------------------------
    "I", "You", "He", "She", "They", "Them",
    # ----------------------------------------------
    # possessive pronouns
    # ----------------------------------------------
    "My", "Yours", "Hers", "His", "Theirs",
    # ----------------------------------------------
    # Other interesting words
    # ----------------------------------------------
    "doctor", "nurse", "model", "physician", "therapist",
    "dog", "cat", "horse", "butterfly", "bird", "fish"
    # Emotions
    "crazy", "happy", "sad", "love", "angry", "joy", "upset", "tired", "anxious", "horror", "fear", "tired", "pain", "calm",

    "negative", "positive", "great", "bad", "good", "terrible", "neutral",
    # Religion
    "christian", "jewish", "muslim",
))))

print("\n" * 8, "#" * 100, "\n", sorted(TOKENS_OF_INTEREST), "\n", "#" * 100, "\n" * 8)

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
    def __init__(self, n: int, max_ngrams: int = 10_000, out_dir = "./temp"):
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

    def add(self, ngram: tuple, subset: str, incr: int = 1):

        subset_id = SUBSET2ID[subset]
        self.ngram2counts[ngram]["total_counts"] += incr
        self.ngram2counts[ngram][subset_id] += incr
        self.all_ngrams += incr

    def head_tail(self):
        # Ascending order
        ord_counts = sorted(
            self.ngram2counts.items(),
            key=lambda ngram: ngram[1]["total_counts"],
            reverse=True,
        )

        # Keep
        head = ord_counts[:self.max_ngrams]
        tail = ord_counts[self.max_ngrams:]
        return head, tail

    def save(self, head: bool = True):
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

            if len(outputs) > 0:
                logger.info(f"Logging counts to file: {filepath}")
                with gzip.open(filepath, "wb") as f_out:
                    with jsonlines.Writer(f_out) as writer:
                        writer.write_all(outputs)

        if len(self.ngram2counts) < self.max_ngrams:
            logger.warning(
                f"Haven't reached max_ngrams yet: {len(self.ngram2counts)} < {self.max_ngrams}"
            )
            return

        filepath = self.counts_filepath
        head, tail = self.head_tail()

        # By default save head
        if head:
            __save_aux__(filepath, head, keep_in_memory=True)

        # keep an idea of how many ngrams have been processed and drop tail
        tail_filepath = f"{filepath}.tail_at_{self.all_ngrams}.gz"
        print("Before: dropping", len(self.ngram2counts), "{self.all_ngrams}")
        __save_aux__(tail_filepath, tail, keep_in_memory=False)
        print("After: dropping", len(self.ngram2counts), "{self.all_ngrams}")

        assert (
            len(self.ngram2counts) == self.max_ngrams
        ), f"Total ngrams {len(self.ngram2counts)} but expected {self.max_ngrams} after save."


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


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-file",
        "--pile-file",
        type=str,
        required=True,
        help="Path to the PILE file to be indexed",
    )
    parser.add_argument(
        "-model",
        "--model-name",
        default="EleutherAI/gpt-neo-125M",
        help="Model name",
    )

    parser.add_argument("-n", "--ngram-size", type=int, default=3)

    parser.add_argument(
        "-c",
        "--max-ngrams-in-memory",
        type=int,
        default=10_000,
        help="Number of ngrams to keep in memory at all times",
    )

    parser.add_argument(
        "-dtn",
        "--drop-tail-ngrams",
        type=int,
        default=200_000,
        help="When to drop tail in terms of ngrams in memory.",
    )

    parser.add_argument(
        "-o", "--output-dir", default="./temp", help="Path to a temporary directory."
    )
    return parser


def load_tokenizer(model_name: str, num_tokens) -> callable:
    from functools import partial

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

    return partial(
        tokenizer,
        max_length=num_tokens,
        truncation=True,
        padding="max_length",
    )


def setup_logger(path):
    global logger

    logger =  logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.DEBUG)
    c_handler.setFormatter(CustomFormatter())

    f_handler = logging.FileHandler(path)
    f_handler.setLevel(logging.INFO)
    f.handler.setFormatter(logging.Formatter(
        "[%(asctime)s][%(levelname)s]: %(message)s", datefmt="%d-%b-%y %H:%M:%S"
    ))

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    logger.info("HELLOOOOO")
    logger.error("HELLOOOOO")



def index_of_tokens(text: str, tokens: list):
    # inspired by https://docs.python.org/3/library/re.html#finding-all-adverbs-and-their-positions
    import re
    text = text.lower()
    regex_expr = "|".join(tokens)

    for m in re.finditer(regex_expr, text):
        # return first index of the token and the token itself
        yield m.start(), m.group(0)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # Create directory
    filename = args.pile_file.rpartition("/")[-1]
    filename = filename.replace(".", "")
    output_dir = f"{args.output_dir}/{filename}"
    os.makedirs(output_dir, exist_ok=True)

    setup_logger(f"{output_dir}/compute_ngrams.log")
    logger.debug(args)

    # Tokenization
    tokenizer: callable = load_tokenizer(args.model_name, args.ngram_size)

    # Data structure with the counts per subset
    counts = Counts(
        n=args.ngram_size,
        max_ngrams=args.max_ngrams_in_memory,
        out_dir=output_dir,
    )

    # Heuristic to define the next ngram_size tokens is that they
    # are encoded in terms of 20 characters.
    SIZE = 20 * args.ngram_size

    # Documents
    data_iter = iter(read_file(args.pile_file))

    processed_docs = 0
    while (data := next(data_iter)) != (None, None):
        num_file, doc = data

        try:

            processed_docs += 1
            subset = doc["meta"]["pile_set_name"]

            # Process text
            text = doc["text"]

            for text_id, token in index_of_tokens(text, TOKENS_OF_INTEREST):
                text_id = max(0, text_id-1) # to capture the idea of whether there's a space or not
                # Tokenize text
                tokenized_text = tokenizer(
                    text[text_id:text_id+SIZE]
                )

                # Collect only the n-gram after the punctuation
                ngram = tokenized_text["input_ids"]
                counts.add(tuple(ngram), subset, 1)

            if len(counts.ngram2counts) % args.drop_tail_ngrams == 0:
                logger.info(f"Num file: {num_file} | #(Unique ngrams):{len(counts.ngram2counts)} \n Dropping tail...")
                counts.save(head=True)

        except Exception as e:
            logger.error("Exception occurred", exc_info=True)

    logger.info("Saving final counts...")
    counts.save()
