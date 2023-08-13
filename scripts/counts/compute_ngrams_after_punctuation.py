"""Combines ngrams counts

Algorithm uses a random sampling approach to avoid
exploding up the memory and taking so much time.

Ideally, we'll get a representative sample of the
most common expressions.
"""
from collections import defaultdict
from datetime import datetime

import argparse, json, jsonlines, os, sys
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
        formatter = logging.Formatter(log_fmt,  datefmt="%d-%b-%y %H:%M:%S")
        return formatter.format(record)


# Add whitespace after the token to capture the "individual" words
TOKENS_OF_INTEREST = (
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
    " I ", "You", " He ", " She ", "They", "Them",
    # ----------------------------------------------
    # possessive pronouns
    # ----------------------------------------------
    "Mine", "Yours", "Hers", "His", "Theirs",
    # ----------------------------------------------
    # Other interesting words
    # ----------------------------------------------
    "doctor", "nurse", "model", "physician", "therapist",
    "dog", "cat", "horse", "butterfly", "bird", "fish"
    # Emotions
    "crazy", "happy", "sad", "love", "angry", "joy", "upset",  "fear", "pain",

    "negative", "positive", "great", "bad", "good", "terrible", "neutral",
    # Religion
    "christian", "jewish", "muslim",
)

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
    def __init__(self, n: int, out_dir = "./temp"):
        # Size of the ngram being processed
        self.ngram_size = n

        # dict<tuple, dict<subset_id, counts>>
        self.ngram2counts = defaultdict(default_init)

        # Output dir
        self.out_dir = out_dir
        self.counts_filedir = f"{self.out_dir}"
        self.counts_extension = ".jsonl.gz"
        self.last_filepath = None

    def add(self, prev_token: list, ngram: list, subset: str, incr: int = 1):
        subset_id = SUBSET2ID[subset]

        ngram = tuple(prev_token + ngram)
        self.ngram2counts[ngram]["prev_token"] = prev_token
        self.ngram2counts[ngram]["total_counts"] += incr
        self.ngram2counts[ngram][subset_id] += incr

    def sort_desc(self):
        return sorted(
            self.ngram2counts.items(),
            key=lambda ngram: tuple(reversed(ngram[1]["prev_token"])), # ngram[1] gets the values of ngrams2counts
        )

    def save(self, filename: str):
        def __save_aux__(filepath, data, keep_in_memory=True):
            outputs = []
            for ngram, counts in data:
                # Serialize ngram tuple representation: (1, 2) --> "1, 2"
                ngram_str = ",".join((str(g) for g in ngram))

                # Create json object with counts
                result = {"ngram": ngram_str}
                result.update(counts)
                outputs.append(result)

            if len(outputs) > 0:
                logger.info(f"Logging counts to file: {filepath}")
                with gzip.open(filepath, "wb") as f_out:
                    with jsonlines.Writer(f_out) as writer:
                        writer.write_all(outputs)

            if not keep_in_memory:
                self.ngram2counts = defaultdict(default_init)


        filepath = f"{self.counts_filedir}/{filename}{self.counts_extension}"
        data = self.sort_desc()

        logger.info(f"Before dropping: {len(self.ngram2counts)}")
        __save_aux__(filepath, data, keep_in_memory=False)
        logger.info(f"After dropping: {len(self.ngram2counts)}")

        # if self.last_filepath is not None and os.path.exists(filepath):
        #     logger.warn(f"Removing last filepath: {self.last_filepath}")
        #     os.remove(self.last_filepath)
        #     self.last_filepath = filepath
        self.last_filepath = filepath

        assert len(self.ngram2counts) == 0, f"Got {len(self.ngram2counts)} but expected 0"



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

    # Add the SLURM job id
    parser.add_argument("-j", "--job-id", type=int, required=True)

    parser.add_argument("-n", "--ngram-size", type=int, default=3)

    parser.add_argument(
        "-dtn",
        "--drop-tail-ngrams",
        type=int,
        default=1_000_000,
        help="When to drop tail in terms of ngrams in memory.",
    )

    parser.add_argument(
        "-ratio",
        "--char-per-token-ratio",
        type=int,
        default=20,
        help="To avoid tokenizing the whole documents we slice the text using the char-per-token-ratio * ngram-size."
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
    ), tokenizer.pad_token_id


def setup_logger(path):
    global logger

    logger =  logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(CustomFormatter())

    f_handler = logging.FileHandler(path)
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(logging.Formatter(
        "[%(asctime)s][%(levelname)s]: %(message)s", datefmt="%d-%b-%y %H:%M:%S"
    ))

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)


def index_of_tokens(text: str, tokens: list, end=True):
    # inspired by https://docs.python.org/3/library/re.html#finding-all-adverbs-and-their-positions
    import re
    text_lower = text.lower()

    # add word boundaries to make sure it only matches the words
    tokens = list(map(lambda x: r"(\b" + x.lower() + r"\b)", tokens))
    regex_expr = "|".join(tokens)

    for m in re.finditer(regex_expr, text_lower):
        m_start = max(0, m.start()-1) # capture's whether it's a beginning of a word
        # or in the middle of a word.
        orig_token = text[m_start:m.end()]

        # return first/last index of the token and the token itself
        # print("index_of_tokens -->", m.end() if end else m_start, orig_token, m.group(0))
        yield m.end() if end else m_start, orig_token


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # Create directory
    filename = args.pile_file.rpartition("/")[-1]
    filename = filename.replace(".", "")

    output_dir = f"{args.output_dir}/{filename}/{args.ngram_size}-gram"
    os.makedirs(output_dir, exist_ok=True)

    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(f"{output_dir}/compute_ngrams_{current_timestamp}_{args.job_id}.log")
    logger.info(args)

    # Tokenization
    tokenizer, pad_token_id = load_tokenizer(args.model_name, args.ngram_size)

    # Data structure with the counts per subset
    counts = Counts(
        n=args.ngram_size,
        out_dir=output_dir,
    )

    # ----------------------------------------------------------------------------
    #                               Metadata
    # ----------------------------------------------------------------------------
    # Write the tokens we're looking for
    with open(f"{output_dir}/PILE_subsets_{current_timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(SUBSET2ID, f, sort_keys=True)

    # Write the tokens we're looking for
    tokens = set(TOKENS_OF_INTEREST)
    tokens2ids = {t: tokenizer(t)["input_ids"] for t in tokens}

    with open(f"{output_dir}/initial_tokens_{current_timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(tokens2ids, f, sort_keys=True)


    # ----------------------------------------------------------------------------
    # Heuristic to define the next ngram_size tokens is that they
    # are encoded in terms of 20 characters.
    SIZE = args.char_per_token_ratio * args.ngram_size

    # Let us register a function that will run before aborting the program
    num_file = 0

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

            # Buffer (if an error occurs in the middle of processing the tokens for a
            # document, we will not add it, which means we will not corrupt the counts).
            # We can simply restart from this document without risking overcounting.
            doc_counts = []
            for text_id, token_text in index_of_tokens(text, TOKENS_OF_INTEREST, end=True):
                token = tokenizer(token_text)["input_ids"]
                token = [t for t in token if t != pad_token_id]

                # Collect only the n-gram after the token of interest
                # This ensures we always get 6, regardless of the number of tokens of token of interest
                tokenized_text = tokenizer(
                    text[text_id:text_id+SIZE]
                )

                ngram = tokenized_text["input_ids"]
                ngram = [t for t in ngram if t != pad_token_id]
                doc_counts.append((list(token), list(ngram), subset))

                print("list of ngrams -->", token_text, token, ngram)
            for prev_token, ngram, subset in doc_counts:
                counts.add(prev_token, ngram, subset, 1)

            if len(counts.ngram2counts) % args.drop_tail_ngrams == 0:
                logger.info(f"Num file: {num_file} | #(Unique ngrams): {len(counts.ngram2counts)}.")
                counts.save(filename=num_file)
                logger.info("Done!")

            break
        except Exception as e:
            logger.error("Exception occurred in file {num_file}", exc_info=True)

    logger.info("Saving final counts...")
    counts.save(filename=f"{num_file}_done")
