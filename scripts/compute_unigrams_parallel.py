from functools import partial
from es_utils import get_text, load, scroll, total_docs

import argparse, os, math
import multiprocessing as mp


def read_yaml_config(config_file: str) -> dict:
    import yaml
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def load_tokenizer(model_name: str, num_tokens: int) -> callable:
    if "gpt-neo" in model_name or "gpt2" in model_name:
        # reference: https://huggingface.co/docs/transformers/model_doc/gpt_neo
        from transformers import GPT2Tokenizer
        tokenizer_class = GPT2Tokenizer
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


def load_elastic_search(config_path: str, index, query):
    elastic_configs = read_yaml_config(config_path)
    es = load(**elastic_configs)
    n_docs = total_docs(es, index=index, query=query)

    return es, n_docs


def compute_unigrams(args, queue: mp.Queue):
    import time
    tokenize = load_tokenizer(args.model_name, args.num_tokens)

    while (data := queue.get(block=True)) is not None:
        # Wait if queue is being written to or until there are items in the queue
        print(os.getpid(), "got", len(data), "documents")


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config-path",
        default="./scripts/configs/elastic-search.yml",
        help="Filepath for Elastic search configuration file.",
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
        default=15,
        help="Number of tokens to process from each document.",
    )
    parser.add_argument(
        "-f",
        "--freq",
        type=int,
        default=5000,
        help="Number of documents to process at a time.",
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

    query = {"match_all": {}}
    engine, n_docs = load_elastic_search(args.config_path, args.index, query=query)

    # Create a task list
    tasks = mp.Queue(maxsize=50)
    results = []

    pool = mp.Pool(args.n_jobs, compute_unigrams, (args, tasks))

    # Iterate over the data
    data = iter(scroll(engine, query, size=args.freq))

    num_iters = math.ceil(n_docs / args.freq)

    for _ in range(num_iters):
        docs = next(data)
        tasks.put(docs, block=True)

    # https://stackoverflow.com/questions/69229049/why-does-multiprocessing-queue-get-block-after-the-queue-is-closed
    for _ in range(args.n_jobs):
        tasks.put(None)

    tasks.close()
    tasks.join_thread()

    pool.close()
    pool.join()