from functools import partial
from tqdm import tqdm

import math
import joblib
import yaml

import es_utils
import frequencies


def read_yaml_config(config_file: str) -> dict:
    import yaml
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def filter_tokens(tokens, tokens_mask):
    """Filter tokens based on the mask"""
    return [token for token, mask in zip(tokens, tokens_mask) if mask]


def tokenization_function(model_name, tokenizer_kwargs, num_tokens) -> callable:
    tokenizer_kwargs = tokenizer_kwargs or {}

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


def compute_unigrams(es, index, query, size, data_size, tokenizer: callable):
    data = iter(es_utils.scroll(es, query, size=size, index=index))
    num_iters = math.ceil(data_size / size)

    total_processed = 0
    counts = frequencies.PositionalFrequencies()
    for _ in tqdm(range(num_iters)):
        docs = next(data)
        total_processed += len(docs)

        results = tokenizer([es_utils.get_text(d) for d in docs])

        for inpt, attn in zip(results["input_ids"], results["attention_mask"]):
            # Do not count padding tokens
            tokens = filter_tokens(inpt, attn)

            # Update counts
            for pos, token in enumerate(tokens):
                counts.add(token, pos)

    assert total_processed == data_size, f"Did not process all documents: {data_size} vs {total_processed}"
    joblib.dump(counts, "unigram_counts.pkl")


if __name__ == "__main__":
    model_name = "EleutherAI/gpt-neo-125M"
    num_tokens = 15

    # Load tokenizer
    tokenize = tokenization_function(model_name, None, num_tokens=num_tokens)

    # Load Elastic Search
    index = "re_pile"
    configs = read_yaml_config("/home/kat/Projects/PhD/constrained-decoding/configs/elastic-search.yml")
    es = es_utils.load(**configs)

    query = {"match_all": {}}
    n_docs = es_utils.total_docs(es, index=index, query=query)
    print("All docs:", n_docs)
    compute_unigrams(es, index, query, 10_000, n_docs, tokenize)
    # compute_unigrams(es, index, query, 1_000, 2_000, tokenize)