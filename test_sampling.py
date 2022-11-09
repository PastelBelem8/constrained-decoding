from sampling import importance_sampling, naive_sampling
from utils import assert_generative_model, create_history, create_model_kwargs, set_seed
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GPT2LMHeadModel


import numpy as np


def __naive_sampling_counts(input_str, avoid_terms, num_samples, max_num_tokens, seed, bos_token_id, model, tokenizer, threshold=10e-9):
    """Tests whether the counting methods are working properly."""
    def get_count_occurrences(samples, terms):
        # naive counting of number of sentences w/ one of the avoid terms
        counts = 0
        for i, sample in enumerate(samples):
            for token_id in sample:
                if token_id in terms:
                    #print(f"'{tokenizer.decode(token_id)}' appeared in sample {i}: '{tokenizer.decode(sample)}'")
                    counts+=1
                    break
        return counts

    set_seed(seed)


    # Note: some models (e.g., gpt2) have different representations for the same
    # term depending on whether they are preceeded with a space or not. Since in
    # general the tokens will be conditioned on some outputs, we do not need to
    # account for these usecases in particular except to add a space before the
    # avoid terms string. This fix is brittle, though, since different models can
    # have different encodings.
    avoid_terms = " ".join([f"{t} {t}" for t in avoid_terms.split()])

    input_ids = tokenizer(input_str, return_tensors="pt", add_special_tokens=False).input_ids
    avoid_terms_ids = tokenizer(avoid_terms, add_special_tokens=False).input_ids

    history = create_history(num_samples, input_ids, bos_token_id)
    mean, _, samples = naive_sampling(
        avoid_term_ids=avoid_terms_ids,
        **create_model_kwargs(history, model, tokenizer),
        max_num_tokens=max_num_tokens,
        model=model,
        tokenizer=tokenizer,
    )

    assert num_samples == len(samples), "Mismatch number of samples"
    num_samples_where_terms_occur = get_count_occurrences(samples, avoid_terms_ids)
    expected_proba = 1 - num_samples_where_terms_occur / num_samples
    assert 0 < expected_proba < 1, "Expected probability is either 0 or 1"
    estimate_gap = np.abs(mean - expected_proba)
    assert estimate_gap <= threshold, "Value error: error counting in naive sampling."


def test_success_naive_sampling_counts_with_gpt2():
    from transformers import GPT2Tokenizer

    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
    default_kwargs = {
        "num_samples": 200,
        "seed": 42,
        "model": model,
        "tokenizer": tokenizer,
        "bos_token_id": tokenizer.bos_token_id or model.config.decoder_start_token_id
    }

    input_str, avoid_terms = "I love", " the"
    __naive_sampling_counts(input_str, avoid_terms, max_num_tokens=5, **default_kwargs)

    input_str, avoid_terms = "Hello! Nice to", " meet you"
    __naive_sampling_counts(input_str, avoid_terms, max_num_tokens=5, **default_kwargs)

    input_str, avoid_terms = "I love", " the this that food"
    __naive_sampling_counts(input_str, avoid_terms, max_num_tokens=5, **default_kwargs)

    input_str, avoid_terms = "The movie was", " great terrible incredible too long"
    __naive_sampling_counts(input_str, avoid_terms, max_num_tokens=5, **default_kwargs)


def test_success_importance_sampling_with_gpt2():
    raise NotImplementedError

if __name__ == "__main__":
    test_success_naive_sampling_counts_with_gpt2()
    test_success_importance_sampling_with_gpt2()
    print("Success!")