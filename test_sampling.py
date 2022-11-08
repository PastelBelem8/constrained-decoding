from sampling import importance_sampling, naive_sampling
from utils import assert_generative_model, create_history, create_model_kwargs, set_seed
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GPT2LMHeadModel

import numpy as np


def _avoid_terms_test_case(
    model,
    tokenizer,
    num_samples,
    max_num_tokens,
    avoid_terms,
    input_str=None,
    seed=42,
):
    set_seed(seed)

    avoid_terms = [avoid_terms] if isinstance(avoid_terms, str) else avoid_terms

    if input_str is not None:
        input_ids = tokenizer(input_str, return_tensors="pt", add_special_tokens=False).input_ids
    else:
        input_ids = None

    bos_token_id = tokenizer.bos_token_id or model.config.decoder_start_token_id
    # History (or past observations) and model_kwargs will be the same
    # for all queries
    history = create_history(num_samples, input_ids, bos_token_id)

    # Common arguments to mc_estimate call
    mc_estimate_kwargs = {
        "max_num_tokens": max_num_tokens,
        "model": model,
        "tokenizer": tokenizer,
    }

    avoid_terms_ids = tokenizer(avoid_terms).input_ids
    print("Avoid terms ids:", avoid_terms_ids)

    # Compute naive_sampling
    p_no_A_occurs_naive = naive_sampling(
        avoid_term_ids=avoid_terms_ids,
        **create_model_kwargs(history, model, tokenizer),
        **mc_estimate_kwargs,
    )

    p_no_A_occurs = importance_sampling(
        avoid_term_ids=avoid_terms_ids,
        **create_model_kwargs(history, model, tokenizer),
        **mc_estimate_kwargs,
    )
    return p_no_A_occurs_naive, p_no_A_occurs


def assert_similar(proba_naive, proba_imp, n, threshold=10e-5):
    naive_mean, naive_var, naive_samples = proba_naive
    imp_mean, imp_var = proba_imp

    print("Absolute difference of means:", np.abs(naive_mean - imp_mean), f"(based on {n} samples)")
    print(" -> Naive:", round(naive_mean, 6))
    print(" -> Imp S:", round(imp_mean, 6), "\t var:", round(imp_var, 4))


def test_t5_model(model_name, input_str, avoid_terms, seed=42, num_samples=200, max_num_tokens=5):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    assert_generative_model(model)

    p_no_A_occurs_naive, p_no_A_occurs_imp = _avoid_terms_test_case(
        model=model,
        tokenizer=tokenizer,
        input_str=input_str,
        avoid_terms=avoid_terms,
        max_num_tokens=max_num_tokens,
        num_samples=num_samples,
        seed=seed,
    )

    assert_similar(p_no_A_occurs_naive, p_no_A_occurs_imp, num_samples)


def test_gpt2_model(model_name, input_str, avoid_terms, seed=42, num_samples=200, max_num_tokens=5):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    assert_generative_model(model)

    p_no_A_occurs_naive, p_no_A_occurs_imp = _avoid_terms_test_case(
        model=model,
        tokenizer=tokenizer,
        input_str=input_str,
        avoid_terms=avoid_terms,
        max_num_tokens=max_num_tokens,
        num_samples=num_samples,
        seed=seed,
    )

    assert_similar(p_no_A_occurs_naive, p_no_A_occurs_imp, num_samples)


def test_naive_sampling_success(model_name, input_str, avoid_terms):
    raise NotImplementedError
    # mock model.forward


def __naive_sampling_counts(input_str, avoid_terms, num_samples, max_num_tokens, seed, bos_token_id, model, tokenizer, threshold=10e-6):
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

    input_str, avoid_terms = "Hello! Nice to", "meet you"
    __naive_sampling_counts(input_str, avoid_terms, max_num_tokens=5, **default_kwargs)

    input_str, avoid_terms = "I love", "the this that food"
    __naive_sampling_counts(input_str, avoid_terms, max_num_tokens=5, **default_kwargs)

    input_str, avoid_terms = "The movie was", "great terrible incredible too long"
    __naive_sampling_counts(input_str, avoid_terms, max_num_tokens=5, **default_kwargs)


if __name__ == "__main__":
    test_success_naive_sampling_counts_with_gpt2()

    print("Success!")