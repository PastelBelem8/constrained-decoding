from sampling import importance_sampling, naive_sampling
from utils import assert_generative_model, create_history, create_model_kwargs, set_seed
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import torch


def test_success(
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
        input_ids = tokenizer(input_str, return_tensors="pt").input_ids
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

    avoid_terms_ids = tokenizer(avoid_terms, add_special_tokens=False).input_ids
    print("Avoid terms ids:", avoid_terms_ids)
    p_no_A_occurs, p_no_A_occurs_var = importance_sampling(
        avoid_term_ids=avoid_terms_ids,
        **create_model_kwargs(history, model, tokenizer),
        **mc_estimate_kwargs,
    )

    p_no_A_occurs_naive, p_no_A_occurs_naive_var = naive_sampling(
        avoid_term_ids=avoid_terms_ids,
        **create_model_kwargs(history, model, tokenizer),
        **mc_estimate_kwargs,
    )

    print("Naive Sampling:", p_no_A_occurs_naive, "\t var:", p_no_A_occurs_var)
    print("Importance sampling", p_no_A_occurs, "\t var:", p_no_A_occurs_naive_var)
    return p_no_A_occurs_naive, p_no_A_occurs


if __name__ == "__main__":

    model_name = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    assert_generative_model(model)

    configs = {
        "seed": 42,
        "num_samples": 500,
        "input_str": "I am",
        "max_num_tokens": 5,
        # [32099, 27, 3, 9, 736, 3, 5, 786, 388, 2335, 3202, 19121, 3, 1765, 4940, 22277]
        "avoid_terms": "<extra_id_1>",
    }

    test_success(
        model=model,
        tokenizer=tokenizer,
        **configs,
    )