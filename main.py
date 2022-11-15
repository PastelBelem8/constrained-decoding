from sampling import importance_sampling
from utils import assert_generative_model, create_history, create_model_kwargs, set_seed

import numpy as np


def log_odds(
    model,
    tokenizer,
    num_samples,
    max_num_tokens,
    terms_A,
    terms_B,
    input_str=None,
    seed=42,
):
    set_seed(seed)

    terms_A = [terms_A] if isinstance(terms_A, str) else terms_A
    terms_B = [terms_B] if isinstance(terms_B, str) else terms_B

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

    terms_A_ids = tokenizer(terms_A, add_special_tokens=False).input_ids
    p_no_A_occurs, _ = importance_sampling(  # TODO: Check return
        avoid_term_ids=terms_A_ids,
        **create_model_kwargs(history, model, tokenizer),
        **mc_estimate_kwargs,
    )

    print(
        "Terms A", terms_A, f"(encoded {terms_A_ids}):", p_no_A_occurs
    )  # inflated because of decomposition into sub pieces

    terms_B_ids = tokenizer(terms_B, add_special_tokens=False).input_ids
    p_no_B_occurs, _ = importance_sampling( # TODO: Check return
        avoid_term_ids=terms_B_ids,
        **create_model_kwargs(history, model, tokenizer),
        **mc_estimate_kwargs,
    )
    print("Terms B", terms_B, f"(encoded {terms_B_ids}):", p_no_B_occurs)

    terms_AB_ids = [terms_A_ids[0] + terms_B_ids[0]]
    p_no_AB_occurs, _ = importance_sampling( # TODO: Check return
        avoid_term_ids=terms_AB_ids,
        **create_model_kwargs(history, model, tokenizer),
        **mc_estimate_kwargs,
    )
    print(f"Terms A and B (encoded {terms_AB_ids}):", p_no_AB_occurs)
    print()

    numerator = 1 + p_no_AB_occurs - p_no_B_occurs - p_no_A_occurs
    denominator = (1 - p_no_B_occurs) * (1 - p_no_A_occurs)

    print(f"log({numerator}/{denominator}) = {np.log(numerator) - np.log(denominator)}")


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    assert_generative_model(model)

    experiment_configs = {
        "seed": 42,
        "num_samples": 100,
        "max_num_tokens": 20,
        # "input_str": "Man is to doctor what", "terms_A": "woman", "terms_B": "nurse",
        "input_str": "Premise: The accountant ate a bagel. Hypothesis: The",
        "terms_A": "man",
        "terms_B": "neutral",
    }

    log_odds(model=model, tokenizer=tokenizer, **experiment_configs)

    experiment_configs = {
        "seed": 42,
        "num_samples": 100,
        "max_num_tokens": 20,
        # "input_str": "Man is to doctor what", "terms_A": "woman", "terms_B": "nurse",
        "input_str": "Premise: The accountant ate a bagel. Hypothesis: The",
        "terms_A": "woman",
        "terms_B": "neutral",
    }
    log_odds(model=model, tokenizer=tokenizer, **experiment_configs)
