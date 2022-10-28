from main import *
import torch


@torch.no_grad()
def naive_mc_estimate(
    max_num_tokens: int,
    input_ids,
    avoid_term_ids: list,
    model,
    tokenizer,
    model_kwargs,
):
    avoid_term_ids = torch.tensor(avoid_term_ids).squeeze().unique().tolist()

    n_samples, samples = input_ids.shape[0], input_ids.clone()
    intermediate_model_log_prob = torch.zeros((n_samples, 1), dtype=torch.float32)
    unfinished_sequences = torch.ones((n_samples, 1), dtype=torch.bool)
    debug = {}

    for i in range(max_num_tokens):
        model_inputs = model.prepare_inputs_for_generation(samples, **model_kwargs)
        model_outputs = model.forward(**model_inputs)
        # logits: (n_samples, current_len, vocab_size)
        logits = model_outputs.logits
        # Select next token logits: (n_samples, vocab_size)
        logits = logits[:, -1, :]

        # ---------------------------------------------------------------------
        # 2. Sample next token based on proposal distribution
        # ---------------------------------------------------------------------
        # Categorical.sample() returns a sampled index per each row.
        # samples is of shape (n_samples, 1)
        next_tokens = (
            torch.distributions.Categorical(logits=logits).sample().unsqueeze(-1)
        )

        # ---------------------------------------------------------------------
        # 4. Handle EOS sequences:
        # ---------------------------------------------------------------------
        # - If sequence is finished, ignore sampled token and use padding.
        next_tokens = torch.where(unfinished_sequences, next_tokens, tokenizer.pad_token_id)

        # - Update the mask when you identify end of sequence tokens
        if tokenizer.eos_token_id is not None:
            # Set current unfinished to 1 if next token is not EOS
            current_unfinished = torch.where(
                next_tokens == tokenizer.eos_token_id, 0, 1
            )
            # Update previously unfinished sequences to be unfinished
            unfinished_sequences = torch.logical_and(
                unfinished_sequences, current_unfinished
            )

        # 5. Update intermediate artifacts
        samples = torch.cat([samples, next_tokens], dim=-1)  # FIXME: Double check this
        # ^Note: decoder-architectures will need the whole sequence at decoding time

        model._update_model_kwargs_for_generation(
            model_outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        # ---------------------------------------------------------------------
        # ^Note: This model call is model-specific and takes care of
        # retrieving the necessary information in `model_outputs` to
        # `model_kwargs`. In the case of T5-based model this will be
        # mostly using the decoders' `past-key-values` in
        # `model_outputs` as the `past` keyword argument in
        # model_kwargs. This avoid having to feed in the whole decoding
        # sequence at generation (thus making it faster).
        # ---------------------------------------------------------------------

        # If all sequences are finished (unfinished==0), don't keep generating
        if (unfinished_sequences == 0).all():
            print(f"Sequences finished prematurely ({i+1}/{max_num_tokens}).")
            break

    # -------------------------------------------------------------------------
    # 5. Compute probability of number of times element in C do not occur
    # -------------------------------------------------------------------------
    samples_with_avoid_terms = torch.isin(samples, test_elements=torch.tensor(avoid_term_ids), assume_unique=True)
    samples_with_avoid_terms = samples_with_avoid_terms.any(dim=-1)

    return 1.0 - samples_with_avoid_terms.float().mean().item()


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
    p_no_A_occurs = mc_estimate(
        avoid_term_ids=avoid_terms_ids,
        **create_model_kwargs(history, model, tokenizer),
        **mc_estimate_kwargs,
    )

    p_no_A_occurs_naive = naive_mc_estimate(
        avoid_term_ids=avoid_terms_ids,
        **create_model_kwargs(history, model, tokenizer),
        **mc_estimate_kwargs,
    )

    print(p_no_A_occurs_naive, p_no_A_occurs)
    return p_no_A_occurs_naive, p_no_A_occurs


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    assert_generative_model(model)

    configs = {
        "seed": 42,
        "num_samples": 1000,
        "input_str": None,
        "max_num_tokens": 2,
        "avoid_terms": "the a and or you he love",
    }

    test_success(
        model=model,
        tokenizer=tokenizer,
        **configs,
    )