from typing import List, Optional

import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F


def assert_generative_model(model):
    """Checks whether the ``model`` is a valid autoregressive model.

    Notes
    -----
    This method is tailored for HuggingFace's Transformers API.

    Raises
    ------
    AttributeError: if the specified ``model`` is not a valid generative model
    """
    if model.get_output_embeddings() is None:
        raise AttributeError(
            "You tried to generate sequences with a model that does not have a LM Head."
            "Please use another model class (e.g. `OpenAIGPTLMHeadModel`,"
            "`XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`,"
            "`T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`,"
            "`T5ForConditionalGeneration`, `BartForConditionalGeneration` )"
        )


def create_history(
    n: int, inputs: List[int] = None, bos_token_id: Optional[int] = None
) -> torch.Tensor:
    """Replicates the ``inputs`` or ``bos_token_id``, called "history".

    Parameters
    ----------
    n: int
        Number of times to repeat the "history".

    inputs: List[int], optional (as long as ``bos_token_id`` is specified)
        The inputs that will be repeated ``n`` times.

    bos_token_id: int, optional (as long as ``inputs`` is specified)
        The begining of sequence token id to be repeated ``n`` times.

    Returns
    -------
    torch.Tensor with shape (n, max(len(inputs), 1))
        If inputs is specified we repeat the inputs n times, and the tensor
        is of size (n, len(inputs_len)). Otherwise, use `bos_token_id` in
        which case output shape is (n, 1).

    Raises
    ------
    AssertionError: if ``inputs`` is none and ``bos_token_id`` is ill-defined.
    """
    if inputs is None:
        assert isinstance(bos_token_id, int) and bos_token_id >= 0
        return torch.ones((n, 1), dtype=torch.long) * bos_token_id
    elif inputs.shape[0] == n:
        return inputs
    else:
        return inputs.repeat(n, 1)


def create_attn_mask(
    tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **_
) -> torch.Tensor:
    """Create an attention mask for the specified ``input_ids``"""
    # set non padded tokens to 1 if no attention mask is specified
    # and pad_token_id is defined for the specified `tokenizer`
    if (
        (attention_mask is None)
        and (tokenizer.pad_token_id is not None)
        and (tokenizer.pad_token_id in input_ids)
    ):
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    # if pad token is not defined or not in input_ids,
    # attention mask is all 1s
    elif attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    return attention_mask


def create_model_kwargs(inputs_tensor: torch.Tensor, model, tokenizer, **model_kwargs):
    """Create the model keyword arguments to be used during model.forward call.

    Parameters
    ----------
    inputs_tensor: tensor-like of shape [n, seq_len]
        The inputs to be encoded by the model (in case of encoder-decoder models).

    model: transformers LM model
        A valid generative LM model according to HuggingFace transformers.

    tokenizer: transformers.tokenizer
        A valid tokenizer corresponding to the model.

    kwargs: optional
        any model-specific keyword arguments to be passed to forward call.
        Check the official documentation of the desired model's "forward"
        method.

    Returns
    -------
    dict(str, Any)
        input_ids: torch.tensor
            The inputs to feed to the model for generation. For encoder-decoder
            models, these will differ from the input argument ``inputs_tensor``.
            For those models, ``inputs_tensor`` are fed into the encoder once and
            mantained constant throughout generation. Only the decoding inputs
            are changing during generation.

        model_kwargs: dict[str, Any]
            The keyword arguments to use throughout generation. This may include
            `encoder_outputs` for encoder-decoder only architectures, or other
            model specific keyword arguments.
    """
    batch_size = inputs_tensor.shape[0]
    # 1. create attention mask
    attention_mask = create_attn_mask(tokenizer, inputs_tensor, **model_kwargs)
    model_kwargs.update(attention_mask=attention_mask)

    # 2. get encoder (if encoder_decoder model)
    if model.config.is_encoder_decoder:
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, "input_ids"
        )

    # 3. Prepare `input_ids` which will be used for auto-regressive generation
    if model.config.is_encoder_decoder:
        input_ids = model._prepare_decoder_input_ids_for_generation(
            batch_size,
            decoder_start_token_id=model.config.decoder_start_token_id,
            bos_token_id=model.config.bos_token_id,
            model_kwargs=model_kwargs,
            device=inputs_tensor.device,
        )
    else:
        # if decoder-only then inputs_tensor has to be `input_ids`
        input_ids = inputs_tensor

    return {
        "input_ids": input_ids,
        "model_kwargs": model_kwargs,
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def naive_mc_estimate(
    max_num_tokens: int,
    input_ids,
    avoid_term_ids: list,
    model,
    tokenizer,
    model_kwargs,
):
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

    return 1 - samples_with_avoid_terms.mean().item()



@torch.no_grad()
def mc_estimate(
    max_num_tokens: int,
    input_ids,
    avoid_term_ids: list,
    model,
    tokenizer,
    model_kwargs,
    debug_kwargs=None,
):
    """Estimates the probability of generating sequences up to length ``max_num_tokens``
    such that no term in ``excluded_terms`` appears.

    It approximates this probability using a Monte Carlo Random Sampling estimation.
    We follow an importance-sampling approach, where we consider the proposal
    distribution q(.|x_{1:i}) to be the renormmalized probabilities of the
    complement set of the excluded_terms (i.e., at every time-step we explicitly
    avoid selecting any of the ``avoid_term_ids``).

    Pseudo-Algorithm
    ----------------
    **Goal**: compute the p(N_C(k)=0), where N_C(k) represents the number of times
    element in C occurs from steps 1 to k.

    # 1. Sample M sequences X_{1:k} that respect C not in X_{1:k}
    # 1.1. When sampling X_i after X_{1:i-1} use a proposal distribution q
    # 1.2. q(x_i = x | X_{1:i-1}) = { 0 if x in C or p(X_i=c | X_{1:i-1})}
    # 1.3. Randomly sample from q
    # 2. For each sequence X_{1:k}^(j) for j=1, ..., M
    # 2.1. Compute h_j = \product_{i=1}^k \sum_{c in C_complement} p(X_i^{j}=c | X_{1:i-1}^j)
    # 3. Compute p(N_c(k)=0) = 1/M sum h_j

    Parameters
    ----------
    max_num_tokens: int
        Maximum sequence length to generate.

    input_ids: tensor-like array of shape (n_samples, history_len)
        The history to condition the generation on. If we're using
        some templates and continuing generation from that, decoder
        methods will have the input_ids of size (n_samples, template_len),
        whereas encoder-decoder architectures will have (n_samples, 1).

    avoid_term_ids: list[int]
        List of token ids to avoid during sampling.

    model: transformers model
        Valid generative language models.

    tokenizer: transformers tokenizer
        Corresponding tokenizer.

    model_kwargs: dict
        Keyword arguments to use during generation of the continuations.

    debug_kwargs: dict, optional
        Keyword arguments to use for debugging purposes. If specified,
        they will be used to persist the intermediate results.
    """
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
        # 1. Create proposal distribution
        # ---------------------------------------------------------------------
        proposal = logits.clone()
        proposal[..., avoid_term_ids] = -np.inf

        # ---------------------------------------------------------------------
        # 2. Sample next token based on proposal distribution
        # ---------------------------------------------------------------------
        # Categorical.sample() returns a sampled index per each row.
        # samples is of shape (n_samples, 1)
        next_tokens = (
            torch.distributions.Categorical(logits=proposal).sample().unsqueeze(-1)
        )

        # ---------------------------------------------------------------------
        # 3. Accumulate log_probabilities
        # ---------------------------------------------------------------------
        proposal_log_prob = torch.gather(proposal, dim=-1, index=next_tokens)
        model_prob = F.softmax(logits, dim=-1)
        model_prob = 1 - model_prob[..., avoid_term_ids].sum(dim=-1)
        # ^Note: model_log_prob contains the probability that none of the
        # avoid_term_ids occurs...

        # ---------------------------------------------------------------------
        # 4. Handle EOS sequences:
        # ---------------------------------------------------------------------
        # - If sequence is finished, ignore sampled token and use padding.
        next_tokens = torch.where(unfinished_sequences, next_tokens, tokenizer.pad_token_id)
        model_prob = torch.where(unfinished_sequences, model_prob, 1)

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
        intermediate_model_log_prob += torch.log(model_prob)

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

        debug[i] = {
            "model_prob": model_prob.tolist(),
            "next_tokens": next_tokens.tolist(),
            "proposal_log_prob": proposal_log_prob.tolist(),
            "intermediate_model_log_prob": intermediate_model_log_prob.tolist(),
            "unfinished_sequences": unfinished_sequences.tolist(),
        }

        # If all sequences are finished (unfinished==0), don't keep generating
        if (unfinished_sequences == 0).all():
            print(f"Sequences finished prematurely ({i+1}/{max_num_tokens}).")
            break

    # -------------------------------------------------------------------------
    # 5. Compute probability of number of times element in C do not occur
    # -------------------------------------------------------------------------
    prob = torch.exp(intermediate_model_log_prob).mean().item()
    return prob # , debug


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

    terms_A_ids = tokenizer(terms_A, add_special_tokens=False).input_ids
    p_no_A_occurs = mc_estimate(
        avoid_term_ids=terms_A_ids,
        **create_model_kwargs(history, model, tokenizer),
        **mc_estimate_kwargs,
    )

    print(
        "Terms A", terms_A, f"(encoded {terms_A_ids}):", p_no_A_occurs
    )  # inflated because of decomposition into sub pieces

    terms_B_ids = tokenizer(terms_B, add_special_tokens=False).input_ids
    p_no_B_occurs = mc_estimate(
        avoid_term_ids=terms_B_ids,
        **create_model_kwargs(history, model, tokenizer),
        **mc_estimate_kwargs,
    )
    print("Terms B", terms_B, f"(encoded {terms_B_ids}):", p_no_B_occurs)

    terms_AB_ids = [terms_A_ids[0] + terms_B_ids[0]]
    p_no_AB_occurs = mc_estimate(
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
    # FIXME
    # - [ ] Model dump debugging structures
    # - [ ] Add argparse + yaml load
    # - [ ] Add batch_size
    # - [ ] Model loading (create load_model)
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    assert_generative_model(model)

    experiment_configs = {
        "seed": 42,
        "num_samples": 200,
        "input_str": "Man is to doctor",
        "max_num_tokens": 20,
        "terms_A": "nurse",
        "terms_B": "she",
    }

    log_odds(model=model, tokenizer=tokenizer, **experiment_configs)
