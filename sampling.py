"""Sampling utils module."""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def importance_sampling(
    max_num_tokens: int,
    input_ids,
    avoid_term_ids: list,
    model,
    tokenizer,
    model_kwargs,
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
        model_prob = 1 - model_prob[..., avoid_term_ids].sum(dim=-1).unsqueeze(-1)
        # ^Note: model_log_prob contains the probability that none of the
        # avoid_term_ids occurs...

        # ---------------------------------------------------------------------
        # 4. Handle EOS sequences:
        # ---------------------------------------------------------------------
        # - If sequence is finished, ignore sampled token and use padding.
        next_tokens = torch.where(
            unfinished_sequences,
            next_tokens,
            tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        # - If sequence is finished set model_prob to 1 so it does not affect sum
        model_prob = torch.where(unfinished_sequences, model_prob, 1)

        # - Update the mask when you identify end of sequence tokens
        if tokenizer.eos_token_id is not None:
            unfinished_sequences = torch.logical_and(
                unfinished_sequences,
                # Set current unfinished to 1 if next token is not EOS
                next_tokens != tokenizer.eos_token_id,
            )

        # 5. Update intermediate artifacts
        intermediate_model_log_prob += torch.log(model_prob)

        samples = torch.cat([samples, next_tokens], dim=-1)
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
            "model_prob": model_prob.clone(),
            "next_tokens": next_tokens.clone(),
            "proposal_log_prob": proposal_log_prob.clone(),
            "intermediate_model_log_prob": intermediate_model_log_prob.clone(),
            "unfinished_sequences": unfinished_sequences.clone(),
        }

        # If all sequences are finished (unfinished==0), don't keep generating
        if (unfinished_sequences == 0).all():
            print(f"Sequences finished prematurely ({i+1}/{max_num_tokens}).")
            break

    # For easier access to last iteration (does not require knowing exact number)
    debug[-1] = debug[i]
    # -------------------------------------------------------------------------
    # 5. Compute probability of number of times element in C do not occur
    # -------------------------------------------------------------------------
    prob_mean = torch.exp(intermediate_model_log_prob).mean().item()
    prob_var = torch.exp(intermediate_model_log_prob).var().item()
    return prob_mean, prob_var, samples


@torch.no_grad()
def naive_sampling(
    max_num_tokens: int,
    input_ids: list,
    avoid_term_ids: list,
    model,
    tokenizer,
    model_kwargs,
) -> tuple:
    """Randomly sample tokens from the provided model, while avoiding specified tokens.

    Output
    ------
    mean: float
        Relative frequency of sequences that do not contain any of the terms in
        "avoid_term_ids".

    var: float
        The variance of the estimate.

    samples: array-like of shape (n_samples, max_num_tokens)
        The sampled sequences. Can be useful for debugging purposes. Note that in
        some cases, the sequences may end prematurely (more likely to happen for
        larger max_num_tokens values).
    """
    avoid_term_ids = torch.tensor(avoid_term_ids).squeeze().unique().tolist()

    n_samples, samples = input_ids.shape[0], input_ids.clone()
    unfinished_sequences = torch.ones((n_samples, 1), dtype=torch.bool)

    for i in range(max_num_tokens):
        model_inputs = model.prepare_inputs_for_generation(samples, **model_kwargs)
        model_outputs = model.forward(**model_inputs)
        # model logits: (n_samples, current_len, vocab_size)
        # Because we're interested in the next tokens [-1], we restrict it to size
        # (n_samples, vocab_size)
        logits = model_outputs.logits[:, -1, :]

        # ---------------------------------------------------------------------
        # 2. Sample next token based on proposal distribution
        # ---------------------------------------------------------------------
        # Categorical.sample() returns a sampled index per each row.
        # samples are of shape (n_samples, 1)
        next_tokens = (
            torch.distributions.Categorical(logits=logits).sample().unsqueeze(-1)
        )

        # ---------------------------------------------------------------------
        # 4. Handle EOS sequences:
        # ---------------------------------------------------------------------
        # If sequence is finished, ignore sampled token and use padding.
        next_tokens = torch.where(
            unfinished_sequences,
            next_tokens,
            tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        # Update the mask when you identify end of sequence tokens
        if tokenizer.eos_token_id is not None:
            # Set current unfinished to 1 if next token is not EOS
            unfinished_sequences = torch.logical_and(
                unfinished_sequences, next_tokens != tokenizer.eos_token_id
            )

        # 5. Update intermediate artifacts
        samples = torch.cat([samples, next_tokens], dim=-1)
        # ^Note: decoder-architectures will need the whole sequence at decoding time

        model._update_model_kwargs_for_generation(
            model_outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        # ---------------------------------------------------------------------
        # ^Note: This model call is model-specific and takes care of retrieving
        # the necessary information in `model_outputs` to `model_kwargs`. In
        # the case of T5-based model this will be mostly using the decoders'
        # `past-key-values` in `model_outputs` as the `past` keyword argument
        # in model_kwargs. This avoid having to feed in the whole decoding
        # sequence at generation (thus making it faster).
        # ---------------------------------------------------------------------

        # Stop whenever all sequences are finished (unfinished==0)
        if (unfinished_sequences == 0).all():
            print(f"Sequences finished prematurely ({i+1}/{max_num_tokens})!")
            break

    # -------------------------------------------------------------------------
    # 5. Compute probability of number of times element in C do not occur
    # -------------------------------------------------------------------------
    samples_with_avoid_terms = torch.isin(
        samples, test_elements=torch.tensor(avoid_term_ids), assume_unique=True
    )
    samples_with_avoid_terms = samples_with_avoid_terms.any(dim=-1)

    proba_mean = 1.0 - samples_with_avoid_terms.float().mean().item()
    proba_var = samples_with_avoid_terms.float().var().item()

    return proba_mean, proba_var, samples


if __name__ == "__main__":
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    from utils import *

    # User definitions
    model_name = "gpt2"
    seed = 42

    num_samples = 200
    input_str, avoid_terms = "I love", " the this that you u"


    # Load models
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, model_max_length=512)
    model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

    # Parse input and set seeds for reproducibility
    set_seed(seed)
    bos_token_id = tokenizer.bos_token_id or model.config.decoder_start_token_id
    input_ids = tokenizer(input_str, return_tensors="pt", add_special_tokens=False).input_ids

    avoid_terms_ids = tokenizer(avoid_terms, add_special_tokens=False).input_ids

    # History (or past observations) and model_kwargs will be the same for all queries
    history = create_history(num_samples, input_ids, bos_token_id)

    # Call IMPORTANCE Sampling
    mean, var, samples = importance_sampling(
        avoid_term_ids=avoid_terms_ids,
        **create_model_kwargs(history, model, tokenizer),
        max_num_tokens=5,
        model=model,
        tokenizer=tokenizer,
    )
