from typing import List, Optional

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
def mc_estimate(
    max_num_tokens: int,
    input_ids,
    excluded_terms_ids: list,
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
    avoid selecting any of the excluded_terms_ids).

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

    excluded_terms_list: list[int]
        List of token ids to avoid during sampling.

    model: transformers model
        Valid generative language models.

    tokenizer: transformers tokenizer
        Corresponding tokenizer.

    model_kwargs:
        Keyword arguments to use during generation of the continuations.
    """
    samples = input_ids.clone().detach()
    intermediate_model_prob = torch.ones_like(input_ids, dtype=torch.float32)
    unfinished_sequences = torch.ones_like(input_ids, dtype=torch.float32)
    debug = {}
    for i in range(max_num_tokens):
        model_inputs = model.prepare_inputs_for_generation(samples, **model_kwargs)
        model_outputs = model.forward(**model_inputs)
        # logits: (n_samples, current_len, vocab_size)
        logits = model_outputs.logits.clone().detach()
        # Select next token logits: (n_samples, vocab_size)
        logits = logits[:, -1, :]
        logits = F.log_softmax(logits, dim=-1)

        # ---------------------------------------------------------------------
        # 1. Create proposal distribution
        # ---------------------------------------------------------------------
        proposal = logits.clone().detach()
        proposal[..., excluded_terms_ids] = -np.inf

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
        model_prob = 1 - model_prob[..., excluded_terms_ids].sum(dim=-1)
        # ^Note: model_log_prob contains the probability that none of the
        # excluded_terms_ids occurs...

        # ---------------------------------------------------------------------
        # 4. Handle EOS sequences:
        # ---------------------------------------------------------------------
        # - If sequence is finished, ignore sampled token and use padding.
        next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (
            1 - unfinished_sequences
        )
        model_prob = model_prob * unfinished_sequences + 1 * (1 - unfinished_sequences)

        # - Update the mask when you identify end of sequence tokens
        if tokenizer.eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul(
                (next_tokens != tokenizer.eos_token_id).long()
            )

        # 5. Update intermediate artifacts
        intermediate_model_prob *= model_prob
        samples = torch.cat([samples, next_tokens.long()], dim=-1)
        model._update_model_kwargs_for_generation(
            model_outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )

        debug[i] = {
            "model_prob": model_prob,
            "unfinished_sequences": unfinished_sequences,
            "proposal_log_prob": proposal_log_prob,
            "intermediate_model_prob": intermediate_model_prob.clone(),
        }

        # If all sequences are finished, don't keep generating
        if unfinished_sequences.sum() == 0:
            break

    # -------------------------------------------------------------------------
    # 5. Compute probability of number of times element in C do not occur
    # -------------------------------------------------------------------------
    prob = intermediate_model_prob.mean().item()
    return prob


if __name__ == "__main__":
    # set random seed
    set_seed(42)

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    assert_generative_model(model)
    encoder_input_str = "translate English to German: How old are you?"
    input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

    n_samples = 5
    excluded_terms = ["Sie"]
    excluded_terms_ids = tokenizer(excluded_terms, add_special_tokens=False).input_ids

    history = create_history(n_samples, input_ids, tokenizer.bos_token_id)
    model_kwargs = create_model_kwargs(history, model, tokenizer)
    result = mc_estimate(
        max_num_tokens=3,
        excluded_terms_ids=excluded_terms_ids,
        model=model,
        tokenizer=tokenizer,
        **model_kwargs,
    )
    print(result)
    print()
