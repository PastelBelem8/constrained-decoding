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
