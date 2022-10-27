# Goal: Compute proba N_C(k) being 0,
# where N_C(k): number of times element in C occurs from steps 1 to k;
# ----------------------------------------------------------------------------
# Importance sampling algorithm to compute N_C(k)=0
# ----------------------------------------------------------------------------
# 1. Sample M sequences X_{1:k} that respect C not in X_{1:k}
# 1.1. When sampling X_i after X_{1:i-1} use a proposal distribution q
# 1.2. q(x_i = x | X_{1:i-1}) = { 0 if x in C or p(X_i=c | X_{1:i-1})}
# 1.3. Randomly sample from q
# 2. For each sequence X_{1:k}^(j) for j=1, ..., M
# 2.1. Compute h_j = \product_{i=1}^k \sum_{c in C} p(X_i^{j}=c | X_{1:i-1}^j)
# 3. Compute p(N_c(k)=0) = 1/M sum h_j
# -----------------------------------------------------------------------------
import numpy as np
import random
import torch
import torch.nn.functional as F

def assert_generative_model(model):
    # cannot generate if the model does not have a LM head
    if model.get_output_embeddings() is None:
        raise AttributeError(
            "You tried to generate sequences with a model that does not have a LM Head."
            "Please use another model class (e.g. `OpenAIGPTLMHeadModel`,"
            "`XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`,"
            "`T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`,"
            "`T5ForConditionalGeneration`, `BartForConditionalGeneration` )"
        )


def create_history(n_samples: int, inputs: list=None, bos_token_id: int=None):
    if inputs is None:
        assert isinstance(bos_token_id, int) and bos_token_id >= 0
        return torch.ones((n_samples, 1), dtype=torch.long) * bos_token_id
    elif inputs.shape[0] == n_samples:
        return inputs
    else:
       return inputs.repeat(n_samples, 1)


def create_attn_mask(tokenizer, input_ids, attention_mask=None, **_) -> torch.Tensor:
    if (attention_mask is None) \
        and (tokenizer.pad_token_id is not None) \
            and (tokenizer.pad_token_id in input_ids):
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    elif attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    return attention_mask

def create_model_kwargs(inputs_tensor, model, tokenizer, **model_kwargs):
    batch_size = inputs_tensor.shape[0]
    # 1. create attention mask
    attention_mask = create_attn_mask(tokenizer, inputs_tensor, **model_kwargs)
    model_kwargs.update(attention_mask=attention_mask)

    # 2. get encoder (if encoder_decoder model)
    if model.config.is_encoder_decoder:
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(inputs_tensor, model_kwargs, "input_ids")

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
def mc_estimate(max_num_tokens: int, input_ids, excluded_terms_ids: list, model, tokenizer, model_kwargs) -> float:
    """"""
    # TODO: Add validation for the history (condition on) - if specified, history should be tensor (n_samples x 1)
    assert_generative_model(model)

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
        logits = logits[:,-1,:]
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
        next_tokens = torch.distributions.Categorical(logits=proposal).sample().unsqueeze(-1)

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
        next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (1 - unfinished_sequences)
        model_prob = model_prob * unfinished_sequences + 1 * (1 - unfinished_sequences)

        # - Update the mask when you identify end of sequence tokens
        if tokenizer.eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens != tokenizer.eos_token_id).long())

        # 5. Update intermediate artifacts
        intermediate_model_prob *= model_prob
        samples = torch.cat([samples, next_tokens.long()], dim=-1)
        model._update_model_kwargs_for_generation(model_outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder)

        debug[i] = {
            "model_prob": model_prob,
            "unfinished_sequences": unfinished_sequences,
            "proposal_log_prob": proposal_log_prob,
            "intermediate_model_prob": intermediate_model_prob.clone(),
        }

    # -------------------------------------------------------------------------
    # 5. Compute probability of number of times element in C do not occur
    # -------------------------------------------------------------------------
    prob = intermediate_model_prob.mean().item()
    return prob


if __name__ == "__main__":
    # set random seed
    set_seed(42)

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    assert_generative_model(model)
    encoder_input_str = "translate English to German: How old are you?"
    input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

    n_samples = 2
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