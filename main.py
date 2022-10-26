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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def mc_estimate(max_len: int, n_samples: int, excluded_terms: list, history: torch.Tensor, model, tokenizer) -> float:
    """"""
    # TODO: Add validation for the history (condition on)
    # - if specified, history should be tensor (n_samples x 1)

    intermediate_model_prob = torch.ones_like(history)
    samples, unfinished_sequences = history, torch.ones_like(history)
    debug = {}
    # prepare_inputs_for_generation
    # model_kwargs = self._update_model_kwargs_for_generation(
    #            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
    # )
    for i in range(max_len):
        # logits: (n_samples, vocab_size)
        model_outputs = model.forward(samples)
        logits = model_outputs.logits.clone().detach()
        logits = F.log_softmax(logits, dim=-1)

        # ---------------------------------------------------------------------
        # 1. Create proposal distribution
        # ---------------------------------------------------------------------
        proposal = logits.clone().detach()
        proposal[..., excluded_terms] = -np.inf

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
        model_prob = 1 - model_prob[..., excluded_terms].sum(dim=-1)
        # ^Note: model_log_prob contains the probability that none of the
        # excluded_terms occurs...

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


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

encoder_input_str = "translate English to German: How old are you?"
input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

# ------------------------------------------------------------
outputs = model.generate(
     input_ids,
     do_sample=False,
     num_return_sequences=1,
     remove_invalid_values=True,
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(outputs[0], skip_special_tokens=True))



if __name__ == "__main__":
    # set random seed
    set_seed(42)

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    encoder_input_str = "translate English to German: How old are you?"
    input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

    n_samples = 2
    excluded_terms = ["Sie"]
    history = input_ids.repeat(n_samples, 1)
    result = mc_estimate(2, n_samples, excluded_terms, history, model, tokenizer)
    print(result)
    print()