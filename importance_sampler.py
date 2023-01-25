from tqdm import tqdm
from base import BaseSampler

import numpy as np
import torch
import torch.nn.functional as F


class ImportanceSampler(BaseSampler):
    """Estimates the probability of generating sequences up to length
    ``max_num_tokens`` such that no term in ``excluded_terms`` appears.

    It approximates this probability using a Monte Carlo Random Sampling
    estimation. We follow an importance-sampling approach, where we consider
    the proposal distribution q(.|x_{1:i}) to be the renormmalized
    probabilities of the complement set of the excluded_terms (i.e., at every
    time-step we explicitly avoid selecting any of the ``avoid_term_ids``).

    Pseudo-Algorithm
    ----------------
    **Goal**: compute the p(N_C(k)=0), where N_C(k) represents the number
    of times element in C occurs from steps 1 to k.

    # 1. Sample M sequences X_{1:k} that respect C not in X_{1:k}
    # 1.1. When sampling X_i after X_{1:i-1} use a proposal distribution q
    # 1.2. q(x_i = x | X_{1:i-1}) = { 0 if x in C or p(X_i=c | X_{1:i-1})}
    # 1.3. Randomly sample from q
    # 2. For each sequence X_{1:k}^(j) for j=1, ..., M
    # 2.1. Compute h_j = \product_{i=1}^k \sum_{c in C_complement} p(X_i^{j}=c | X_{1:i-1}^j)
    # 3. Compute p(N_c(k)=0) = 1/M sum h_j

    Notes
    -----
    - The algorithm currently does not support sequences of tokens but rather
    a set of tokens. Which means that words like "terrorist" or phrases like
    "emergency room" are not currently supported. If "emergency room" is
    specified, the algorithm will prevent occurrences of "emergency" and "room"
    from occurring, i.e., it won't generate any sentence containing "emergency"
    nor "room", even if it's not related to "emergency room".

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

    model_kwargs: dict
        Keyword arguments to use during generation of the continuations.
    """

    def _sample(self, input_ids, avoid_terms_ids, max_num_tokens, model_kwargs):
        n_samples, samples = input_ids.shape[0], input_ids.clone()
        intermediate_model_log_prob = torch.zeros((n_samples, 1), dtype=torch.float32)
        unfinished_sequences = torch.ones((n_samples, 1), dtype=torch.bool)

        for i in tqdm(range(max_num_tokens)):
            model_inputs = self.model.prepare_inputs_for_generation(
                samples, **model_kwargs
            )
            model_outputs = self.model.forward(**model_inputs)
            # logits: (n_samples, current_len, vocab_size)
            logits = model_outputs.logits
            # Select next token logits: (n_samples, vocab_size)
            logits = logits[:, -1, :]

            # ---------------------------------------------------------------------
            # 1. Create proposal distribution
            # ---------------------------------------------------------------------
            proposal = logits.clone()
            proposal[..., avoid_terms_ids] = -np.inf

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
            model_prob = 1 - model_prob[..., avoid_terms_ids].sum(dim=-1).unsqueeze(-1)
            # ^Note: model_log_prob contains the probability that none of the
            # avoid_terms_ids occurs...

            # ---------------------------------------------------------------------
            # 4. Handle EOS sequences:
            # ---------------------------------------------------------------------
            # - If sequence is finished, ignore sampled token and use padding.
            next_tokens = torch.where(
                unfinished_sequences,
                next_tokens,
                self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            # - If sequence is finished set model_prob to 1 so it does not affect sum
            model_prob = torch.where(unfinished_sequences, model_prob, 1)

            # - Update the mask when you identify end of sequence tokens
            if self.tokenizer.eos_token_id is not None:
                unfinished_sequences = torch.logical_and(
                    unfinished_sequences,
                    # Set current unfinished to 1 if next token is not EOS
                    next_tokens != self.tokenizer.eos_token_id,
                )

            # 5. Update intermediate artifacts
            intermediate_model_log_prob += torch.log(model_prob)

            samples = torch.cat([samples, next_tokens], dim=-1)
            # ^Note: decoder-architectures will need the whole sequence at decoding time

            self.model._update_model_kwargs_for_generation(
                model_outputs,
                model_kwargs,
                is_encoder_decoder=self.model.config.is_encoder_decoder,
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
            self.model_prob_occur.append(1 - model_prob.clone())
            self.logits.append(F.log_softmax(logits, dim=-1))
            self.next_tokens.append(next_tokens.clone())
            self.cum_model_log_prob_not_occur.append(
                intermediate_model_log_prob.clone()
            )
            self.unfinished_sequences.append(unfinished_sequences.clone())
            self.proposal_log_prob.append(proposal_log_prob.clone())

            # If all sequences are finished (unfinished==0), don't keep generating
            if (unfinished_sequences == 0).all():
                print("=========================================================")
                print(f"Sequences finished prematurely ({i+1}/{max_num_tokens}).")
                print("=========================================================")
                break

        # -------------------------------------------------------------------------
        # 5. Compute probability of number of times element in C do not occur
        # -------------------------------------------------------------------------
        return torch.exp(intermediate_model_log_prob), samples

    def _reset_estimates(self):
        self.proposal_log_prob = []

    def _estimate_marginals(self, *args, **kwargs):
        if self.model_prob_occur == []:
            print(args, kwargs)
            if args or kwargs:
                self.estimate(*args, **kwargs)
            else:
                raise ValueError(
                    "Could not estimate marginals."
                    'Please call "ImportanceSampling.estimate" first.'
                )

        # --------------------------------------------------------------
        # Assumption: the intermediate artifacts have been stored
        # --------------------------------------------------------------
        num_tokens = self.model_prob_occur.shape[1]
        # hit_probs: hitting time probability, i.e., the probability
        # that the first time any of the terms appears is at timestep i
        hit_probs = []
        # miss_probs: cmf of misses, i.e., the probability that neither
        # of these terms occurs before timestep i
        miss_probs = []

        for i in tqdm(range(num_tokens)):
            prob_occur = self.model_prob_occur[i]

            if miss_probs == []:
                miss_probs.append(1 - prob_occur)
            else:
                miss_probs.append(miss_probs[-1] * (1 - prob_occur))

            if len(miss_probs) > 1:
                # Probability of terms not occurring before timestep i
                # and occurring exactly on timestep i
                prob_occur *= miss_probs[-1]

            hit_probs.append(prob_occur.mean().item())

        return hit_probs, miss_probs