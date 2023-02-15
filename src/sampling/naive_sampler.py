from sampling.data_objects import SamplingOutput
from sampling.base import BaseSampler

import torch
import torch.nn.functional as F
import numpy as np


class NaiveSampler(BaseSampler):
    """Randomly sample tokens from the specified model."""

    def _sample_not_occur(
        self,
        input_ids,
        terms_ids,
        max_num_tokens,
        model_kwargs,
        return_logits=False,
    ):
        input_ids = input_ids.to(self.device)
        terms_ids = terms_ids.to(self.device)
        model_kwargs = {k: v.to(self.device) for k, v in model_kwargs.items()}

        n_samples, history_length = input_ids.shape
        samples = input_ids.clone()
        unfinished_sequences = torch.ones((n_samples, 1), dtype=torch.bool).to(
            self.device
        )

        all_logits = []
        for i in range(max_num_tokens):
            model_inputs = self.model.prepare_inputs_for_generation(
                samples, **model_kwargs, device=self.device
            )
            model_outputs = self.model.forward(**model_inputs)
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
            ).to(self.device)

            # ---------------------------------------------------------------------
            # 4. Handle EOS sequences:
            # ---------------------------------------------------------------------
            # If sequence is finished, ignore sampled token and use padding.
            next_tokens = torch.where(
                unfinished_sequences,
                next_tokens,
                self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

            # Update the mask when you identify end of sequence tokens
            if self.tokenizer.eos_token_id is not None:
                # Set current unfinished to 1 if next token is not EOS
                unfinished_sequences = torch.logical_and(
                    unfinished_sequences, next_tokens != self.tokenizer.eos_token_id
                )

            # 5. Update intermediate artifacts
            samples = torch.cat([samples, next_tokens], dim=-1)
            # ^Note: decoder-architectures will need the whole sequence at decoding time

            self.model._update_model_kwargs_for_generation(
                model_outputs,
                model_kwargs,
                is_encoder_decoder=self.model.config.is_encoder_decoder,
            )
            # ---------------------------------------------------------------------
            # ^Note: This model call is model-specific and takes care of retrieving
            # the necessary information in `model_outputs` to `model_kwargs`. In
            # the case of T5-based model this will be mostly using the decoders'
            # `past-key-values` in `model_outputs` as the `past` keyword argument
            # in model_kwargs. This avoid having to feed in the whole decoding
            # sequence at generation (thus making it faster).
            # ---------------------------------------------------------------------
            if return_logits:
                all_logits.append(F.log_softmax(logits, dim=-1).cpu().detach().numpy())

            # Stop whenever all sequences are finished (unfinished==0)
            if (unfinished_sequences == 0).all():
                print(f"Sequences finished prematurely ({i+1}/{max_num_tokens})!")
                break

        # -------------------------------------------------------------------------
        # 5. Compute probability of number of times element in C do not occur
        # -------------------------------------------------------------------------
        # prob_of_not_occurring at timestep k, implies that it shouldn't have
        # occurred <= timestep k.
        #
        # Mathematically, this means that
        #       P(N_c(K) = 0) = 1/N sum_{i=1..N} 1[ \cap_{j=1,...,K} S_ij=0]
        #
        # , where 1[] is the kronecker delta (or indicator function), which is one
        # only when the condition inside is true.
        #
        # Pseudo Algorithmm
        # ----------------
        # Inputs:
        # - S: num_sequences x max_num_tokens sample matrix S, where S_ij = 1
        # if sampled sequence i has one of the tokens in avoid_terms at position j,
        # and S_ij = 0 otherwise.
        # - C: set of terms to avoid.
        #
        # Outputs:
        # - probs: list of size max_num_tokens with the probabilities of neither
        # of the terms occurring at timestep k=1, ..., max_num_tokens
        # -------------------------------------------------------------------------

        # Determine which of the sequences contain any of the terms in "avoid_terms"
        samples_with_avoid_terms = torch.isin(
            samples[:, history_length:],
            test_elements=terms_ids,
        )
        # ^Note: samples[:, history_length:] aims to avoid counting tokens in the
        # prefix/history for models that require feeding the history as input.

        # Since the probability at timestep K requires that a term in avoid_terms
        # never occurred before, we will accumulate in the matrix that information
        # by stating whether any of the terms has occurred before timestep K.
        # (we use cummax for it, since the matrix is bound between 0s and 1s).
        samples_with_avoid_terms = torch\
            .cummax(samples_with_avoid_terms, dim=-1)[0]\
            .cpu().detach().numpy()

        # At timestep k, S_ij = 0 if none of the terms has occurred before or at
        # timestep k and it S_ij = 1 if one of the terms has occurred at least once
        # before.
        # Since we're interested in the probability of never occurring, we will
        # count the number of sequences that have no occurrence of terms up until
        # position k.
        probs_per_decoding_step = [
            (samples_with_avoid_terms[:,k] == 0).reshape(-1, 1)
            for k in range(max_num_tokens)
        ]

        return SamplingOutput(
            probs=probs_per_decoding_step,
            samples=samples[:, history_length:].cpu().detach().numpy(),
            terms_ids=terms_ids.cpu().detach().numpy(),
            desc="NaiveSampler._sample_not_occur",
            logits=all_logits,
        )

    def _sample_marginal(self, sampling_out: SamplingOutput, **kwargs) -> SamplingOutput:
        assert sampling_out.description == "NaiveSampler._sample_not_occur"
        assert len(sampling_out.probs) == sampling_out.samples.shape[1]

        # Marginal probability concerns the likelihood of choosing a term
        # at random from decoding step k and the term belonging to set C.

        # We can re-use samples from before to compute this
        samples_term_mask = np.isin(
            sampling_out.samples,
            test_elements=sampling_out.terms_ids,
        )
        num_tokens = len(sampling_out.probs)

        # For each decoding step k, if any of the terms in C
        # occurs, then mark it has 1
        marginals = [samples_term_mask[:, k] for k in range(num_tokens)]

        return SamplingOutput(
            probs=marginals,
            samples=sampling_out.samples,
            terms_ids=sampling_out.terms_ids,
            desc="NaiveSampler._sample_marginal",
            logits=sampling_out.logits,
        )

    def estimate_hit_probability(self, sampling_out: SamplingOutput) -> SamplingOutput:
        assert sampling_out.description == "NaiveSampler._sample_not_occur"
        assert len(sampling_out.probs) == sampling_out.samples.shape[1]

        samples_mask = np.isin(sampling_out.samples, test_elements=sampling_out.terms_ids)
        # Unlike torch, numpy has no cummax method, therefore we replace that
        # with the application of cumsum and then masking the output as whether
        # any of the elements happens 1 or more times, thus achieving the same
        # behavior as the torch.cummax
        samples_mask = samples_mask.cumsum(axis=1)
        samples_mask = (samples_mask >= 1)

        # Hit probabilities at timestep k refer to the probability of
        # not observing any of the elements in C before decoding step k
        # and observing them exactly at decoding step k.
        #
        # samples_mask matrix indicates the moment in which any of the
        # terms in terms_ids occur.
        hit_probs = [samples_mask[:, i]]
        for i in range(1, len(sampling_out.probs)):
            # Computing the hit probability implies that for columns before k
            # should be 0's and they become 1 at timestep k. Hence, since
            # samples_mask is a cumulative indication of whether any of the terms
            # already occurred in previous decoding steps, by subtracting the
            # current indicator column k to the previous one, we obtain the
            # exact number of rows whose first hitting time probability is at
            # decoding step k.
            hit_probs.append(samples_mask[:, i] - samples_mask[:, i-1])

        return SamplingOutput(
            probs=hit_probs,
            samples=sampling_out.samples,
            terms_ids=sampling_out.terms_ids,
            desc="NaiveSampler.estimate_hit_probability",
            logits=sampling_out.logits,
        )

    def estimate_hit_probability_A_before_B(*args, **kwargs):
        raise NotImplemented