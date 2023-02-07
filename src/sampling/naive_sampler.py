from tqdm import tqdm
from sampling.data_objects import SamplingOutput
from sampling.base import BaseSampler

import torch
import torch.nn.functional as F


class NaiveSampler(BaseSampler):
    """Randomly sample tokens from the specified model."""

    def _sample_not_occur(
        self,
        input_ids,
        avoid_terms_ids,
        max_num_tokens,
        model_kwargs,
        return_logits=False,
    ):
        input_ids = input_ids.to(self.device)
        avoid_terms_ids = avoid_terms_ids.to(self.device)
        model_kwargs = {k: v.to(self.device) for k, v in model_kwargs.items()}

        n_samples, history_length = input_ids.shape
        samples = input_ids.clone()
        unfinished_sequences = torch.ones((n_samples, 1), dtype=torch.bool).to(
            self.device
        )

        all_logits = []
        for i in tqdm(range(max_num_tokens)):
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
                all_logits.append(F.log_softmax(logits, dim=-1))

            # Stop whenever all sequences are finished (unfinished==0)
            if (unfinished_sequences == 0).all():
                print(f"Sequences finished prematurely ({i+1}/{max_num_tokens})!")
                break

        # -------------------------------------------------------------------------
        # 5. Compute probability of number of times element in C do not occur
        # -------------------------------------------------------------------------
        samples_with_avoid_terms = torch.isin(
            samples[:, history_length:],
            test_elements=avoid_terms_ids,
        )

        # ^Note: samples[:, history_length:] aims to avoid counting tokens in the
        # prefix/history for models that require feeding the history as input.
        samples_with_avoid_terms = torch.cumsum(samples_with_avoid_terms, dim=-1)

        probs_per_decoding_step = [
            1 - (samples_with_avoid_terms[:,:i].any(dim=-1))
            for i in range(max_num_tokens)
        ]

        return SamplingOutput(
            probs=probs_per_decoding_step,
            samples=samples[:, history_length:],
            terms_ids=avoid_terms_ids,
            desc="NaiveSampler._sample_not_occur",
            logits=all_logits,
        )

    def _sample_marginal(self, sampling_out: SamplingOutput) -> SamplingOutput:
        assert sampling_out.description == "NaiveSampler._sample_not_occur"
        assert len(sampling_out.probs) == sampling_out.samples.shape[1]

        samples_term_mask = torch.isin(
            sampling_out.samples,
            test_elements=sampling_out.terms_ids,
        )

        num_tokens = len(sampling_out.probs)
        marginals = [
            samples_term_mask[:, i].any(dim=-1)
            for i in range(num_tokens)
        ]

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

        samples_mask = torch.isin(
            sampling_out.samples, test_elements=sampling_out.terms_ids,
        )
        samples_mask = torch.cumsum(samples_mask, dim=-1)

        hit_probs = [samples_mask[:, i]]
        for i in range(1, len(sampling_out.probs)):
            hit_probs.append(samples_mask[:, i] - samples_mask[:, i-1])

        return SamplingOutput(
            probs=hit_probs,
            samples=sampling_out.samples,
            terms_ids=sampling_out.terms_ids,
            desc="NaiveSampler.estimate_hit_probability",
            logits=sampling_out.logits,
        )