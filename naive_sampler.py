from tqdm import tqdm
from base import BaseSampler

import torch
import torch.nn.functional as F


class NaiveSampler(BaseSampler):
    """Randomly sample tokens from the specified model."""

    def _sample(self, input_ids, avoid_terms_ids, max_num_tokens, model_kwargs):
        input_ids = input_ids.to(self.device)
        avoid_terms_ids = avoid_terms_ids.to(self.device)
        model_kwargs = {k: v.to(self.device) for k, v in model_kwargs.items()}

        n_samples, history_length = input_ids.shape
        samples = input_ids.clone()
        unfinished_sequences = torch.ones((n_samples, 1), dtype=torch.bool).to(self.device)

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
            model_prob = F.softmax(logits, dim=-1)
            model_prob_occur = (
                model_prob[..., avoid_terms_ids].sum(dim=-1).unsqueeze(-1)
            )
            self.model_prob_occur.append(model_prob_occur.clone())
            self.logits.append(F.log_softmax(logits, dim=-1))
            self.next_tokens.append(next_tokens)
            self.unfinished_sequences.append(unfinished_sequences.clone())

            # In naive sampling, the probability of occurring specific tokens
            # consists of computing how many times any of the terms happened
            # self.cum_model_log_prob_not_occur.append()  # FIXME: Implement this

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
            assume_unique=True,
        )
        # ^Note: samples[:, history_length:] aims to avoid counting tokens in the
        # prefix/history for models that require feeding the history as input.

        samples_with_avoid_terms = samples_with_avoid_terms.any(dim=-1)
        return 1.0 - samples_with_avoid_terms.float(), samples

    def estimate_hit_probability(self):
        raise NotImplementedError
