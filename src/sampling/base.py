""""""
from abc import ABC, abstractmethod
from sampling.utils import set_seed, create_history, create_model_kwargs

import math
import torch
import sampling.utils_models as utils_models


class BaseSampler:
    """

    Attributes
    ----------
    model_name: str
        Name of the model to be used during sampling.

    model:
        Instance of the specified model name. The model used during sampling.
        It should be an autoregressive model, whose generation capabilities are
        implemented and available in huggingface.

    tokenizer:
        Instance of the tokenizer corresponding to the specified model_name.

    model_kwargs: optional[dict], defaults to None
        Additional model arguments to be passed when loading the model.

    tokenizer_kwargs: optional[dict], defaults to None
        Additional tokenizer arguments to be passed when loading the tokenizer.

    device: optional[str], defaults to "gpu" if available
        Text representation of the device to use during sampling.
    """

    def __init__(
        self,
        model_name: str,
        model_kwargs=None,
        tokenizer_kwargs=None,
        device=None,
        debug=False,
    ):
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}

        self.tokenizer, self.model = utils_models.load_model(
            model_name, self.tokenizer_kwargs, self.model_kwargs
        )
        self.device = utils_models.get_device(device)
        self.model.to(self.device)

        self.reset_intermediate_results()

    @abstractmethod
    def _sample(
        self, input_ids, avoid_terms_ids, max_num_tokens, model, tokenizer, model_kwargs
    ):
        """Specific sampling procedure (e.g., random, importance sampling) that estimates
        the probability of the specified avoid_terms_ids not occurring in any position up
        to max_num_tokens.

        Return
        ------
        probabilities: array-like of shape (n_samples, 1)
            The total probability that neither of the specified ids in avoid_terms_ids occurs.

        samples: array-like of shape (n_samples, max_num_tokens)
            The sampled sequences. Can be useful for debugging purposes. Note that in
            some cases, the sequences may end prematurely (more likely to happen for
            larger max_num_tokens values).
        """
        raise NotImplemented

    @abstractmethod
    def _sample_marginal(self, input_ids, terms_ids, max_num_tokens, model_kwargs):
        raise NotImplementedError

    @abstractmethod
    def estimate_hit_probability(self, *args, **kwargs):
        """Estimates the hitting time probabilities of any of the terms."""
        raise NotImplementedError

    @abstractmethod
    def estimate_hit_probability_A_before_B(self, terms_A, terms_B, history=None, *args, **kwargs):
        """Estimates the probability that any term in terms_A occurs before
        any of the terms in B.

        Notes
        -----
        terms_A and terms_B should be non-overlapping.
        """
        raise NotImplemented

    def estimate_marginals(
        self,
        input_str: str,
        terms: str,
        num_sequences: int,
        max_num_tokens: int,
        seed: int,
        add_special_tokens: bool=False,
    ):
        set_seed(seed)
        self.reset_intermediate_results()

        bos_token_id = (
            self.tokenizer.bos_token_id or self.model.config.decoder_start_token_id
        )

        input_ids = self.tokenizer(
            input_str, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids if input_str is not None else None

        terms_ids = self.tokenizer(terms, add_special_tokens=add_special_tokens).input_ids

        history = create_history(num_sequences, input_ids, bos_token_id)
        sampling_specific_kwargs = create_model_kwargs(
            history, self.model, self.tokenizer
        )

        # Avoid duplicate ids (FIXME: May not make sense, when we add support for phrases)
        avoid_terms_ids = torch.tensor(avoid_terms_ids).squeeze().unique()

        results = self._sample_marginal(
            terms_ids=terms_ids,
            max_num_tokens=max_num_tokens,
            **sampling_specific_kwargs,
        )

        return results

    def _reset_intermediate_results(self):
        pass

    def estimate(
        self,
        input_str: str,
        avoid_terms: str,
        num_sequences: int,
        max_num_tokens: int,
        seed: int,
        add_special_tokens: bool=False,
    ):
        """"""
        set_seed(seed)
        self.reset_intermediate_results()

        bos_token_id = (
            self.tokenizer.bos_token_id or self.model.config.decoder_start_token_id
        )

        input_ids = self.tokenizer(
            input_str, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids if input_str is not None else None

        avoid_terms_ids = self.tokenizer(
            avoid_terms, add_special_tokens=add_special_tokens
        ).input_ids
        # ^Note: some tokenizers encode the same term differently depending on
        # whether they are preceeded with a space or not
        # ----------------------------------------------------------------------
        # Update: On Jan 26th, we agreed that this should be left to the user
        # of the framework to handle. It would depend on the context in which
        # the word would appear.
        # ----------------------------------------------------------------------

        history = create_history(num_sequences, input_ids, bos_token_id)
        sampling_specific_kwargs = create_model_kwargs(
            history, self.model, self.tokenizer
        )

        # Avoid duplicate ids (FIXME: May not make sense, when we add support for phrases)
        avoid_terms_ids = torch.tensor(avoid_terms_ids).squeeze().unique()

        results = self._sample(
            avoid_terms_ids=avoid_terms_ids,
            max_num_tokens=max_num_tokens,
            **sampling_specific_kwargs,
        )

        return results

    def reset_intermediate_results(self):
        self.model_prob_occur = []
        self.logits = []
        self.next_tokens = []
        self.cum_model_log_prob_not_occur = []  # cumulative prob distribution
        self.unfinished_sequences = []
        self._reset_intermediate_results()

    def compute_confidence_intervals(self, values, width=1):
        values = values if isinstance(values, list) else [values]

        mean, lb, ub = [], [], []
        for val in values:
            mean = val.mean().item()
            std = val.std().item() / math.sqrt(len(val))

            mean.append(mean)
            ub.append(mean + width * std)
            lb.append(mean - width * std)

        return mean, lb, ub

    def decode(self, samples, skip_special_tokens=True, clean_up_tokenization_spaces=True, **kwargs):
        return self.tokenizer.batch_decode(
            samples,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            *kwargs,
        )

    def to_device(self, *args, **kwargs):
        # https://stackoverflow.com/questions/59560043/what-is-the-difference-between-model-todevice-and-model-model-todevice
        # Models' modification is in place but for tensors it creates a copy
        print("Setting device:", self.device)
        self.model.to(self.device)

        args = [a.to(self.device) for a in args]

        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, dict):
                new_kwargs[k] = {}
                for k2, v2 in v.items():
                    new_kwargs[k][k2] = v2.to(self.device)
            else:
                new_kwargs[k] = v.to(self.device)

        return args, kwargs
