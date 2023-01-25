""""""
from abc import ABC, abstractmethod
from utils import set_seed, create_history, create_model_kwargs

import torch
import utils_models


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
    def _estimate_marginals(self):
        """Estimates the hitting time probabilities of any of the terms."""
        raise NotImplemented

    def _reset_estimates(self):
        pass

    def _parse_results(*args):
        return args

    def estimate(
        self,
        input_str,
        avoid_terms,
        num_sequences: int,
        max_num_tokens: int,
        seed: int,
        add_special_tokens=False,
    ):
        set_seed(seed)
        self.to_device(input_str, avoid_terms)

        bos_token_id = (
            self.tokenizer.bos_token_id or self.model.config.decoder_start_token_id
        )
        input_ids = self.tokenizer(
            input_str, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids

        avoid_terms_ids = self.tokenizer(
            avoid_terms, add_special_tokens=add_special_tokens
        ).input_ids
        # ^Note: some tokenizers encode the same term differently depending on
        # whether they are preceeded with a space or not
        # FIXME: Address this situation

        history = create_history(num_sequences, input_ids, bos_token_id)
        sampling_specific_kwargs = create_model_kwargs(
            history, self.model, self.tokenizer
        )

        results = self._sample(
            avoid_terms_ids=avoid_terms_ids,
            max_num_tokens=max_num_tokens,
            **sampling_specific_kwargs,
        )

        return self._parse_results(results)

    def reset_intermediate_results(self):
        self.model_prob_occur = []
        self.logits = []
        self.next_tokens = []
        self.cum_model_log_prob_not_occur = []  # cumulative prob distribution
        self.unfinished_sequences = []

    def to_device(self, *args):
        self.model.to(self.device)
        for arg in args:
            arg.to(self.device)
