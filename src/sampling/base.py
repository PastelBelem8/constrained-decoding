""""""
from abc import ABC, abstractmethod
from sampling.utils import set_seed, create_history, create_model_kwargs
from typing import Union, List, Optional

import math
import torch
import sampling.utils_models as utils_models

# Type alias
Tensor = torch.Tensor


class BaseSampler(ABC):
    """Abstract sampling class.

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
        self,
        input_ids: Tensor,
        avoid_terms_ids: Tensor,
        max_num_tokens: int,
        model,
        tokenizer,
        model_kwargs: dict,
    ):
        """Class-specific sampling procedure that estimates the probability of the
        specified avoid_terms_ids not occurring in any position up to max_num_tokens.

        Notes
        -----
        Examples of class-specific procedures are random/naive sampling and
        importance sampling.

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
    def _sample_marginal(
        self,
        input_ids: Tensor,
        terms_ids: Tensor,
        max_num_tokens: int,
        model_kwargs: dict,
    ):
        """Compute the probability of any of terms appearing in any position."""
        raise NotImplementedError

    @abstractmethod
    def estimate_hit_probability(self, *args, **kwargs):
        """Estimates the hitting time probabilities of any of the terms.

        Notes
        -----
        It differs from the marginal probability in that it will estimate the
        probability that any of the terms appears for the first time at position
        k. This method computes this value for all values of K=1...<max_num_tokens>.
        """
        raise NotImplementedError

    @abstractmethod
    def estimate_hit_probability_A_before_B(
        self, terms_A: str, terms_B: str, history: Optional[str] = None, *args, **kwargs
    ):
        """Estimates the probability that any term in terms_A occurs before
        any of the terms in B when conditioned on the history.

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
        add_special_tokens: bool = False,
    ):
        """Estimates the probability that any of the terms occurs in any of
        the positions. That is, what's the probability of randomly selecting
        a token from our model distribution and picking one of the specified
        terms at each position k."""
        set_seed(seed)
        self.reset_intermediate_results()

        bos_token_id = (
            self.tokenizer.bos_token_id or self.model.config.decoder_start_token_id
        )

        input_ids = (
            self.tokenizer(
                input_str, return_tensors="pt", add_special_tokens=add_special_tokens
            ).input_ids
            if input_str is not None
            else None
        )

        terms_ids = self.tokenizer(
            terms, add_special_tokens=add_special_tokens
        ).input_ids

        history = create_history(num_sequences, input_ids, bos_token_id)
        sampling_specific_kwargs = create_model_kwargs(
            history, self.model, self.tokenizer
        )

        # Avoid duplicate ids (FIXME: May not make sense, when we add support for phrases)
        terms_ids = torch.tensor(terms_ids).squeeze().unique()

        results = self._sample_marginal(
            terms_ids=terms_ids,
            max_num_tokens=max_num_tokens,
            **sampling_specific_kwargs,
        )

        return results

    def _reset_intermediate_results(self):
        """Class-specific variables"""
        pass

    def estimate(
        self,
        input_str: str,
        avoid_terms: str,
        num_sequences: int,
        max_num_tokens: int,
        seed: int,
        add_special_tokens: bool = False,
    ):
        """Estimates the probability of avoid_terms not occurring in the
        next max_num_tokens.

        Notes
        -----
        Intermediate artifacts will be kept after the execution of this method.
        """
        set_seed(seed)
        self.reset_intermediate_results()

        bos_token_id = (
            self.tokenizer.bos_token_id or self.model.config.decoder_start_token_id
        )

        input_ids = (
            self.tokenizer(
                input_str, return_tensors="pt", add_special_tokens=add_special_tokens
            ).input_ids
            if input_str is not None
            else None
        )

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

    def compute_confidence_intervals(
        self, values: List[Tensor], width: int = 1
    ) -> tuple:
        """Calculate confidence intervals for the specified values.

        Notes
        -----
        When provided a list, the method assumes it is a K-sized list
        where each element is a tensor with N probability values.
        K would be the number of decoding steps and N would be the
        number of sampled sequences.
        """
        values = values if isinstance(values, list) else [values]

        means, stds = [], []
        for val in values:
            mean = val.mean().item()
            std = val.std().item() / math.sqrt(len(val))

            means.append(mean)
            stds.append(std)

        return means, stds

    def decode(
        self,
        samples: Tensor,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        **kwargs
    ) -> List[str]:
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
