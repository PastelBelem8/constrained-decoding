"""This is part of a multi-server Master/Slave solution
to count the term frequencies of the PILE dataset.

The Master will be responsible for iterating all
the documents and retrieve the ids of the documents.
It will create bulks of 128 ids that will be assigned
to the registered workers.

When subscribing to the Master, slaves/workers will
communicate number of available threads/cores.

In terms of implementation, the worker has:
- elastic server engine
- name
- number of jobs (n_jobs)

Since these are the actually workers, we will also
ensure that the worker will have a fail-safe
mechanism for the documents whose Deserialization
errs. We will create a file that contains their IDs
that is periodically dumped.
"""
from representations import Data, ElasticSearchMixin, HTTPMixin
from frequencies import PositionalFrequencies

import functools as ft
import os


def filter_tokenizer_results(results: dict):
    """Filter tokens based on the masks."""
    batch_tokens = []

    for tokens, tokens_mask in zip(results["input_ids"], results["attention_mask"]):
        valid_tokens = [token for token, mask in zip(tokens, tokens_mask) if mask]
        batch_tokens.append(valid_tokens)

    # Returns: list[list[int]]
    # The first list constitutes batch_size list of tokens (list[int]).
    # Each list of tokens constitutes up to 15 ints.
    return batch_tokens


class Worker(ElasticSearchMixin, HTTPMixin):

    def __init__(self, name: str, n_jobs: int, master: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.n_jobs = n_jobs or os.cpu_count()
        self.master = master

        self.processed_docs = list()
        self.frequencies = PositionalFrequencies()

        self.tokenizer: callable = None #FIXME -- How to deal with multiprocessing?

    def _update_frequencies(self, tokens): #FIXME multiprocessing
        for pos, token in enumerate(tokens):
            self.frequencies.add(token, pos)

    def register(self):
        results = self.put_http(self.master, name=self.name, n_jobs=self.jobs) #FIXME: how to process results?

    def execute(self, data: Data, slice: int=200):
        docs: list = self.get(data)

        # Process docs to have up to K characters
        docs_text = [d["_source"]["text"][:slice] for d in docs]

        # Tokenizer
        tokenized_text = self.tokenizer(docs_text)
        batch_tokens = filter_tokenizer_results(tokenized_text)

        for tokens in batch_tokens:
            self._update_frequencies(tokens)

        self.completed()

    def completed(self):
        self.put_http(self.master + "/mark_completed", name=self.name)