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
from scripts.representations import Data, ElasticSearchMixin

import os


class Worker(ElasticSearchMixin):

    def __init__(self, name: str, n_jobs: int, master: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.n_jobs = n_jobs or os.cpu_count()
        self.master = master

    def subscribe(self):
        # TODO: Send http request to Master notifying name, n_jobs



    def execute(self, ids: list):
        raise NotImplemented