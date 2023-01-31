"""This is part of a multi-server pub/sub solution
to count the term frequencies of the PILE dataset.

The publisher will be responsible for iterating all
the documents and retrieve the ids of the documents.
It will create bulks of 128 ids that will be assigned
to the registered subscribers.

When subscribing to the publisher, subscribers will
communicate number of available threads/cores.

In terms of implementation, the subscriber has:
- elastic server engine
- name
- number of jobs (n_jobs)

Since these are the actually workers, we will also
ensure that the subscriber will have a fail-safe
mechanism for the documents whose Deserialization
errs. We will create a file that contains their IDs.
"""
from data_objects import Data


class Subscriber:
    pass