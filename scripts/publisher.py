"""This is part of a multi-server pub/sub solution
to count the term frequencies of the PILE dataset.

The publisher will be responsible for iterating all
the documents and retrieve the ids of the documents.
It will create bulks of 128 ids that will be assigned
to the registered subscribers.

When subscribing to the publisher, subscribers will
communicate number of available threads/cores.

Implementation wise, the publisher will have two threads:
- main thread will receive requests from the subscribers
and will assign them id sets.
- secondary thread will be scanning documents ids,
pushing them to a queue.

In terms of structures, we will have two data structures:
- ids::Queue which contain the sets of ids to be assigned
to subscribers (and populated by the secondary thread).
- subscribers::Queue, which contains the ids of the
available subscribers.
"""
from data_objects import Data


class Publisher:
    pass