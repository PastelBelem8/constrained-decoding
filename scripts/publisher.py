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
from multiprocessing import Lock
from scripts.representations import Data, ElasticSearchMixin
import queue as q

mutex = Lock()

class Publisher(ElasticSearchMixin):
    def __init__(self, num_ids: int, max_subscribers: int=1_000, max_ids: int=1_000, **kwargs):
        super(ElasticSearchMixin, self).__init__(**kwargs)

        self.num_ids = num_ids
        self.max_subscribers = max_subscribers
        self.ids = q.Queue(maxsize=max_ids)
        self.subscribers = q.Queue(maxsize=max_subscribers)

        # Maps int --> name
        self.subscribers_ids = dict()

    def publish(self):
        # need to have ES
        raise NotImplementedError

    def subscribe(self, name: str, n_jobs: int):
        with mutex:
            sid = self.subscribers_ids.setdefault(len(self.subscribers_ids), name)

            for _ in range(n_jobs):
                self.subscribers.put(sid)

    def terminate(self):
        for sub_id in self.subscribers_ids.values():
            # FIXME - send request and terminate subscriber
            raise NotImplementedError