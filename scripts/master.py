"""This is part of a multi-server Master-Slave solution
to count the term frequencies of the PILE dataset.

The Master will be responsible for iterating all
the documents and retrieve the ids of the documents.
It will create bulks of 128 ids that will be assigned
to the registered workers.

When subscribing to the Master, slaves/workers will
communicate number of available threads/cores.

Implementation wise, the Master will have two threads:
- main thread will receive requests from the workers
and will assign them id sets.
- secondary thread will be scanning documents ids,
pushing them to a queue.

In terms of structures, we will have two data structures:
- tasks::Queue which contain the sets of ids to be assigned
to available workers (and populated by the secondary thread).
- available_workers::Queue, which contains the ids of the
available workers.
"""
from multiprocessing import Lock
from scripts.representations import Data, ElasticSearchMixin
import queue as q

mutex = Lock()

class Master(ElasticSearchMixin):
    def __init__(self, num_ids: int, max_workers: int=1_000, max_ids: int=1_000, **kwargs):
        super().__init__(**kwargs)
        self.running = False

        self.num_ids = num_ids
        self.max_workers = max_workers
        self.tasks = q.Queue(maxsize=max_ids)
        self.available_workers = q.Queue(maxsize=max_workers)

        # Mapping <int> id --> <str> name of subscriber
        self.available_workers_ids = dict()

    def _estimate_scroll_time(self):
        # Apply a hard time estimate of the scroll time based on the
        # capacity of the queue: if queue is at 90% ot its capacity
        # set scroll_time to 20m, otherwise set it to "30s"
        capacity = (0.9 * self.tasks.maxsize)
        return "20m" if self.tasks.qsize() >= capacity else "30s"

    def start(self):
        self.running=True
        # FIXME: Launch thread to run publish (?)

    def assign(self):
        # TODO: Assign valid ids to workers

    def publish(self):
        scroll_id, data = None, [None]

        while self.running and len(data) > 0:
            scroll_time = self._estimate_scroll_time()
            data, scroll_id = self.scroll_all_docs(
                self.num_ids, scroll=scroll_time, scroll_id=scroll_id)

            if len(data) != 0:
                self.tasks.put(data)

    def subscribe(self, worker: str, n_jobs: int):
        with mutex:
            sid = self.available_workers_ids.setdefault(len(self.available_workers_ids), worker)

            for _ in range(n_jobs):
                self.available_workers.put(sid)

    def assign(self):


    def terminate(self):
        for worker_id in self.available_workers_ids.values():
            # FIXME - send request and terminate worker
            raise NotImplementedError

        self.running = False