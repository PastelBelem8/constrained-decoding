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
from scripts.representations import Data, ElasticSearchMixin, HTTPMixin
import queue as q
import configs as cfg


class Master(ElasticSearchMixin, HTTPMixin):
    def __init__(self, num_ids: int=128, max_workers: int=1_000, max_ids: int=1_000, **kwargs):
        super().__init__(**kwargs)
        self.running = False

        self.num_ids = num_ids
        self.max_workers = max_workers
        self.tasks = q.Queue(maxsize=max_ids)
        self.available_workers = q.Queue(maxsize=max_workers)

        self.mutex = Lock()
        # Mapping <int> id --> <str> name of worker
        self.available_workers_ids = dict()

    def _estimate_scroll_time(self):
        # Apply a hard time estimate of the scroll time based on the
        # capacity of the queue: if queue is at 90% ot its capacity
        # set scroll_time to 20m, otherwise set it to "30s"
        capacity = (0.9 * self.tasks.maxsize)
        return "20m" if self.tasks.qsize() >= capacity else "30s"

    def start(self): # FIXME: Launch thread to run publish (?)
        self.running=True

    def assign_task(self):
        while self.running == True:
            worker_id = self.available_workers.get(block=True)

            with self.mutex:
                worker = self.available_workers_ids[worker_id]

            task = self.tasks.get(block=True)
            result = self.put_http(worker, data=task)
            # TODO: Process result?

    def publish(self):
        scroll_id, data = None, [None]

        while self.running and len(data) > 0:
            scroll_time = self._estimate_scroll_time()
            data, scroll_id = self.scroll_all_docs(
                self.num_ids, scroll=scroll_time, scroll_id=scroll_id)

            if len(data) != 0:
                self.tasks.put(data, block=True)

    def register(self, worker: str, n_jobs: int):
        with self.mutex:
            worker_id = self.available_workers_ids\
                .setdefault(len(self.available_workers_ids), worker)

        for _ in range(n_jobs):
            self.available_workers.put(worker_id, block=True)

        return worker_id

    def mark_completed(self, worker_id: int):
        self.available_workers.put(worker_id, block=True)

    def terminate(self):
        workers = []
        for worker in self.available_workers.values():
            terminated = self.put_http(worker + cfg.WORKER_TERMINATE_ENDPOINT)
            workers.append(terminated)

        self.running = False
        # TODO: Empty queues?
        return len(workers)