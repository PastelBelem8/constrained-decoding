from fastapi import FastApi
from representations import Data
from worker import Worker

import configs as cfg


worker_app = FastApi()

worker = Worker(
    name= #TODO:ADD hostname,
    master=cfg.MASTER_HOST,
    **cfg.ELASTIC_SEARCH_CONFIGS,
)

@worker_app.post(cfg.WORKER_EXECUTE_ENDPOINT)
def execute(data: Data):
    raise NotImplemented

@worker_app.get(cfg.WORKER_TERMINATE_ENDPOINT)
def terminate():
    raise NotImplemented
