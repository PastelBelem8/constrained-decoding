from fastapi import FastApi
from master import Master

import uvicorn
import configs as cfg


if __name__ == "__main__":

    master =  Master(**cfg.ELASTIC_SEARCH_CONFIGS)
    master_app = FastApi()

    # API
    @master_app.get(cfg.MASTER_START_ENDPOINT)
    def start(self):
        master.start()

        # TODO - Start two threads/processes (using the same instance):
        # 1. Run publish
        # 2. Run assign


    @master_app.post(cfg.MASTER_REGISTER_ENDPOINT)
    def register(name: str, n_jobs: int):
        assert n_jobs > 0, "Reported number of jobs < 0"
        return {
            "assigned_id": master.register(name, n_jobs),
        }


    @master_app.post(cfg.MASTER_COMPLETED_ENDPOINT)
    def mark_completed(worker_id: int):
        master.mark_completed(worker_id)


    @master_app.get(cfg.MASTER_TERMINATE_ENDPOINT)
    def terminate(id: int):
        n = master.terminate()
        print(f"Num of terminated workers: {n}")

        return {
            "num_terminated": n,
        }


    uvicorn.run("server:master_app", port=cfg.MASTER_PORT, log_level="info", reload=True)
