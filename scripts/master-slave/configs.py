ELASTIC_SEARCH_CONFIGS = {
    "config_path": "./configs/elastic_search.yml",
    "index": "re_pile",
}

MASTER_PORT = 8000
MASTER_HOST = f"ucinlp-dandelion.ics.uci.edu:{MASTER_PORT}"

# ------------------------------------------
# Master endpoints
# ------------------------------------------
MASTER_START_ENDPOINT = "/start"
MASTER_COMPLETED_ENDPOINT = "/completed"
MASTER_REGISTER_ENDPOINT = "/register"
MASTER_TERMINATE_ENDPOINT = "/terminate"

# ------------------------------------------
# Worker endpoints
# ------------------------------------------
WORKER_EXECUTE_ENDPOINT = "/execute"
WORKER_TERMINATE_ENDPOINT = "/terminate"
