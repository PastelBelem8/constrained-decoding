"""Elastic-search utils."""
from elasticsearch import Elasticsearch


def load(cloud_id, api_key, retry_on_timeout=True, http_compress=True) -> Elasticsearch:
    # Create an elastic search engine
    return Elasticsearch(cloud_id=cloud_id, api_key=api_key, retry_on_timeout=retry_on_timeout, http_compress=http_compress)


def total_docs(es: Elasticsearch, index: str, query: dict, size=1) -> int:
    data = es.search(index=index, query=query, size=size, track_total_hits=True)
    return data["hits"]["total"]["value"]


def scroll_all_docs(engine: Elasticsearch, index: str, size: int=128, scroll_id: str=None, scroll_time: str="10m") -> dict:
    query = {"match_all": {}}

    if scroll_id is None:
        data = engine.search(
            index=index, query=query, size=size, scroll=scroll_time
        )
    else:
        data = engine.scroll(scroll_id=scroll_id, scroll=scroll_time)

        if len(data["hits"]["hits"]) == 0:
            engine.clear_scroll(scroll_id=scroll_id)

    # Get document identifiers
    return data["hits"]["hits"], data["_scroll_id"]



def get_name(d: dict) -> str:
    return d["_id"]

def get_subset(d: dict) -> str:
    return d["_source"]["meta"]["pile_set_name"]

def get_text(d: dict) -> str:
    return d["_source"]["text"]