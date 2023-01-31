"""Elastic-search utils."""
from elasticsearch import Elasticsearch


def load(cloud_id, api_key, retry_on_timeout=True, http_compress=True) -> Elasticsearch:
    # Create an elastic search engine
    return Elasticsearch(cloud_id=cloud_id, api_key=api_key, retry_on_timeout=retry_on_timeout, http_compress=http_compress)


def total_docs(es: Elasticsearch, index: str, query: dict, size=1) -> int:
    data = es.search(index=index, query=query, size=size, track_total_hits=True)
    return data["hits"]["total"]["value"]


def scroll(es: Elasticsearch, query: dict, size: int=100, scroll: str="10m", **kwargs) -> dict:
    data = es.search(query=query, size=size, scroll=scroll, **kwargs)
    print("Total documents found", data["hits"]["total"])

    sid = data["_scroll_id"]
    scroll_size = len(data["hits"]["hits"])

    while scroll_size > 0:
        # print("Processing", scroll_size, "documents")

        yield data["hits"]["hits"]

        data = es.scroll(scroll_id=sid, scroll=scroll)
        sid = data["_scroll_id"]

        scroll_size = len(data["hits"]["hits"])

    es.clear_scroll(scroll_id=sid)
    yield None


def get_name(d: dict) -> str:
    return d["_id"]

def get_subset(d: dict) -> str:
    return d["_source"]["meta"]["pile_set_name"]

def get_text(d: dict) -> str:
    return d["_source"]["text"]