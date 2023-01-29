"""Elastic-search related utils."""
from elasticsearch import Elasticsearch


def load(**kwargs) -> Elasticsearch:
    # Create an elastic search engine
    return Elasticsearch(**kwargs)


def total_docs(es: Elasticsearch, index: str, query: dict, size=1) -> int:
    data = es.search(index=index, query=query, size=size, track_total_hits=True)
    return data["hits"]["total"]["value"]


def scroll(es: Elasticsearch, query: dict, size: int=100, scroll: str="10m", **kwargs) -> dict:
    data = es.search(query=query, size=size, scroll=scroll, **kwargs)
    print("Total documents found", data["hits"]["total"])

    sid = data["_scroll_id"]
    scroll_size = len(data["hits"]["hits"])


    while scroll_size > 0:
        print("Processing", scroll_size, "documents")

        yield data["hits"]["hits"]

        data = es.scroll(scroll_id=sid, scroll="2m")
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