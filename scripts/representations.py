from dataclasses import dataclass
from json.decoder import JSONDecodeError
from io_utils import read_yaml_config

import requests
import traceback
import es_utils as es


@dataclass
class Data:
    ids: list

    def len(self):
        return len(self.ids)


class ElasticSearchMixin:
    def __init__(self, config_path, index: str):
        self.configs = read_yaml_config(config_path)
        self.engine = es.load(**self.configs)

        self.index = index

        indices = self.engine.indices.get(index="*").keys()
        indices = [i for i in indices if not i.startswith(".")]
        assert index == "*" or index in indices, \
            f"Invalid index: {index}. Expected '*' or one of the following {indices}"

        self.errored_docs = list()

    def get(self, data: Data):
        try:
            # Get multiple documents from elastic search
            return self.engine.m_get(index=self.index, docs=data.ids)["docs"]

        except JSONDecodeError as e:
            self.errored_docs.extend(data.ids)
            traceback.print_exception(e)
            return []

    def scroll_all_docs(self, size: int=128, scroll: str="30s", scroll_id: str=None):
        query = {"match_all": {}}

        if scroll_id is None:
            data = self.engine.search(
                index=self.index, query=query, size=size, scroll=scroll
            )
        else:
            data = self.engine.scroll(scroll_id=scroll_id, scroll=scroll)

            if len(data["hits"]["hits"]) == 0:
                self.clear_scroll(scroll_id=scroll_id)

        # Get document identifiers
        docs = [doc["_id"] for doc in data["hits"]["hits"]]
        return Data(docs), data["_scroll_id"]


@dataclass
class HTTPMixin:
    url: str

    def get_http(self, endpoint: str, **kwargs):
        r = requests.get(self.url + endpoint, json=kwargs)
        print(f"Status Code: {r.status_code}, Response: {r.json()}")
        return r # FIXME: What is the format of the payload? What are we interested in returning?

    def put_http(self, endpoint: str, **kwargs):
        r = requests.post(self.url + endpoint, json=kwargs)
        print(f"Status Code: {r.status_code}, Response: {r.json()}")
