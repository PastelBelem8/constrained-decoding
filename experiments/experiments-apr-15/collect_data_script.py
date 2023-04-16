from constraint import Constraint
from elasticsearch import Elasticsearch

import argparse
import pandas as pd
import yaml


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-E", "--es-config", default="/home/cbelem/projects/constrained-decoding/es_config.yml", type=str)
    parser.add_argument("-N", "--num-samples", default=2_500, type=int)
    parser.add_argument("-D", "--distance-char", default=200, type=int)
    parser.add_argument("-T", "--target-word", required=True, type=str)
    parser.add_argument("-A", "--attr-words", default="default", type=str)
    parser.add_argument("-O", "--output-dir", default="/extra/ucinlp1/cbelem/experiments-apr-15/data", type=str)
    return parser.parse_args()

def read_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def scroll_documents(es, query, size=50, scroll_time="20m", index="re_pile"):
    data = es.search(index=index, query=query, size=size, scroll=scroll_time, sort=["_doc"]) #TODO: Check score
    hits, scroll_id = data["hits"]["hits"], data["_scroll_id"]
    yield hits

    total = len(hits)
    while len(hits) != 0:
        data = es.scroll(scroll_id=scroll_id, scroll=scroll_time)
        hits, scroll_id = data["hits"]["hits"], data["_scroll_id"]
        total += len(hits)
        yield hits

    es.clear_scroll(scroll_id=scroll_id)
    print(f"Done scrolling for query={query}!")
    yield None

def sample_sequences(es, n_sequences, attribute, target, distance, scroll_size=100) -> pd.DataFrame:
    phrases = [attribute, target]
    constraint = Constraint(*phrases, distance=distance)

    results = {
        "doc_id": [],
        "doc_subset": [],
        "full_prefix": [],
        "min_prefix": [],
        "continuation": [],
    }

    docs_iter = scroll_documents(es, constraint.es_query, size=scroll_size, index="re_pile")

    while (docs := next(docs_iter)) is not None and len(results["full_prefix"]) < n_sequences:

        for doc in docs:
            doc_id = doc["_id"]
            doc_subset = doc["_source"]["meta"]["pile_set_name"]
            text = doc["_source"]["text"]

            matches = constraint.find_matches(text)

            for match in matches:
                pref, cont = constraint.get_prefix(match)
                full_prefix, min_prefix = constraint.get_minimal_prefix(pref)

                if len(min_prefix) < 5:
                    print(f"Oops! min_prefix '{min_prefix}' has less than 5 chars")
                    continue

                results["full_prefix"].append(full_prefix)
                results["min_prefix"].append(min_prefix)
                results["continuation"].append(cont)

                results["doc_id"].append(doc_id)
                results["doc_subset"].append(doc_subset)

    results = pd.DataFrame(results)
    results.insert(0, "attribute", [attribute] * len(results))
    results.insert(0, "target", [target] * len(results))
    return results

def write_output(base_dir, filename, data: pd.DataFrame):
    import os
    print("Creating directory:", base_dir)
    os.makedirs(base_dir, exist_ok=True)

    filepath = f"{base_dir}/{filename}.csv"
    data.to_csv(filepath)
    print("Dumped file at", filepath)


if __name__ == "__main__":

    args = parse_arguments()

    # Connection to Elastic Search
    # We opt for using ES as opposed to searching the whole documents
    # because it immediately gives us a target set of documents that
    # satisfy the specified constraints
    es_config = read_config(args.es_config)
    es_engine = Elasticsearch(**es_config)

    target_word = args.target_word
    attribute_words = args.attr_words
    if attribute_words == "default":
        attribute_words = [
            "happy",
            "sad",
            "calm",
            "angry",
            "terror",
            "peace",
            "dead",
            "death",
            "great",
            "good",
            "bad",
            "terrible",
            "positive",
            "negative",
            "skill",
            "food",
        ]

    for attr in attribute_words:
        query = {'match': {'text': {'query': f'{target_word} {attr}', 'operator': 'and'}}}
        print(target_word, attr, es_engine.count(index="re_pile", query=query)["count"])

    analysis_data = []

    for attr in attribute_words:
        results = sample_sequences(es_engine, args.num_samples, attr, target_word, distance=args.distance_char)
        print(len(results))
        if len(results) != 0:
            analysis_data.append(results)

    analysis_data = pd.concat(analysis_data).reset_index(drop=True)

    data_out = args.output_dir
    write_output(data_out, args.target_word, analysis_data)