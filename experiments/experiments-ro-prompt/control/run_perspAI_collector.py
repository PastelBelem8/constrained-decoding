from googleapiclient.http import HttpError
from tqdm import tqdm

import pandas as pd
import jsonlines, time
import traceback as tb


### CONSTANTS
ATTRIBUTES = ["TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "PROFANITY", "THREAT"]
BASE_DIR = "/extra/ucinlp1/cbelem/experiment-ro-prompts/generations-results/uncond-half-multinomial/EleutherAI__pythia-1.4b"
ORIG_PREFIX_PATH = f"{BASE_DIR}/raw_data.csv"
OUTPUT_PATH = f"{BASE_DIR}/raw_data__perspective_AI.jsonl"


def setup_perspective_api_client(path: str) -> str:
    from googleapiclient import discovery

    with open(path, "r") as f:
        api_key = f.read()

    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=api_key,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )
    return client


client = setup_perspective_api_client("/home/cbelem/projects/constrained-decoding/experiment-ro-prompts/perspective_api.txt")
print("Created client!")


def get_toxicity(seq, client, attributes=ATTRIBUTES):
    def try_request(i=0) -> bool:
        analyze_request = {
            'comment': { 'text': seq },
            'requestedAttributes': {attr: {} for attr in attributes},
            "spanAnnotations": False,
            "doNotStore": True,
        }
        try:
             return client.comments().analyze(body=analyze_request).execute(), i
        except HttpError as e:
            print("------------------------------------------------------")
            tb.print_exc()
            print("------------------------------------------------------")
            reason = e._get_reason().lower()
            if "quota exceeded for quota metric" in reason:
                return None, i+5
            else:
                details = e.error_details
                result = {
                    "reason": reason,
                    "error_type": details[0]["errorType"]
                }
                return result, 0


    result, time_incr = None, 0
    while result is None:
        result, time_incr = try_request(time_incr)
        if result is None:
            print("Trying again in ", 60+time_incr, "seconds")
            time.sleep(60+time_incr)
            print("About to try...")

    return result


def collect_toxicity(data: pd.DataFrame, prefix_col: str="prefix", sequence_col: str ="sequence", output_path=OUTPUT_PATH):
    results = []
    for pref, seq in tqdm(zip(data[prefix_col].values, data[sequence_col].values)):
        time.sleep(1)
        r = get_toxicity(pref, client)
        r[prefix_col] = pref
        r[sequence_col] = seq
        results.append(r)

        if len(results) % 1000 == 0:
            print("Writing", len(results), "toxic results, at", output_path)
            with jsonlines.open(output_path, mode='w') as writer:
                writer.write_all(results)

    return results


if __name__ == "__main__":
    d = pd.read_csv(ORIG_PREFIX_PATH, index_col=0)
    print("Read file", ORIG_PREFIX_PATH, "with", len(d), "examples.")
    print("Finished collecting toxicity w/ perspective api")


    toxic = collect_toxicity(d, "prefix", "sequence", OUTPUT_PATH)

    import time
    start = time.time()
    print("Writing to:", OUTPUT_PATH)
    with jsonlines.open(OUTPUT_PATH, mode='w') as writer:
        writer.write_all(toxic)

    duration = time.time() - start
    print("Duration (min):", duration/60)
