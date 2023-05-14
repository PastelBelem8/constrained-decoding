from generate import top_p_sampling, top_k_sampling, temp_sampling, multinomial_generation
from functools import partial
from typing import Tuple

import os
import torch
import time


def parse_arguments() -> dict:
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str)
    parser.add_argument("-m", "--model-name", required=True, type=str)
    parser.add_argument("-r", "--model-revision", default=None, type=str)
    parser.add_argument("-d", "--device", default="cuda", type=str)
    parser.add_argument("-bs", "--batch_size", default=None, type=int)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["model"] = {
        "name": args.model_name,
        "revision": args.model_revision,
        "device": args.device,
    }

    if args.batch_size is not None:
        config["sampling"].update(batch_size=args.batch_size)

    return config


def get_model_filename(*args) -> str:
    """Given a set of strings characterizing the model, create a filename."""
    args = [a for a in args if a]
    args = [a.replace("/", "__") for a in args]
    args = [a for a in args if a]
    return "__".join(args)


def load_model(name, revision=None, device=None) -> Tuple[str, object, object, str]:
    from transformers import AutoTokenizer
    def update_model_and_tokenizer(model, tokenizer): pass

    model_kwargs = {}
    tokenizer_kwargs = {}
    # Load GPT2 model
    if "gpt2" in name:
        from transformers import GPT2LMHeadModel
        model_class = GPT2LMHeadModel

        def update_model_and_tokenizer(model, tokenizer):
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = model.config.eos_token_id

    elif "gpt-neo" in name:
        from transformers import GPTNeoForCausalLM
        model_class = GPTNeoForCausalLM

        def update_model_and_tokenizer(model, tokenizer):
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

    elif "pythia" in name:
        # GPTNeoXTokenizerFast
        from transformers import GPTNeoXForCausalLM
        model_class = GPTNeoXForCausalLM
        if revision:
            model_kwargs.update(revision=revision)
        tokenizer_kwargs.update(padding_side="left")
    else:
        raise ValueError(f"Undefined: {name}")

    model = model_class.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name, **tokenizer_kwargs)
    update_model_and_tokenizer(model, tokenizer)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_filename = get_model_filename(name, revision)

    model.to(device)
    return model_filename, model, tokenizer, device


def load_decoding_algorithm(method, params=None, **kwargs) -> Tuple[str, str]:
    default_params = {}
    default_params.update(**kwargs)

    if method.lower() in ("top-p", "top_p", "topp"):
        decoding_name = "top_p"
        default_params.update(params=params)
        decoding_method = top_p_sampling

    elif method.lower() in ("top-k", "top_k", "topk"):
        decoding_name = "top_k"
        default_params.update(params=params)
        decoding_method = top_k_sampling

    elif method.lower() in ("temp", "temperature"):
        decoding_name = "temperature"
        default_params.update(params=params)
        decoding_method = temp_sampling

    elif method.lower() in ("multinomial", "random"):
        decoding_name = "multinomial"
        decoding_method = multinomial_generation

    else:
        raise ValueError(f"Unknown decoding algorithm: {method} with params {params}")

    decoding_method = partial(
        decoding_method,
        **default_params,
    )

    return decoding_name, decoding_method


if __name__ == "__main__":
    config = parse_arguments()
    print(f"Starting Experiment\n[Experiment] Configs: {config}")

    sampling_configs = config.pop("sampling")
    output_dir = sampling_configs.pop("output_dir")
    os.makedirs(output_dir, exist_ok=True)

    model_configs = config.pop("model")
    model_name, model, tokenizer, device = load_model(**model_configs)
    print("Cuda available:", torch.cuda.is_available())

    decoding_configs = config.pop("decoding")
    decoding_name, decoding_method = load_decoding_algorithm(**decoding_configs, **sampling_configs)

    output_fp = f"{output_dir}/{model_name}-{decoding_name}"
    if os.path.exists(output_fp):
        print(output_fp, "already exists!")
        os.remove(output_fp)

    from transformers.utils import logging
    logging.set_verbosity_error() # not the best way

    start = time.time()
    decoding_method(model=model, tokenizer=tokenizer, device=device, output_path=output_fp)
    end = time.time()

    print("Decoding duration:", (end - start) / 3600, "h")
    print("Created file", output_fp)