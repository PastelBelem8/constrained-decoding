from transformers import AutoTokenizer
from typing import Any, List, Tuple


import torch


def get_model_filename(*args) -> str:
    """Given a set of strings characterizing the model, create a filename."""
    args = [a for a in args if a]
    args = [a.replace("/", "__") for a in args]
    args = [a for a in args if a]
    return "__".join(args)


def load_model(name, revision=None, device=None) -> Tuple[str, object, object, str]:
    def update_model_and_tokenizer(model, tokenizer):
        pass

    model_kwargs = {}
    tokenizer_kwargs = {}
    # -------------------------------------------------------------------------
    # Load GPT2 model
    # -------------------------------------------------------------------------
    if "gpt2" in name:
        from transformers import GPT2LMHeadModel
        model_class = GPT2LMHeadModel

        def update_model_and_tokenizer(model, tokenizer):
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = model.config.eos_token_id

    # -------------------------------------------------------------------------
    # GPT-NEO
    # -------------------------------------------------------------------------
    elif "gpt-neo" in name:
        from transformers import GPTNeoForCausalLM
        model_class = GPTNeoForCausalLM

        def update_model_and_tokenizer(model, tokenizer):
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

    # -------------------------------------------------------------------------
    # Pythia models
    # -------------------------------------------------------------------------
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


def batch_fn(iterable: List[str], batch_size: int, fn: callable) -> List[Any]:
    results = []

    for bstart in range(0, len(iterable), batch_size):
        bend = min(bstart+batch_size, len(iterable))
        batch = iterable[bstart:bend].tolist()

        out = fn(batch)
        results.extend(out)

    return results
