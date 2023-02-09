def load_model(model_name, tokenizer_kwargs=None, model_kwargs=None):
    tokenizer_kwargs = tokenizer_kwargs or {}
    model_kwargs = model_kwargs or {}
    if "gpt-neo" in model_name:
        # reference: https://huggingface.co/docs/transformers/model_doc/gpt_neo
        from transformers import GPT2Tokenizer, GPTNeoForCausalLM

        tokenizer_class, model_class = GPT2Tokenizer, GPTNeoForCausalLM
        default_tokenizer_kwargs = {"model_max_length": 256}
    elif "gpt2" in model_name:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel

        tokenizer_class, model_class = GPT2Tokenizer, GPT2LMHeadModel
        default_tokenizer_kwargs = {"model_max_length": 512}
    else:
        raise NotImplemented

    default_tokenizer_kwargs.update(tokenizer_kwargs)

    # Load models
    tokenizer = tokenizer_class.from_pretrained(model_name, **default_tokenizer_kwargs)

    pad_token = (
        tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    )
    tokenizer.pad_token_id = pad_token

    model = model_class.from_pretrained(
        model_name, pad_token_id=pad_token, **model_kwargs
    )

    print("Importing classes for model", model_name)
    print(" ->", tokenizer_class)
    print(" ->", model_class)
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Pad token id:", tokenizer.pad_token_id)
    return tokenizer, model


def get_device(device: str) -> str:
    import torch

    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:
        return device
