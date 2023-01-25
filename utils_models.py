

def load_model(model_name, tokenizer_kwargs, model_kwargs):
    if "gpt-neo" in model_name:
        # reference: https://huggingface.co/docs/transformers/model_doc/gpt_neo
        from transformers import GPT2Tokenizer, GPTNeoForCausalLM
        tokenizer_class, model_class = GPT2Tokenizer, GPTNeoForCausalLM
    elif "gpt2" in model_name:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        tokenizer_class, model_class = GPT2Tokenizer, GPT2LMHeadModel
    else:
        raise NotImplemented

    # Load models
    tokenizer = tokenizer_class.from_pretrained(model_name, **tokenizer_kwargs)

    pad_token = tokenizer.eos_token_id if tokenizer.pad_token is None else tokenizer.pad_token
    model = model_class.from_pretrained(model_name, pad_token_id=pad_token, **model_kwargs)

    print("Importing classes for model", model_name)
    print(" ->", tokenizer_class)
    print(" ->", model_class)
    print("Vocabulary size:", tokenizer.vocab_size)
    return tokenizer, model


def get_device(device: str) -> str:
    import torch

    if device is None:
        return "gpu" if torch.cuda.is_available() else "cpu"
    else:
        return device