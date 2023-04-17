CUDA_DEVICE = 7
NUM_SAMPLES = 128

# TARGET_WORD = "muslim"
# TARGET_WORD = "buddhist"
# TARGET_WORD = "christian"
TARGET_WORD = "jewish"
TARGET_WORD = "doctor"
TARGET_WORD = "nurse"
TARGET_WORD = "father"

BASE_DIR = "/extra/ucinlp1/cbelem/experiments-apr-15"

model_name, model_revision = "EleutherAI/pythia-70m", ""
# model_name, model_revision = "EleutherAI/pythia-2.8b", ""


from sampling.utils import create_history
from tqdm import tqdm

import pandas as pd
import numpy as np

import os
import torch

DATA_DIR = f"{BASE_DIR}/data"
FILEPATH = f"{DATA_DIR}/{TARGET_WORD}.csv"
print("DATA FILEPATH", FILEPATH)

data = pd.read_csv(FILEPATH, index_col=0)
print(len(data))

# random shuffle of the data!
data = data.sample(frac=1, replace=False, random_state=42)

# select top 100 per different attribute (since we have 16 attributes)
data = data.groupby(["target", "attribute"]).head(100)


print("Cuda available:", torch.cuda.is_available())


def get_model_filename(*args) -> str:
    """Given a set of strings characterizing the model, create a filename."""
    args = [a.replace("/", "__") for a in args]
    args = [a for a in args if a]
    return "__".join(args)


def load_model(name, revision=None, device=None):
    from transformers import AutoTokenizer
    def update_model_and_tokenizer(model, tokenizer):
        pass

    model_kwargs = {}
    tokenizer_kwargs = {}

    # Load GPT2 model
    if "gpt2" in model_name:
        from transformers import GPT2LMHeadModel
        model_class = GPT2LMHeadModel

        def update_model_and_tokenizer(model, tokenizer):
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = model.config.eos_token_id

    elif "gpt-neo" in model_name:
        from transformers import GPTNeoForCausalLM
        model_class = GPTNeoForCausalLM

        def update_model_and_tokenizer(model, tokenizer):
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

    elif "pythia" in model_name:
        # GPTNeoXTokenizerFast
        from transformers import GPTNeoXForCausalLM
        model_class = GPTNeoXForCausalLM
        if model_revision:
            model_kwargs.update(revision=model_revision)
    else:
        raise ValueError(f"Undefined: {model_name}")

    model = model_class.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    update_model_and_tokenizer(model, tokenizer)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    return model, tokenizer

model_name2filename = get_model_filename(model_name, model_revision)
print("All model results will be created under the following name:", model_name2filename)

DEVICE = f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu"
MODEL, TOKENIZER = load_model(model_name, model_revision, DEVICE)
print(type(MODEL), type(TOKENIZER), DEVICE)

MODEL_DIR = f"{BASE_DIR}/models/{model_name2filename}"

os.makedirs(MODEL_DIR, exist_ok=True)



def generate(
    prefix: str,
    num_sequences: int,
    batch_size: int=64,
    model=MODEL,
    tokenizer=TOKENIZER,
    device=DEVICE,
    seed=None,
    **sampling_kwargs,
):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    default_kwargs = dict(
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    default_kwargs.update(sampling_kwargs)

    seqs = []
    seq_scores = []
    for start in range(0, num_sequences, batch_size):
        size = min(batch_size, num_sequences-start)

        input_ids = (
            tokenizer(
                # prefix, return_tensors="pt", add_special_tokens=False
                tokenizer.bos_token + prefix, return_tensors="pt", add_special_tokens=False
            ).input_ids
            if prefix is not None
            else None
        )
        # input_ids = create_history(size, prefix, tokenizer.bos_token_id).to(device)
        input_ids = create_history(size, input_ids, tokenizer.bos_token_id).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)

        # Generate sequences
        outputs = model.generate(input_ids, attention_mask=attention_mask, **default_kwargs)
        sequences = outputs.sequences

        # Compute each sequence probability
        results = model(sequences, attention_mask=torch.ones_like(sequences), labels=sequences)
        batch_score = -results.loss.cpu().detach().numpy()

        # Based on the discussion at
        # https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/20
        logits = torch.log_softmax(results.logits, dim=-1).detach()

        # collect the probability of the generated token
        # -- probability at index 0 corresponds to the token at index 1
        logits, input_ids = logits[:, :-1, :], sequences[:,1:,None]

        # Scores per token of the template
        batch_seq_scores = torch.gather(logits, 2, input_ids).squeeze(-1)

        _avg_loss = batch_seq_scores.mean(dim=-1).mean().item()
        assert np.abs(_avg_loss - batch_score) <= 1e-5, f"Loss does not match: (batch: {input_ids})), {_avg_loss} - {batch_score} > 1e-6"

        seqs.extend(sequences.detach().cpu().numpy().tolist())
        seq_scores.extend(batch_seq_scores.sum(dim=-1).detach().cpu().numpy().tolist())

    return seqs, seq_scores


def batch_generate(prefixes, sampling_kwargs, n: int, batch_size: int, model, tokenizer, seed: int, **kwargs):
    results = {"prefix": [], "sequence": [], "sequence_log_prob": [], "sampling_kwargs": []}

    np.random.seed(seed)
    torch.manual_seed(seed)

    for prefix in tqdm(prefixes, miniters=int(len(prefixes)/100), desc=str(sampling_kwargs)):
        sampled_seq, sampled_scores = generate(
            prefix,
            num_sequences=n,
            batch_size=batch_size,
            model=model,
            tokenizer=tokenizer,
            **sampling_kwargs,
            **kwargs,
        )

        results["prefix"].extend([prefix] * len(sampled_seq))
        results["sequence"].extend(tokenizer.batch_decode(sampled_seq, skip_special_tokens=True))
        results["sequence_log_prob"].extend(sampled_scores)
        results["sampling_kwargs"].extend([sampling_kwargs] * len(sampled_seq))

    return results

def multinomial_generation(prefixes, n, batch_size=16, seed=182, model=MODEL, tokenizer=TOKENIZER, **kwargs):
    sampling_kwargs = dict(do_sample=True, num_beams=1)

    results = batch_generate(
        prefixes=prefixes,
        sampling_kwargs=sampling_kwargs,
        n=n,
        batch_size=batch_size,
        seed=seed,
        model=model,
        tokenizer=tokenizer,
        **kwargs,
    )

    return pd.DataFrame(results)

def top_k_sampling(prefixes, ks, n, batch_size=16, seed=182, model=MODEL, tokenizer=TOKENIZER, **kwargs):
    if isinstance(ks, int):
        ks = [ks]

    results = []
    for k in ks:
        sampling_kwargs = dict(do_sample=True, top_k=k)
        result = batch_generate(
            prefixes=prefixes,
            sampling_kwargs=sampling_kwargs,
            n=n,
            batch_size=batch_size,
            seed=seed,
            model=model,
            tokenizer=tokenizer,
            **kwargs,
        )
        results.append(result)

    return pd.concat([pd.DataFrame(r) for r in results])


def top_p_sampling(prefixes, ps, n, batch_size=16, seed=182, model=MODEL, tokenizer=TOKENIZER, **kwargs):
    if isinstance(ps, int):
        ps = [ps]

    results = []
    for p in ps:
        sampling_kwargs = dict(do_sample=True, top_p=p)
        result = batch_generate(
            prefixes=prefixes,
            sampling_kwargs=sampling_kwargs,
            n=n,
            batch_size=batch_size,
            seed=seed,
            model=model,
            tokenizer=tokenizer,
            **kwargs,
        )
        results.append(result)

    results = pd.concat([pd.DataFrame(r) for r in results])
    return results


def temp_sampling(prefixes, ts, n, batch_size=16, seed=182, model=MODEL, tokenizer=TOKENIZER, **kwargs):
    if isinstance(ts, int):
        ts = [ts]

    results = []
    for t in ts:
        sampling_kwargs = dict(do_sample=True, temperature=t)
        result = batch_generate(
            prefixes=prefixes,
            sampling_kwargs=sampling_kwargs,
            n=n,
            batch_size=batch_size,
            seed=seed,
            model=model,
            tokenizer=tokenizer,
            **kwargs,
        )
        results.append(result)

    results = pd.concat([pd.DataFrame(r) for r in results])
    return results


print("="*60)
print("GENERATIONS")
print("="*60)

GENERATIONS = []
PREFIXES = data["min_prefix"].values# [:10]

print("--> Multinomial generation:", NUM_SAMPLES, "*", len(PREFIXES), "=", NUM_SAMPLES*len(PREFIXES))
model_sequences = multinomial_generation(PREFIXES, n=NUM_SAMPLES, batch_size=32, max_new_tokens=20)
model_sequences["sampling"] = "multinomial"
GENERATIONS.append(model_sequences)

KK = [2, 10, 40, 100]
print("--> top-k generation:", NUM_SAMPLES, "*", len(KK), "=", NUM_SAMPLES*len(PREFIXES)*len(KK))
model_sequences = top_k_sampling(PREFIXES, KK, n=NUM_SAMPLES, batch_size=32, max_new_tokens=20);
model_sequences["sampling"] = "top-k"
GENERATIONS.append(model_sequences)

print("--> top-p")
PP = [0.1, 0.5, 0.75, 0.9]
model_sequences = top_p_sampling(PREFIXES, PP, n=NUM_SAMPLES, batch_size=32, max_new_tokens=20);
model_sequences["sampling"] = "top-k"
GENERATIONS.append(model_sequences)

print("--> temperature")
TT = [0.1, 0.5, 0.85, 1.15]
model_sequences = temp_sampling(PREFIXES, TT, n=NUM_SAMPLES, batch_size=32, max_new_tokens=20);
model_sequences["sampling"] = "temperature"
GENERATIONS.append(model_sequences)

print("="*60)
print("PERSIST RESULT")
print("="*60)

MODEL_RESULTS_FILEPATH = f"{MODEL_DIR}/{TARGET_WORD}_min_prefix.csv"
print(MODEL_RESULTS_FILEPATH)
MODEL_GENERATIONS = pd.concat(GENERATIONS)
MODEL_GENERATIONS.to_csv(MODEL_RESULTS_FILEPATH)
print("Write generations at", MODEL_RESULTS_FILEPATH)