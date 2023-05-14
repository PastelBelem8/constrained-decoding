from sampling.utils import create_history
from typing import List

import pandas as pd
import numpy as np
import torch
import tqdm


def generate(
    prefix: str, num_sequences: int, batch_size: int, model, tokenizer, device, seed=None, **sampling_kwargs,
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
    seq_trans_scores = []
    seq_entr_scores = []
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
        seq_trans_scores.extend(batch_seq_scores.cpu().detach().numpy())

        # Compute entropy
        probits = torch.softmax(logits, dim=-1)
        torch_entropy = -1 * torch.mul(probits, logits).sum(dim=-1)
        seq_entr_scores.extend(torch_entropy.detach().cpu().detach().numpy())

    return seqs, seq_scores, seq_trans_scores, seq_entr_scores


def batch_generate(dump_freq: int, num_sequences: int, sampling_kwargs, batch_size: int, model, tokenizer, seed: int, output_path="./temp", **kwargs) -> dict:
    results = {"sampling_kwargs": [], "sequence_log_prob": [], "sequence": [], "seq_trans_log_probs": [], "seq_entropy_log_probs": []}

    np.random.seed(seed)
    torch.manual_seed(seed)

    pd.DataFrame(results).to_csv(f"{output_path}.csv") # overwrite existing filepath

    for num_seqs in tqdm.tqdm(range(0, num_sequences, dump_freq), desc=str(sampling_kwargs)):
        num_seqs = min(dump_freq, num_sequences - num_seqs)

        sampled_seq, sampled_scores, sampled_seq_trans_scores, sampled_seq_entr_scores = generate(
            prefix=None, num_sequences=num_seqs, batch_size=batch_size,
            model=model, tokenizer=tokenizer,
            **sampling_kwargs,
            **kwargs,
        )
        results["sampling_kwargs"].extend([sampling_kwargs] * len(sampled_seq))
        results["sequence_log_prob"].extend(sampled_scores)
        results["sequence"].extend(tokenizer.batch_decode(sampled_seq, skip_special_tokens=True))
        results["seq_trans_log_probs"].extend(sampled_seq_trans_scores)
        results["seq_entropy_log_probs"].extend(sampled_seq_entr_scores)

        print("Writing down", len(sampled_seq), "to file:", output_path)
        pd.DataFrame(results).to_csv(f"{output_path}.csv", mode="a", header=False) # append existing filepath
        results = {"sampling_kwargs": [], "sequence_log_prob": [], "sequence": [], "seq_trans_log_probs": [], "seq_entropy_log_probs": []}

    return results


def multinomial_generation(**kwargs):
    sampling_kwargs = dict(do_sample=True, num_beams=1)

    results = batch_generate(sampling_kwargs=sampling_kwargs, **kwargs)

    return pd.DataFrame(results)

def top_k_sampling(params: List[int], output_path: str, **kwargs):
    if isinstance(params, int):
        params = [params]

    results = []
    for k in params:
        output_path_k = f"{output_path}_{k}"
        sampling_kwargs = dict(do_sample=True, top_k=k)
        result = batch_generate(sampling_kwargs=sampling_kwargs, output_path=output_path_k, **kwargs)  # todo
        results.append(result)

    return pd.concat([pd.DataFrame(r) for r in results])


def top_p_sampling(params: List[float], output_path: str, **kwargs):
    if isinstance(params, int):
        params = [params]

    results = []
    for p in params:
        output_path_p = f"{output_path}_{p}"
        sampling_kwargs = dict(do_sample=True, top_p=p)
        result = batch_generate(sampling_kwargs=sampling_kwargs,  output_path=output_path_p, **kwargs)
        results.append(result)

    results = pd.concat([pd.DataFrame(r) for r in results])
    return results


def temp_sampling(params: List[float], output_path: str, **kwargs):
    if isinstance(params, (float, int)):
        params = [params]

    results = []
    for t in params:
        output_path_t = f"{output_path}_{t}"
        sampling_kwargs = dict(do_sample=True, temperature=t)
        result = batch_generate(sampling_kwargs=sampling_kwargs, output_path=output_path_t, **kwargs)
        results.append(result)

    results = pd.concat([pd.DataFrame(r) for r in results])
    return results
