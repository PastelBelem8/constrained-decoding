from tqdm import tqdm
from typing import List

import numpy as np
import pandas as pd
import os, time, sys
import torch

import sys, pprint
sys.path.append(f"{__file__.rpartition('/')[0]}/../control")
from run_decoder import load_model


pp = pprint.PrettyPrinter(indent=4)

def parse_arguments() -> dict:
    import argparse, yaml
    #
    # Example use: python -m run_half_experiment_ro_prompts --config ./configs/generate-ro.yml --model_name EleutherAI/pythia-70m
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--model_revision", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--dataset_path", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--exclusive_decoding", default=None, type=str)

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

    if args.output_dir is not None:
        config["sampling"].update(output_dir=args.output_dir)

    if args.dataset_path is not None:
        config["sampling"].update(dataset_path=args.dataset_path)

    if args.exclusive_decoding is not None:
        config["decoding"] = {k: v for k, v in config["decoding"].items()
         if not isinstance(v, dict) or k == args.exclusive_decoding}

    return config


def init_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

def init_pad_token(tokenizer):
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("SETTING pad token to eos token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

def generate(
        prefixes: List[str],
        batch_size: int,
        model,
        tokenizer,
        seed: str=None,
        **sampling_kwargs,
    ):
    init_seed(seed)
    init_pad_token(tokenizer)
    device = model.device

    # Default generation kwargs
    default_kwargs = dict(
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    default_kwargs.update(sampling_kwargs)

    seqs = []
    seq_scores = []
    seq_trans_scores = []
    seq_entr_scores = []
    for start in tqdm(range(0, len(prefixes), batch_size)):
        end = min(start+batch_size, len(prefixes))

        batch = prefixes[start:end]
        batch = [tokenizer.bos_token + p for p in batch]
        #^Comment line above to remove BOS_token

        batch_enc = tokenizer.batch_encode_plus(batch, return_tensors="pt", add_special_tokens=False, padding=True)
        input_ids = batch_enc.input_ids.to(device)
        attention_mask = batch_enc.attention_mask.to(device)

        # Generate sequences
        outputs = model.generate(input_ids, attention_mask=attention_mask, **default_kwargs)
        sequences = outputs.sequences # check if compute_transition_scores is good enough

        # Make sure the pad token is not accounted for in the loss
        targets = sequences.clone()
        targets[sequences == tokenizer.pad_token_id] = -100

        # Compute each continuation's probability
        mask = sequences == tokenizer.pad_token_id
        attention_mask = torch.ones_like(sequences)
        attention_mask[mask] = 0
        results = model(sequences, attention_mask=attention_mask, labels=sequences)
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


def load_decoding_configs(configs: dict):
    """
    Returns
    -------
    (dict[str, any], List[(str, dict[str, float])])
        The default configurations that should be used by every
        decoding algorithm and list of tuples. Each tuple consists
        of the unique name of the decoding algorithm together
        and the sampling_configs.
    """
    default_configs = {k: v for k,v in configs.items()}
    decoding_configs = []

    decoding_configs.append((f"multinomial", dict(do_sample=True, num_beams=1)))

    # Temp
    temp_configs = default_configs.pop("temperature", None)
    if temp_configs is not None:
        for t in temp_configs["params"]:
            decoding_configs.append((f"temperature_{t}", dict(do_sample=True, temperature=t)))

    # Top-p
    top_p_configs = default_configs.pop("top_p", None)
    if top_p_configs is not None:
        for p in top_p_configs["params"]:
            decoding_configs.append((f"top_p_{p}", dict(do_sample=True, top_p=p)))

    # Top-k
    top_k_configs = default_configs.pop("top_k", None)
    if top_k_configs is not None:
        for k in top_k_configs["params"]:
            decoding_configs.append((f"top_k_{k}", dict(do_sample=True, top_k=k)))

    return default_configs, decoding_configs



if __name__ == "__main__":
    config = parse_arguments()
    print(f"\n ----- Starting Experiment ----")
    pp.pprint(config)

    # -------------------------------------------------------
    # Handle sampling configurations
    # -------------------------------------------------------
    sampling_configs = config.pop("sampling")

    # Read data
    dataset_path = sampling_configs.pop("dataset_path")
    print("Reading dataset from:\n-->", dataset_path)
    data = pd.read_csv(dataset_path, index_col=0)

    output_dir = sampling_configs.pop("output_dir")
    # -------------------------------------------------------
    # Load model
    # -------------------------------------------------------
    model_configs = config.pop("model")
    model_name, model, tokenizer, device = load_model(**model_configs)
    print("--> Cuda available:", torch.cuda.is_available())
    output_dir = f"{output_dir}/{model_name}"
    print("--> Creating dir:", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # -------------------------------------------------------
    # Load decoding algorithms
    # -------------------------------------------------------
    decoding_configs = config.pop("decoding")
    default_configs, decoding_configs = load_decoding_configs(decoding_configs)

    # from transformers.utils import logging
    # logging.set_verbosity_error() # not the best way

    for decoder_name, decoder_config in decoding_configs:
        print("\n\n====================")
        print("DECODER:", decoder_name)
        print("===================")
        decoder_config.update(**default_configs)
        decoder_config.update(**sampling_configs)
        start = time.time()
        results = generate(prefixes=data["prefix_text"].values, model=model, tokenizer=tokenizer, **decoder_config)
        end = time.time()
        print("\n\nDecoding duration:", (end - start) / 3600, "h")

        sampled_seq, sampled_scores, sampled_seq_trans_scores, sampled_seq_entr_scores = results
        decoded_data = data.copy()
        decoded_data["sampling_kwargs"] = [decoder_config] * len(decoded_data)
        decoded_data["sampled_sequence"] = tokenizer.batch_decode(sampled_seq, skip_special_tokens=True)

        prefixes = data["prefix_text"].values
        samples = decoded_data["sampled_sequence"].values
        continuations = [seq.replace(pref, "") for seq, pref in zip(samples, prefixes)]

        decoded_data["sampled_continuation"] = continuations
        decoded_data["sampled_sequence_log_prob"] = sampled_scores
        decoded_data["sampled_seq_trans_log_probs"] = sampled_seq_trans_scores
        decoded_data["sampled_seq_entropy_log_probs"] = sampled_seq_entr_scores

        output_fp = f"{output_dir}/{decoder_name}.csv"
        print("---> Created file", output_fp)
        decoded_data.to_csv(output_fp)

    print("Done!")
