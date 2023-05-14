from torch.distributions import Binomial

import torch
import numpy as np


def tensor2numpy(t):
    if isinstance(t, torch.Tensor):
        return t.cpu().numpy()
    else:
        return t


def quantile_intervals(
    n,  # (int) Number of data points in your original sample
    desired_quantiles,  # (1D torch.tensor, dtype=float) Contains values between 0 and 1 indicating the quantiles you want intervals for
    desired_confidence=0.95,  # (float) Single confidence level you want the intervals to span
):
    desired_quantiles = torch.tensor(desired_quantiles, dtype=float)

    desired_ranks = desired_quantiles.double()*n
    possible_ranks = torch.arange(0, n, 1).double()
    dists = Binomial(total_count=n*torch.ones_like(desired_ranks), probs=desired_quantiles.double())

    pmfs = dists.log_prob(possible_ranks.unsqueeze(1)).exp().T
    cdfs = pmfs.cumsum(dim=1)

    dists = cdfs.unsqueeze(-1) - cdfs.unsqueeze(-2)  # (batch, upper_interval_idx, lower_interval_idx)
    valid_upper_indices = possible_ranks.unsqueeze(0) > desired_ranks.unsqueeze(-1)
    valid_lower_indices = possible_ranks.unsqueeze(0) < desired_ranks.unsqueeze(-1)
    valid_indices = valid_upper_indices.unsqueeze(-1) & valid_lower_indices.unsqueeze(-2)

    valid_confs = dists >= desired_confidence
    valid_dists = torch.where(valid_indices & valid_confs, dists, torch.finfo(torch.float64).max)

    interval_info = valid_dists.view(valid_dists.shape[0], -1).min(dim=-1)
    interval_indices, interval_widths = interval_info.indices, interval_info.values
    interval_indices = torch.stack([interval_indices // n, interval_indices % n], -1)

    upper_interval_ranks = interval_indices[:, 0]
    upper_interval_quantiles = upper_interval_ranks / n
    lower_interval_ranks = interval_indices[:, 1]
    lower_interval_quantiles = lower_interval_ranks / n

    metadata = {
        "n": n,
        "desired_confidence": desired_confidence,
        "desired_quantiles": desired_quantiles,
        "desired_ranks": desired_ranks,
        # For whatever sample of results you have that you are taking quantiles on, the interval will be defined be other specific quantiles of the same sample
        "upper_interval_quantiles": upper_interval_quantiles,
        "lower_interval_quantiles": lower_interval_quantiles,
        # These are the exact ranks of the interval endpoints. i.e., if the upper_interval_rank=4, then the upper end of the interval is the 4th largest value in the sample
        "upper_interval_ranks": upper_interval_ranks,
        "lower_interval_ranks": lower_interval_ranks,
        # The estimated intervals are not exact, and will often span more than the desired confidence. A good sanity check is to make sure the interval widths are close enough.
        "interval_widths": interval_widths,
    }

    return {k: tensor2numpy(v) for k, v in metadata.items()}