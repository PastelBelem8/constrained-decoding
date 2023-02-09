import numpy as np


class SamplingOutput:
    def __init__(self, probs, samples, terms_ids, desc, logits=None):
        self.probs = probs
        self.samples = samples
        self.terms_ids = terms_ids
        self.description = desc
        self.logits = logits

    def __add__(self, output):
        if output == None:
            return self

        assert self.description == output.description, f"{self.description} vs {output.description}"
        assert np.array_equal(self.terms_ids, output.terms_ids), f"{self.terms_ids} vs {output.terms_ids}"
        # -------------------------------------------------------------------------------------------
        # May not be true. Samples unlike probs and logits are not tracked on a decoding basis.
        # In fact, they represent a single tensor of size batch_size x max_num_tokens.
        # In rare cases, it may happen that all sampled sequences finish earlier than the specified
        # number of tokens. Therefore making these comparisons somewhat irrelevant.
        # -------------------------------------------------------------------------------------------
        # assert len(self.samples) == len(output.samples), f"{len(self.samples)} vs {len(output.samples)}"
        # assert len(self.probs) == len(output.probs), f"{len(self.probs)} vs {len(output.probs)}"
        # assert len(self.logits) == len(output.logits), f"{len(self.logits)} vs {len(output.logits)}"
        assert len(self.probs) == self.samples.shape[1], f"Error: len(probs) vs shape(samples, 1): {len(self.probs)} vs {self.samples.shape[1]}"
        assert len(output.probs) == output.samples.shape[1], f"Error: len(probs) vs shape(samples, 1): {len(output.probs)} vs {output.samples.shape[1]}"

        assert len(self.logits) == 0 or len(self.probs) == len(self.logits), f"Error: len(probs) vs len(logits): {len(self.probs)} vs {len(self.logits)}"
        assert len(output.logits) == 0 or len(output.probs) == len(output.logits), f"Error: len(probs) vs len(logits): {len(output.probs)} vs {len(output.logits)}"

        new_probs = []
        new_logits = []

        n = min(len(self.probs), len(output.probs))
        max_seq_size = max(len(self.probs), len(output.probs))

        for i in range(n):
            new_probs.append(np.vstack((self.probs[i], output.probs[i])))
            # new_samples.append(torch.vstack((self.samples[i], output.samples[i])))

            if len(self.logits) > 0:
                new_logits.append(np.vstack((self.logits[i], output.logits[i])))

        if max_seq_size - n > 0:
            # TODO
            # Need to handle new_probs (set them to 1 -- indicating seqs are finished)
            # Need to handle logits (set them to 1 -- uniform distribution)
            # Need to handle samples:
            raise NotImplemented
        else:
            new_samples = np.vstack((self.samples, output.samples))

        return SamplingOutput(
            probs=new_probs,
            samples=new_samples,
            terms_ids=self.terms_ids,
            desc=self.description,
            logits=new_logits,
        )
