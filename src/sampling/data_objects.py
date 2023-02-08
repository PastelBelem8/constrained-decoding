import torch


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

        assert self.description == output.description
        assert self.terms_ids == output.terms_ids
        assert len(self.probs) == len(output.probs)
        assert len(self.samples) == len(output.samples)
        assert len(self.logits) == len(output.logits)

        new_probs = []
        new_samples = []
        new_logits = []

        for i in range(len(self.probs)):
            new_probs.append(torch.vstack((self.probs[i], output.probs[i])))
            new_samples.append(torch.vstack((self.samples[i], output.samples[i])))

            if len(self.logits) > 0:
                new_logits.append(torch.vstack((self.logits[i], output.logits[i])))

        return SamplingOutput(
            probs=new_probs,
            samples=new_samples,
            terms_ids=self.terms_ids,
            desc=self.description,
            logits=new_logits,
        )
