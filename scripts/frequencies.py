from collections import Counter, defaultdict

def default_0():
    return 0

def default_init():
    return defaultdict(default_0)


class PositionalFrequencies:
    """Computes the frequencies of tokens in each position"""
    def __init__(self):
        self.total_tokens = 0
        self.counts = defaultdict(default_init)

    def add(self, token, pos):
        self.counts[token][pos] += 1
        self.total_tokens += 1

    def unigram_count(self) -> Counter:
        unigram_count = {}

        for token, pos_counts in self.counts.items():
            unigram_count[token] = sum(pos_counts.values())

        return Counter(unigram_count)

    def most_common(self, n: int=None, pos=None):
        if pos is None:
            return self.unigram_count().most_common(n)

        # If a position is specified, order by position counts
        counts = {}

        for token, pos_counts in self.counts.items():
            if pos_counts.get(pos):
                counts[token] = pos_counts[pos]

        return Counter(counts).most_common(n)
