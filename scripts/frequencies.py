from collections import Counter, defaultdict

def default_0():
    return 0

def default_init():
    return defaultdict(default_0)


class PositionalFrequencies:
    """Computes the frequencies of tokens in each position.

    Attributes
    ----------
    total_tokens: int
        Number of total ngrams.

    counts: dict<tuple, dict<int, int>>
        Positional counts associated with the ngrams (tuple).
    """
    def __init__(self):
        self.total_tokens = 0
        self.counts = defaultdict(default_init)

    def __add__(self, counts):
        def update_counts(new_counts, vals):
            for ngram, pos_counts in vals.counts.items():
                for pos, count in pos_counts.items():
                    new_counts.add(ngram, pos, count)

        total_counts = PositionalFrequencies()
        update_counts(total_counts, self)
        update_counts(total_counts, counts)

        return total_counts

    def add(self, token, pos, incr=1):
        self.counts[token][pos] += incr
        self.total_tokens += incr

    def count(self) -> Counter:
        count = {}

        for token, pos_counts in self.counts.items():
            count[token] = sum(pos_counts.values())

        return Counter(count)

    def most_common(self, n: int=None, pos=None):
        if pos is None:
            return self.count().most_common(n)

        # If a position is specified, order by position counts
        counts = {}

        for token, pos_counts in self.counts.items():
            if pos_counts.get(pos):
                counts[token] = pos_counts[pos]

        return Counter(counts).most_common(n)
