import spacy

__nlp__ = spacy.load("en_core_web_sm", disable=["ner"])


def get_phrases(text, phrases):
    try:
        return [text.index(p.lower()) + len(p) for p in phrases]
    except:
        return None


class Constraint:
    def __init__(self, *words, distance: int=30):
        self.words = list(words)
        self.wordsl = [p.lower() for p in self.words]

        self.distance = distance
        assert distance > 0

    @property
    def es_query(self):
        return {'match': {'text': {'query': " ".join(self.words), 'operator': 'and'}}}

    def find_matches(self, text: str) -> list:
        textl = text.lower()

        # Indices
        indices = get_phrases(textl, self.wordsl)
        if indices is None:
            return []

        windows = []
        for i in indices:
            wstart = max(0, i-self.distance)
            wend = min(len(text), i+self.distance+1)
            text_i = textl[wstart:wend]
            window_i = get_phrases(text_i, self.wordsl)

            if window_i is not None:
                windows.append(text[wstart:wend])

        return windows

    def get_prefix(self, window: str):
        # Index returns the first occurrence of specified word
        # We sum the length of the word w to obtain end character
        prefixes = get_phrases(window.lower(), self.wordsl)
        # The largest prefix will definitely contain both words
        prefixes = sorted(prefixes)
        # We'll pick the longest prefix
        prefix, continuation = window[:prefixes[-1]],  window[prefixes[-1]:]

        return prefix, continuation

    def get_minimal_prefix(self, prefix: str):
        sentences = __nlp__(prefix).sents
        sentences_ids = [prefix.index(s.text) for s in sentences]

        full_prefix = prefix
        # Because of the way we create the prefixes we will
        # prioritize right most prefix matching
        sentences_ids = sentences_ids[::-1]

        for index in sentences_ids:
            minimal_prefix = prefix[index:]

            # Check match of phrases
            ids = get_phrases(minimal_prefix.lower(), self.wordsl)
            if ids is not None:
                return full_prefix, minimal_prefix

        return full_prefix, full_prefix
