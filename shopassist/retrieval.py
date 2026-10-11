import math
import re
from collections import Counter

WORD = re.compile(r"[a-z0-9]+")


def words(text):
    return WORD.findall(text.lower())


class BM25:
    """Keyword ranking: rare query words in a short chunk win."""

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.docs = [Counter(words(c["text"])) for c in chunks]
        sizes = [sum(d.values()) for d in self.docs]
        self.avg = sum(sizes) / len(sizes)
        n = len(self.docs)
        df = Counter(w for d in self.docs for w in d)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5))
                    for w, f in df.items()}
        self.k1, self.b = k1, b

    def score(self, query, doc):
        size = sum(doc.values())
        norm = self.k1 * (1 - self.b + self.b * size / self.avg)
        return sum(self.idf.get(w, 0) * doc[w] * (self.k1 + 1)
                   / (doc[w] + norm) for w in set(words(query)))

    def search(self, query, k=3):
        ranked = sorted(zip(self.chunks, self.docs),
                        key=lambda cd: -self.score(query, cd[1]))
        return [chunk for chunk, _ in ranked[:k]]


def chunk_fixed(text, size=400):
    return [{"id": f"chars {i}-{i + size}",
             "text": text[i:i + size]}
            for i in range(0, len(text), size)]


def chunk_by_section(text):
    chunks = []
    for block in text.split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) >= 2:
            chunks.append({"id": lines[0].title(),
                           "text": block.strip()})
    return chunks
