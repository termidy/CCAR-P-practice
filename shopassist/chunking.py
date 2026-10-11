from retrieval import BM25, chunk_fixed, chunk_by_section

POLICY = open("policy.txt").read()

# question -> the exact fact the answer needs
QUESTIONS = [
    ("Can I return earrings?", "earrings"),
    ("How long does a card refund take?",
     "5 to 10 business days"),
    ("Who pays for shipping on a wrong item?",
     "caused by a store error"),
    ("How do I report a damaged parcel?", "within 7 days"),
    ("Can I exchange for another size?",
     "a return plus a new order"),
    ("Do I pay customs on an international return?",
     "Customs duties paid on the original delivery"),
    ("Who approves a refund of 250?",
     "Refunds above 100 require a human"),
    ("What happens if my return parcel is lost?",
     "14 days after the scan"),
    ("Can I return a gift without the buyer knowing?",
     "store credit in the name of the person"),
    ("My jacket broke after two months. What now?",
     "handled as warranty claims"),
    ("Can I return shoes without the shoe box?",
     "original shoe box"),
    ("How do I start a return?",
     "selecting the item and a reason"),
]


def recall(chunks, k=3):
    index = BM25(chunks)
    misses = []
    for question, fact in QUESTIONS:
        found = index.search(question, k)
        if not any(fact in chunk["text"] for chunk in found):
            misses.append(question)
    return 1 - len(misses) / len(QUESTIONS), misses


for name, chunks in [("fixed 400 chars", chunk_fixed(POLICY)),
                     ("by section", chunk_by_section(POLICY))]:
    score, misses = recall(chunks)
    print(f"{name:16} {len(chunks):3} chunks   "
          f"recall@3 = {score:.0%}")
    for question in misses:
        print("    missed:", question)
