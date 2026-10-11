import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]
# dollars per million tokens: input, output (pricing page)
PRICES = {"claude-haiku-5-5": (0.10, 0.50),
          "claude-sonnet-5-5": (2.00, 10.00)}
POLICY = open("policy.txt").read()
SYSTEM = ("You are ShopAssist. Answer from the policy in one "
          "sentence.\n\n" + POLICY)

# question -> words a correct answer must contain
CASES = [
    ("Can I return earrings?", ["no"]),
    ("How long does a card refund take?", ["10 business days"]),
    ("Who approves a refund of 250?", ["human"]),
    ("Is the original shipping refunded for a wrong item?",
     ["yes"]),
    ("How many days do I have to report a damaged item?", ["7"]),
    ("Can I return opened cosmetics?", ["no"]),
    ("When is a lost return parcel treated as received?",
     ["14 days"]),
    ("Does a gift return refund the buyer's card?", ["no"]),
    ("Who reviews a warranty claim?", ["human"]),
    ("Can I return shoes without the shoe box?", ["no"]),
    ("I was in hospital and missed the 30 days. Can someone "
     "review my late return?", ["human"]),
    ("A sale item marked final sale: can I return it?", ["no"]),
]


def one_case(model, question, must):
    price_in, price_out = PRICES[model]
    start = time.time()
    response = client.messages.create(
        model=model, max_tokens=200,
        output_config={"effort": "low"}, system=SYSTEM,
        messages=[{"role": "user", "content": question}])
    seconds = time.time() - start
    answer = " ".join(b.text for b in response.content
                      if b.type == "text").lower()
    ok = all(re.search(rf"\b{re.escape(w)}\b", answer)
             for w in must)
    usage = response.usage
    cost = (usage.input_tokens * price_in
            + usage.output_tokens * price_out) / 1e6
    return ok, seconds, cost


def run(model):
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda c: one_case(model, *c),
                                CASES))
    correct = sum(ok for ok, _, _ in results)
    p50 = statistics.median(s for _, s, _ in results)
    return correct, p50, sum(c for _, _, c in results)


print(f"{'model':18} {'correct':>8} {'p50 s':>6} "
      f"{'$ per 1k requests':>18}")
for model in MODELS:
    correct, p50, cost = run(model)
    per_1k = cost / len(CASES) * 1000
    print(f"{model:18} {correct:>5}/{len(CASES)} {p50:6.1f} "
          f"{per_1k:18.2f}")
