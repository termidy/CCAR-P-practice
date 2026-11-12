import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]
EFFORTS = ["low", "medium"]
# dollars per million tokens: input, output (pricing page)
PRICES = {"claude-haiku-5-5": (0.10, 0.50),
          "claude-sonnet-5-5": (2.00, 10.00)}

ROUTE = """You are ShopAssist's router. Reply with one word,
the category of the customer's message:
status - where an order is, also a parcel marked delivered
  that did not arrive
return - wants to send an item back
refund - asks about money for a return already sent back
damaged - an item arrived broken, torn or wrong, whatever
  the customer asks for
other - anything else"""
POLICY = open("policy.txt").read()
ANSWER = ("You are ShopAssist. Answer from the policy in one "
          "short sentence. Start with Yes or No when you can."
          "\n\n" + POLICY)
SYSTEM = {"route": ROUTE, "answer": ANSWER}

# kind, customer message, words a correct reply must contain
CASES = [
    ("route", "Where is my order? It was due on Monday.",
     ["status"]),
    ("route", "The jacket is too small, I want to send it back.",
     ["return"]),
    ("route", "The coat came torn. Just give me my money back.",
     ["damaged"]),
    ("route", "I sent the shoes back two weeks ago and still "
     "have no money.", ["refund"]),
    ("route", "The app says delivered, but there is no parcel "
     "at my door.", ["status"]),
    ("route", "Wrong size arrived, please refund me.",
     ["damaged"]),
    ("answer", "Can I return a swimsuit with the hygiene strip "
     "still on?", ["yes"]),
    ("answer", "An item arrived damaged 10 days ago. Can I "
     "still report it as damaged?", ["no"]),
    ("answer", "Can I return earrings?", ["no"]),
    ("answer", "Who approves a refund of 600?", ["human"]),
    ("answer", "I return a jacket because it is too small. Do "
     "I get the shipping cost back?", ["no"]),
    ("answer", "My return was scanned 16 days ago but never "
     "arrived. Do I get my money?", ["yes"]),
]


def is_correct(reply, must):
    for sign in ".,;:!?*":
        reply = reply.replace(sign, " ")
    words = reply.lower().split()
    for word in must:
        if word not in words:
            return False
    return True


def one_case(model, effort, case):
    kind, message, must = case
    start = time.time()
    response = client.messages.create(
        model=model, max_tokens=2000,
        output_config={"effort": effort},
        system=SYSTEM[kind],
        messages=[{"role": "user", "content": message}])
    seconds = time.time() - start
    reply = response.content[-1].text
    price_in, price_out = PRICES[model]
    usage = response.usage
    cost = (usage.input_tokens * price_in
            + usage.output_tokens * price_out) / 1e6
    return is_correct(reply, must), seconds, cost


def score(model, effort):
    jobs = []
    with ThreadPoolExecutor(12) as pool:
        for case in CASES * 3:
            job = pool.submit(one_case, model, effort, case)
            jobs.append((case, job))
    times = []
    cost = 0
    missed = []
    for case, job in jobs:
        ok, seconds, case_cost = job.result()
        times.append(seconds)
        cost += case_cost
        if not ok:
            missed.append(case[1])
    correct = len(jobs) - len(missed)
    p50 = statistics.median(times)
    p95 = statistics.quantiles(times, n=20)[-1]
    per_1k = cost / len(jobs) * 1000
    return correct, p50, p95, per_1k, missed


for model in MODELS:
    for effort in EFFORTS:
        correct, p50, p95, per_1k, missed = score(model, effort)
        print(model, effort, "|", correct, "of 36 correct | p50",
              round(p50, 1), "s | p95", round(p95, 1), "s |",
              round(per_1k, 2), "dollars per 1k")
        for message in sorted(set(missed)):
            print("   missed", missed.count(message), "of 3:",
                  message)
