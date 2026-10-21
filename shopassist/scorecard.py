import statistics
import time
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
    ("How long does a card refund take?", ["10"]),
    ("Who approves a refund of 250?", ["human"]),
    ("How many days do I have to report a damaged item?", ["7"]),
    ("When is a lost return parcel treated as received?", ["14"]),
    ("Who reviews a warranty claim?", ["human"]),
]


def is_correct(answer, must):
    for sign in ".,;:!?*":
        answer = answer.replace(sign, " ")
    words = answer.lower().split()
    for word in must:
        if word not in words:
            return False
    return True


def one_case(model, question, must):
    price_in, price_out = PRICES[model]
    start = time.time()
    response = client.messages.create(
        model=model, max_tokens=200,
        output_config={"effort": "low"}, system=SYSTEM,
        messages=[{"role": "user", "content": question}])
    seconds = time.time() - start
    answer = ""
    for block in response.content:
        if block.type == "text":
            answer += block.text
    usage = response.usage
    cost = (usage.input_tokens * price_in
            + usage.output_tokens * price_out) / 1e6
    return is_correct(answer, must), seconds, cost


def run(model):
    correct = 0
    times = []
    cost = 0
    for question, must in CASES:
        ok, seconds, case_cost = one_case(model, question, must)
        if ok:
            correct += 1
        times.append(seconds)
        cost += case_cost
    return correct, statistics.median(times), cost


for model in MODELS:
    correct, p50, cost = run(model)
    per_1k = cost / len(CASES) * 1000
    print(model, correct, "of", len(CASES), "correct, median",
          round(p50, 1), "s,", round(per_1k, 2), "dollars per 1k")
