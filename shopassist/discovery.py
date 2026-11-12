from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-5-5"
# dollars per million tokens, in and out (pricing page)
PRICE_IN, PRICE_OUT = 2.00, 10.00
SYSTEM = "You are ShopAssist. Answer in one sentence."

# customer question, a word the right answer contains
QUESTIONS = [
    ("Can I return earrings?", "hygiene"),
    ("Who approves a refund of 600?", "human"),
    ("Do I pay customs on an international return?",
     "authority"),
    ("How long does a refund to my card take?",
     "business days"),
    ("My return parcel was scanned but never arrived. "
     "Do I get my money?", "14"),
    ("How long is a gift receipt valid?", "60"),
    ("My jacket broke after two months. What now?",
     "warranty"),
    ("Can I exchange shoes for a bigger size?", "new order"),
]


def answer_text(response):
    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text
    return text


def whole(question, document):
    response = client.messages.create(
        model=MODEL, max_tokens=300,
        output_config={"effort": "low"},
        system=SYSTEM + "\n\n" + document,
        messages=[{"role": "user", "content": question}])
    usage = response.usage
    return (answer_text(response), usage.input_tokens,
            usage.output_tokens)


def sections_of(document):
    sections = {}
    for block in document.split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) >= 2:
            sections[lines[0].title()] = block.strip()
    return sections


def progressive(question, document):
    sections = sections_of(document)
    read_section = {
        "name": "read_section",
        "description": "Read one section of the handbook.",
        "input_schema": {
            "type": "object", "required": ["title"],
            "properties": {"title": {
                "type": "string", "enum": list(sections)}}},
    }
    messages = [{"role": "user", "content": question}]
    tokens_in = 0
    tokens_out = 0
    for turn in [1, 2, 3]:
        response = client.messages.create(
            model=MODEL, max_tokens=300,
            output_config={"effort": "low"},
            system=SYSTEM, tools=[read_section],
            messages=messages)
        tokens_in += response.usage.input_tokens
        tokens_out += response.usage.output_tokens
        if response.stop_reason != "tool_use":
            break
        messages.append({"role": "assistant",
                         "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                title = block.input["title"]
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": sections[title]})
        messages.append({"role": "user", "content": results})
    return answer_text(response), tokens_in, tokens_out


def measure(name, strategy, file):
    document = open(file).read()
    jobs = []
    with ThreadPoolExecutor(24) as pool:
        for question, word in QUESTIONS * 3:
            job = pool.submit(strategy, question, document)
            jobs.append((word, job))
    tokens = []
    cost = 0
    right = 0
    for word, job in jobs:
        answer, tokens_in, tokens_out = job.result()
        tokens.append(tokens_in)
        cost += (tokens_in * PRICE_IN
                 + tokens_out * PRICE_OUT) / 1e6
        if word in answer.lower():
            right += 1
    mean = round(sum(tokens) / len(tokens))
    per_1k = round(cost / len(jobs) * 1000, 2)
    print(file, name, "|", mean, "tokens in, from",
          min(tokens), "to", max(tokens), "|", right, "of",
          len(jobs), "right |", per_1k, "dollars per 1k")


measure("whole", whole, "policy.txt")
measure("progressive", progressive, "policy.txt")
measure("whole", whole, "handbook.txt")
measure("progressive", progressive, "handbook.txt")
