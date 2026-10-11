from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from anthropic import Anthropic
from retrieval import chunk_by_section

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-5-5"
SYSTEM = "You are ShopAssist. Answer in one sentence."

QUESTIONS = [
    ("Can I return earrings?", "no"),
    ("How long does a card refund take?", "10 business days"),
    ("Who approves a refund of 250?", "human"),
    ("What happens if my return parcel is lost?", "14 days"),
    ("Do I pay customs on an international return?", "customs"),
    ("Can I exchange for another size?", "new order"),
]


def ask(**request):
    return client.messages.create(
        model=MODEL, max_tokens=300,
        output_config={"effort": "low"}, **request)


def text_of(response):
    return " ".join(b.text for b in response.content
                    if b.type == "text").lower()


def whole(question, text):
    response = ask(system=f"{SYSTEM}\n\n{text}", messages=[
        {"role": "user", "content": question}])
    return response, response.usage.input_tokens


def progressive(question, text):
    chunks = chunk_by_section(text)
    sections = {c["id"]: c["text"] for c in chunks}
    read_section = {
        "name": "read_section",
        "description": "Read one section.",
        "input_schema": {
            "type": "object", "required": ["title"],
            "properties": {"title": {
                "type": "string", "enum": list(sections)}}},
    }
    messages = [{"role": "user", "content": question}]
    used = 0
    while True:
        response = ask(system=SYSTEM, tools=[read_section],
                       messages=messages)
        used += response.usage.input_tokens
        if response.stop_reason != "tool_use":
            return response, used
        messages.append({"role": "assistant",
                         "content": response.content})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": b.id,
             "content": sections[b.input["title"]]}
            for b in response.content if b.type == "tool_use"]})


for corpus in ["policy.txt", "handbook.txt"]:
    text = open(corpus).read()
    for name, strategy in [("whole", whole),
                           ("progressive", progressive)]:
        with ThreadPoolExecutor(max_workers=6) as pool:
            runs = list(pool.map(lambda q: strategy(q[0], text),
                                 QUESTIONS))
        tokens = sum(used for _, used in runs)
        correct = sum(expected in text_of(response)
                      for (response, _), (_, expected)
                      in zip(runs, QUESTIONS))
        per = tokens // len(QUESTIONS)
        print(f"{corpus:13} {name:12} {per:5} tokens/question"
              f"  {correct}/{len(QUESTIONS)} right")
