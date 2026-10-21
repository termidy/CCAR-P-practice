from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-5-5"
SYSTEM = "You are ShopAssist. Answer in one sentence."

QUESTIONS = [
    ("Can I return earrings?", "hygiene"),
    ("Who approves a refund of 250?", "human"),
    ("Do I pay customs on an international return?", "authority"),
]


def answer_text(response):
    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text
    return text.lower()


def whole(question, document):
    response = client.messages.create(
        model=MODEL, max_tokens=300,
        output_config={"effort": "low"},
        system=SYSTEM + "\n\n" + document,
        messages=[{"role": "user", "content": question}])
    return response, response.usage.input_tokens


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
        "description": "Read one section.",
        "input_schema": {
            "type": "object", "required": ["title"],
            "properties": {"title": {
                "type": "string", "enum": list(sections)}}},
    }
    messages = [{"role": "user", "content": question}]
    used = 0
    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=300,
            output_config={"effort": "low"},
            system=SYSTEM, tools=[read_section],
            messages=messages)
        used += response.usage.input_tokens
        if response.stop_reason != "tool_use":
            return response, used
        messages.append({"role": "assistant",
                         "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": sections[block.input["title"]]})
        messages.append({"role": "user", "content": results})


def test(name, strategy, file):
    document = open(file).read()
    tokens = 0
    right = 0
    for question, expected in QUESTIONS:
        response, used = strategy(question, document)
        tokens += used
        if expected in answer_text(response):
            right += 1
    per = tokens // len(QUESTIONS)
    print(file, name, per, "tokens per question,",
          right, "of", len(QUESTIONS), "right")


test("whole", whole, "policy.txt")
test("whole", whole, "handbook.txt")
test("progressive", progressive, "handbook.txt")
