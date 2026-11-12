import time
from datetime import datetime
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-5-5"
# dollars per million tokens (pricing page)
PRICE = {"in": 2.00, "write": 2.50, "read": 0.10,
         "out": 10.00}
POLICY = open("policy.txt").read()
INSTRUCTION = ("You are ShopAssist. Answer from the policy "
               "below in one short sentence.")
QUESTIONS = ["Can I return earrings?",
             "How long does a card refund take?",
             "Who approves a refund of 600?"]


def ask(question):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start = time.time()
    times = []
    with client.messages.stream(
        model=MODEL,
        max_tokens=200,
        output_config={"effort": "low"},
        system=[
            {"type": "text", "text": INSTRUCTION},
            {"type": "text", "text": POLICY,
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Current time: " + now},
        ],
        messages=[{"role": "user", "content": question}],
    ) as stream:
        for text in stream.text_stream:
            times.append(time.time() - start)
        response = stream.get_final_message()
    u = response.usage
    cost = (u.input_tokens * PRICE["in"]
            + u.cache_creation_input_tokens * PRICE["write"]
            + u.cache_read_input_tokens * PRICE["read"]
            + u.output_tokens * PRICE["out"]) / 1_000_000
    print(f"in {u.input_tokens} | write "
          f"{u.cache_creation_input_tokens} | read "
          f"{u.cache_read_input_tokens} | first token "
          f"{times[0]:.2f} s | ${cost * 1000:.2f} per 1k")


for question in QUESTIONS:
    ask(question)
