import re
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-5-5"

message = (
    "Hi, my email is alex@example.com and my phone is "
    "+353 85 123 4567. I paid with card 4111 1111 1111 1111 "
    "for order ORD-12345678 and want to return the jacket."
)

PATTERNS = {
    "EMAIL": r"\S+@\S+",
    "PHONE": r"\+[\d ]+\d",
    "CARD": r"\d{4} \d{4} \d{4} \d{4}",
}


def redact(text):
    for name, pattern in PATTERNS.items():
        text = re.sub(pattern, "<" + name + ">", text)
    return text


safe = redact(message)
print("sent to Claude:", safe)

SYSTEM = (
    "You are ShopAssist, an online store's support assistant. "
    "Returns are accepted within 30 days of delivery. "
    "Personal data arrives as placeholders such as <EMAIL>; "
    "never ask the customer to send it. Answer in two sentences."
)

response = client.messages.create(
    model=MODEL,
    max_tokens=200,
    output_config={"effort": "low"},
    system=SYSTEM,
    messages=[{"role": "user", "content": safe}],
)
for block in response.content:
    if block.type == "text":
        print(block.text)
