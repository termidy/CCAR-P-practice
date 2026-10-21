import json
import os
import sys
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-5-5"
LIMIT = 100
QUEUE = "review_queue.json"
SYSTEM = ("You are ShopAssist. Use process_refund for refunds. "
          "Answer the customer in two sentences.")
TOOLS = [{
    "name": "process_refund",
    "description": "Refund an amount for an order.",
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "amount": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["order_id", "amount", "reason"],
    },
}]


def load_queue():
    if not os.path.exists(QUEUE):
        return []
    return json.load(open(QUEUE))


def agent(text):
    messages = [{"role": "user", "content": text}]
    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=300,
            output_config={"effort": "low"},
            system=SYSTEM, tools=TOOLS, messages=messages)
        messages.append({"role": "assistant",
                         "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                out = process_refund(**block.input)
                print("  process_refund", block.input, "->", out)
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(out)})
        messages.append({"role": "user", "content": results})
    for block in response.content:
        if block.type == "text":
            return block.text


def process_refund(order_id, amount, reason):
    if amount <= LIMIT:
        return {"status": "refunded", "refund_id": "REF-7781"}
    queue = load_queue()
    ticket = f"REV-{len(queue) + 1}"
    queue.append({"ticket": ticket, "order_id": order_id,
                  "amount": amount, "reason": reason,
                  "decision": None})
    json.dump(queue, open(QUEUE, "w"), indent=2)
    return {"status": "pending_review", "ticket": ticket,
            "customerMessage": "A person will review it."}


def review(ticket, decision):
    queue = load_queue()
    for c in queue:
        if c["ticket"] == ticket:
            case = c
    case["decision"] = decision
    if decision == "approved":
        case["refund_id"] = "REF-7790"  # code refunds, not Claude
    json.dump(queue, open(QUEUE, "w"), indent=2)
    response = client.messages.create(
        model=MODEL, max_tokens=200,
        output_config={"effort": "low"}, system=SYSTEM,
        messages=[{"role": "user", "content":
                   "Write the customer update for this reviewed "
                   f"refund: {json.dumps(case)}"}])
    for block in response.content:
        if block.type == "text":
            return block.text


if len(sys.argv) == 1:
    requests = [
        "Refund 60 for the jacket in ORD-12345678, "
        "it was too small.",
        "Refund 250 for the broken TV in ORD-87654321.",
        "Refund 180 for the coat in ORD-24681357, "
        "I changed my mind.",
    ]
    for text in requests:
        print("Customer:", text)
        print("ShopAssist:", agent(text))
else:
    print("ShopAssist:", review(sys.argv[2], sys.argv[3]))
