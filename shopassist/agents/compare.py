import json
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from anthropic import Anthropic
from backend import FUNCTIONS
from policy_search import search_policy
from cases import CASES

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-5-5"
# dollars per million tokens: input, output (pricing page)
PRICE_IN, PRICE_OUT = 2.00, 10.00
LOG = "runs.jsonl"
LOCK = threading.Lock()
FUNCTIONS["search_policy"] = search_policy
SPECIALISTS = {}
DONE = {}    # trace ID -> what the backend did
USAGE = {}   # trace ID -> input and output tokens


def make_tool(name, description, fields):
    return {"name": name, "description": description,
            "input_schema": {"type": "object",
                             "properties": fields,
                             "required": list(fields)}}


ORDER_TOOLS = [
    make_tool("lookup_order", "Look up one of the customer's "
              "orders.", {"order_id": {"type": "string"}}),
    make_tool("start_return", "Start a return for an order.",
              {"order_id": {"type": "string"}}),
    make_tool("process_refund", "Refund an amount for an order.",
              {"order_id": {"type": "string"},
               "amount": {"type": "number"},
               "reason": {"type": "string"}}),
    make_tool("escalate", "Hand the case to a person.",
              {"reason": {"type": "string"}}),
]
POLICY_TOOLS = [
    make_tool("search_policy", "Search the return policy.",
              {"query": {"type": "string"}}),
]
SYSTEM = ("You are ShopAssist, DemoShop's support assistant. "
          "Do everything the customer asks. A refund above "
          "the automatic limit goes to a person: escalate it. "
          "Answer from the policy, never from memory. Start "
          "a yes-or-no answer with Yes or No. Answer in at "
          "most three sentences.")


def log_step(trace, agent, step, data):
    record = {"trace": trace, "agent": agent, "step": step}
    record.update(data)
    with LOCK:
        with open(LOG, "a") as f:
            f.write(json.dumps(record) + "\n")


def call_tool(trace, agent, name, arguments):
    log_step(trace, agent, name, {"input": arguments})
    if name in SPECIALISTS:
        return SPECIALISTS[name](trace, **arguments)
    output = FUNCTIONS[name](**arguments)
    if name in ["start_return", "process_refund"]:
        if not output.get("isError"):
            DONE[trace].append(name + " " + arguments["order_id"])
    if name == "escalate":
        DONE[trace].append(name)
    return output


def run_agent(trace, name, system, tools, task):
    messages = [{"role": "user", "content": task}]
    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=1000,
            output_config={"effort": "low"},
            system=system, tools=tools, messages=messages)
        USAGE[trace][0] += response.usage.input_tokens
        USAGE[trace][1] += response.usage.output_tokens
        if response.stop_reason != "tool_use":
            break
        messages.append({"role": "assistant",
                         "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = call_tool(trace, name, block.name,
                                   block.input)
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(output)})
        messages.append({"role": "user", "content": results})
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def single(trace, message):
    return run_agent(trace, "single", SYSTEM,
                     ORDER_TOOLS + POLICY_TOOLS, message)


def grade(case, answer, done):
    text = answer.lower()
    for action in case["actions"]:
        if action not in done:
            return False
    for action in case["not"]:
        if action in done:
            return False
    for word in case["words"]:
        if word not in text:
            return False
    for word in case.get("never", []):
        if word in text:
            return False
    return True


def one(config, case):
    trace = "trc-" + uuid.uuid4().hex[:6]
    DONE[trace] = []
    USAGE[trace] = [0, 0]
    start = time.time()
    answer = config(trace, case["message"])
    seconds = time.time() - start
    ok = grade(case, answer, DONE[trace])
    log_step(trace, config.__name__, "answer",
             {"case": case["id"], "ok": ok, "text": answer})
    tokens_in, tokens_out = USAGE[trace]
    cost = (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6
    return ok, tokens_in + tokens_out, cost, seconds


def measure(config):
    jobs = []
    with ThreadPoolExecutor(36) as pool:
        for run in [1, 2, 3]:
            for case in CASES:
                job = pool.submit(one, config, case)
                jobs.append((run, case, job))
    right = {}
    tokens, times, cost, missed = [], [], 0, []
    for run, case, job in jobs:
        ok, case_tokens, case_cost, seconds = job.result()
        key = (case["kind"], run)
        right[key] = right.get(key, 0) + ok
        tokens.append(case_tokens)
        times.append(seconds)
        cost += case_cost
        if not ok:
            missed.append(case["id"])
    simple, compound = [], []
    for run in [1, 2, 3]:
        simple.append(str(right[("simple", run)]))
        compound.append(str(right[("compound", run)]))
    p50 = statistics.median(times)
    p95 = statistics.quantiles(times, n=20)[-1]
    per_1k = cost / len(jobs) * 1000
    print(config.__name__, "| right in each run: simple",
          " ".join(simple), "of 8 | compound",
          " ".join(compound), "of 4")
    print("   ", round(statistics.mean(tokens)), "tokens |",
          round(per_1k, 1), "dollars per 1k cases | p50",
          round(p50, 1), "s | p95", round(p95, 1), "s")
    for case_id in sorted(set(missed)):
        print("    missed", case_id, "in",
              missed.count(case_id), "of 3 runs")


def show_trace(case_id):
    rows = []
    for line in open(LOG):
        rows.append(json.loads(line))
    trace = None
    for row in rows:
        if row["step"] == "answer" and row["case"] == case_id:
            if row["agent"] == "coordinated" and not row["ok"]:
                trace = row["trace"]
    for row in rows:
        if row["trace"] == trace:
            text = json.dumps(row.get("input", row.get("text")))
            print(row["agent"], "->", row["step"], text[:90])


def orders_specialist(trace, customer_message, policy):
    task = ("The customer wrote: " + customer_message
            + "\nThe policy rules: " + policy)
    return run_agent(trace, "orders",
                     "You handle the customer's orders: look "
                     "up, return, refund, escalate. "
                     "Follow the policy rules in your task. "
                     "A refund above the automatic limit goes "
                     "to a person: escalate it. Report what "
                     "you did in one sentence.", ORDER_TOOLS,
                     task)


def policy_specialist(trace, question):
    return run_agent(trace, "policy",
                     "You answer questions from the return "
                     "policy. Search it first. Answer in one "
                     "sentence.", POLICY_TOOLS, question)


SPECIALISTS["orders_specialist"] = orders_specialist
SPECIALISTS["policy_specialist"] = policy_specialist
COORDINATOR_TOOLS = [
    make_tool("orders_specialist", "Delegate order work: look "
              "ups, returns, refunds.",
              {"customer_message": {"type": "string",
                   "description": "The customer's message, "
                                  "word for word."},
               "policy": {"type": "string", "description":
                   "The policy rules for this case, from the "
                   "policy specialist. Empty for a look up."}}),
    make_tool("policy_specialist", "Delegate a question about "
              "the return policy.",
              {"question": {"type": "string"}}),
]


def coordinated(trace, message):
    return run_agent(trace, "coordinator",
                     SYSTEM + " You have no data tools: "
                     "delegate to the specialists.",
                     COORDINATOR_TOOLS, message)


if len(sys.argv) > 1:
    show_trace(sys.argv[1])
else:
    open(LOG, "w").close()
    measure(single)
    measure(coordinated)
