import json
import statistics
import time
import uuid
from anthropic.resources.messages import Messages

# dollars per million tokens: input, output (pricing page)
PRICES = {"claude-sonnet-5-5": (2.00, 10.00)}
AGENTS = {"coordinator": "coordinator", "check orders": "order",
          "billing": "billing"}
TRACE_ID = uuid.uuid4().hex[:8]
LOG = "runs.jsonl"
original_create = Messages.create


def agent_of(system):
    for words, name in AGENTS.items():
        if words in system:
            return name
    return "unknown"


def traced_create(self, **request):
    start = time.time()
    response = original_create(self, **request)
    ms = round((time.time() - start) * 1000)
    usage = response.usage
    price_in, price_out = PRICES[request["model"]]
    tools = []
    for block in response.content:
        if block.type == "tool_use":
            tools.append(block.name)
    record = {
        "trace": TRACE_ID,
        "agent": agent_of(request["system"]),
        "ms": ms,
        "in": usage.input_tokens, "out": usage.output_tokens,
        "cost": (usage.input_tokens * price_in
                 + usage.output_tokens * price_out) / 1e6,
        "stop": response.stop_reason,
        "tools": tools,
    }
    with open(LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
    return response


Messages.create = traced_create
import coordinator  # runs the agents, unchanged

rows = []
for line in open(LOG):
    row = json.loads(line)
    if row["trace"] == TRACE_ID:
        rows.append(row)
print("trace", TRACE_ID, "has", len(rows), "calls")
for r in rows:
    tools = " ".join(r["tools"])
    print(f"  {r['agent']:12} {r['ms']:6} ms {r['in']:6} in "
          f"{r['out']:5} out  {r['stop']:9} {tools}")

times = []
cost = 0
for r in rows:
    times.append(r["ms"])
    cost += r["cost"]
times.sort()
p95 = times[round(0.95 * (len(times) - 1))]
print("median", round(statistics.median(times)), "ms, p95", p95,
      "ms, cost", round(cost, 4), "dollars")
slowest = rows[0]
for r in rows:
    if r["ms"] > slowest["ms"]:
        slowest = r
print("slowest step:", slowest["agent"], slowest["ms"], "ms")
