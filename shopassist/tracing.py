import json
import statistics
import time
import uuid
from anthropic.resources.messages import Messages
import coordinator

# dollars per million tokens: input, output (pricing page)
PRICES = {"claude-sonnet-5-5": (2.00, 10.00)}
AGENTS = {"coordinator": "coordinator", "check orders": "order",
          "billing": "billing"}
LOG = "runs.jsonl"


def agent_of(system):
    for words, name in AGENTS.items():
        if words in system:
            return name
    return "unknown"


def log_step(step, ms, tokens_in, tokens_out, cost, status):
    record = {"trace": TRACE_ID, "step": step, "ms": ms,
              "in": tokens_in, "out": tokens_out,
              "cost": cost, "status": status}
    with open(LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


original_create = Messages.create


def traced_create(self, **request):
    start = time.time()
    response = original_create(self, **request)
    ms = round((time.time() - start) * 1000)
    price_in, price_out = PRICES[request["model"]]
    usage = response.usage
    cost = (usage.input_tokens * price_in
            + usage.output_tokens * price_out) / 1e6
    log_step(agent_of(request["system"]) + " model", ms,
             usage.input_tokens, usage.output_tokens, cost,
             response.stop_reason)
    return response


original_tool = coordinator.call_tool


def traced_tool(agent, name, arguments):
    start = time.time()
    output = original_tool(agent, name, arguments)
    ms = round((time.time() - start) * 1000)
    status = "ok"
    if output.get("isError"):
        status = "error"
    log_step(agent + " " + name, ms, 0, 0, 0, status)
    return output


Messages.create = traced_create
coordinator.call_tool = traced_tool


def summary(trace_id):
    rows = []
    for line in open(LOG):
        row = json.loads(line)
        if row["trace"] == trace_id:
            rows.append(row)
    total = 0
    cost = 0
    slowest = rows[0]
    for row in rows:
        total += row["ms"]
        cost += row["cost"]
        if row["ms"] > slowest["ms"]:
            slowest = row
    share = round(slowest["ms"] / total * 100)
    print(trace_id, "|", len(rows), "steps |",
          round(total / 1000, 1), "s |", round(cost, 3),
          "dollars | slowest:", slowest["step"],
          slowest["ms"], "ms,", share, "% of the run")


for run in [1, 2, 3]:
    TRACE_ID = uuid.uuid4().hex[:8]
    coordinator.handle(coordinator.MESSAGE)
    summary(TRACE_ID)

times = []
for line in open(LOG):
    times.append(json.loads(line)["ms"])
p50 = statistics.median(times)
p95 = statistics.quantiles(times, n=20)[-1]
print("every step in the log | p50", round(p50), "ms | p95",
      round(p95), "ms")
