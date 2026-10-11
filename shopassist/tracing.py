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


def traced_create(self, **request):
    start = time.time()
    response = original_create(self, **request)
    usage = response.usage
    price_in, price_out = PRICES[request["model"]]
    record = {
        "trace": TRACE_ID,
        "agent": next(name for key, name in AGENTS.items()
                      if key in request["system"]),
        "ms": round((time.time() - start) * 1000),
        "in": usage.input_tokens, "out": usage.output_tokens,
        "cost": (usage.input_tokens * price_in
                 + usage.output_tokens * price_out) / 1e6,
        "stop": response.stop_reason,
        "tools": [b.name for b in response.content
                  if b.type == "tool_use"],
    }
    with open(LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
    return response


Messages.create = traced_create
import coordinator  # noqa: E402  - the agent code runs unchanged

rows = [json.loads(line) for line in open(LOG)]
rows = [r for r in rows if r["trace"] == TRACE_ID]
print(f"\ntrace {TRACE_ID}: {len(rows)} calls")
for r in rows:
    tools = " ".join(r["tools"])
    print(f"  {r['agent']:12} {r['ms']:6} ms {r['in']:6} in "
          f"{r['out']:5} out  {r['stop']:9} {tools}")
times = sorted(r["ms"] for r in rows)
p95 = times[round(0.95 * (len(times) - 1))]
cost = sum(r["cost"] for r in rows)
print(f"p50 {statistics.median(times):.0f} ms  p95 {p95} ms  "
      f"cost ${cost:.4f}")
slowest = max(rows, key=lambda r: r["ms"])
print(f"slowest step: {slowest['agent']}, {slowest['ms']} ms")
