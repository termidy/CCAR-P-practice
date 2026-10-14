# Labs

Hands-on labs: a way to practise without an API key.

**You do not need an API key.** Each lab ships with `shopassist_lab.py`, a small
standard-library-only simulator of the `anthropic` SDK. The code you write is exactly the
code you would write against the real API; the replies come from rules, not a real model.
Set `ANTHROPIC_API_KEY` in your environment (or in a `.env` file) and the very same notebook
calls Claude for real.

## How to run a lab

```bash
cd labs/lab_progressive_discovery
jupyter lab            # or open notebook.ipynb in VS Code
```

Open **`notebook.ipynb`** and work through it top to bottom. Every cell already holds the
code, with blanks marked `...` and a comment saying what goes in each one. Each task ends
with a `check(...)` call that prints PASS with an explanation or FAIL with a specific hint.
`START_HERE.md` in the lab folder has the details; `solution.ipynb` is the completed
notebook if you get stuck.

The only requirement is Python 3.10+ and Jupyter (`pip install jupyter`). `anthropic` and
`python-dotenv` are optional: when one is missing, the simulator stands in for it. An
installed `anthropic` is used only when `ANTHROPIC_API_KEY` is set; without a key the
simulator answers, so no lab fails with an authentication error.

## The labs

In course order. Each lab mirrors a script in [`shopassist/`](../shopassist/) and takes it
one step further, the way an architect would: from a working script to a decision you can
defend with numbers.

| Lab | Practises | Script |
| --- | --- | --- |
| `lab_eval_release` | A test set by category and on both sides of a boundary, scores per category, a release decision with known gaps and its cost | `evaluate.py` |
| `lab_cache_economics` | What a cached request is made of, the price of each kind and the break-even of a write, the cache lifetime chosen by the traffic | `caching.py` |
| `lab_trust_boundaries` | Untrusted text labelled where it enters, risk tiers in one guard with default deny, an audit log when the prompt layer fails | `injection.py` |
| `lab_progressive_discovery` | The whole document in the prompt vs a table of contents and a read tool, measured on a small and a large document | `discovery.py` |
| `lab_chunking` | Fixed-size vs section chunks, recall@3 on real questions, chunk size weighed against what it sends | `chunking.py` |
| `lab_grounded_citations` | Search results with citations, the answer with its sources, no citation - no answer, and the grounding rate | `grounded.py` |
| `lab_retrieval_diagnosis` | A record of what retrieval returned, a staleness check against the source, a rebuild, no answers from a stale index | `diagnose.py` |
| `lab_least_privilege` | Tools per role priced with count_tokens, identity from the session inside the tool, a claimed-identity attack | `privilege.py` |
| `lab_agentic_system` | Outcomes from case facts, automation and escalation rates, a policy change replayed before anyone decides | `agent.py` |
| `lab_model_scorecard` | Correct, median time and price per thousand for Haiku and Sonnet, and the cheapest model that meets the bar per step | `scorecard.py` |
| `lab_agent_trace` | messages.create wrapped once, one trace as a table with p50 and p95, cost per agent across runs | `tracing.py` |
| `lab_review_queue` | Human approval inside the tool, the decision carried out by code and explained by Claude, a queue with an SLA | `hitl.py` |

The other practice lessons are the same videos as in the CCAR-F and CCDV-F courses; their
labs are in [CCAR-F-practice](https://github.com/termidy/CCAR-F-practice) and
[CCDV-F-practice](https://github.com/termidy/CCDV-F-practice).

## The simulator follows the current API

The labs use `claude-sonnet-5-5` (and `claude-haiku-5-5` in the scorecard) with
`output_config={"effort": "low"}`, as the lessons do. The simulator answers from the text a
request carries - a policy in the system prompt, excerpts, search results or sections a tool
returned - so a fact it was not given cannot be in the answer. It checks search_result blocks
and returns citations, counts tokens with `messages.count_tokens()`, caches prompts up to
`cache_control`, and raises the SDK's error classes. Where a lab needs the simulator to play a
weaker model or a model that follows an injected instruction, the notebook says so.
