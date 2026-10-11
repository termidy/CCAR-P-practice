# CCAR-P practice: ShopAssist in code

Companion code for the practice lessons of the **Claude Certified Architect - Professional (CCAR-P)** course. Each practice lesson is recorded in VS Code against the real Claude API, and this repository holds the project you see on screen, so you can run every step yourself.

| Folder | What it is |
| --- | --- |
| [`shopassist/`](shopassist/) | The practice project: one small Python script per lesson, all built on the same fictional support assistant, ShopAssist. Needs an Anthropic API key. |
| [`starting-points/`](starting-points/) | The "before" version of the files that a lesson edits on screen, so you can replay the edits yourself. |
| [`labs/`](labs/) | Jupyter notebooks to practise **without an API key**. |

## Setup

```bash
git clone https://github.com/termidy/CCAR-P-practice.git
cd CCAR-P-practice/shopassist
python3 -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt
cp .env.example .env                                    # put your key in .env, never in the code
```

`.env` holds one line, `ANTHROPIC_API_KEY=...`; every script loads it with `python-dotenv`. Some lessons also need [Claude Code](https://docs.claude.com/en/docs/claude-code) installed and logged in.

The scripts use the current API: `claude-sonnet-5-5`, `output_config={"effort": "low"}`, structured outputs through
`output_config.format`, and the `mcp` package 2.x for the MCP server. If something returns a 400, check the model id and the SDK
version first (`anthropic>=1.12`).

## The lessons

In the order of the course.

| Lesson | File(s) | What you run |
| --- | --- | --- |
| First request in code | `first_request.py` | `python first_request.py` - one `messages.create`, then `content`, `stop_reason` and `usage`; lower `max_tokens` to 20 and watch the reply get cut |
| The conversation lives in your code | `conversation.py` | `python conversation.py` - the `messages` list is the only memory |
| A chat UI talks to your script | `app.py`, `static/index.html` | `uvicorn app:app --port 8000`, open http://127.0.0.1:8000/ |
| System prompt rules vs code rules | `rules.py` (start: `starting-points/p04_rules.start.py`) | `python rules.py` - a prompt rule is a request, a code rule is a guarantee |
| Evaluate a prompt change on a test set | `evaluate.py` | `python evaluate.py` - two system prompts, same cases, scored by code and by a model grader |
| Chunking strategies, measured | `retrieval.py`, `chunking.py` | `python chunking.py` - fixed-size vs by-section chunks, recall@3 on twelve questions |
| Grounded answers with citations | `grounded.py` | `python grounded.py` - retrieved sections go to Claude as `search_result` blocks; the answer cites its source |
| Diagnose retrieval before blaming the model | `diagnose.py`, `policy_live.txt`, `index.json` | `python diagnose.py`, then `python diagnose.py rebuild` - a stale index gives a confident wrong answer |
| Least privilege and authorization in tools | `privilege.py` | `python privilege.py` - six tools vs a role of three (token cost), identity from the session inside the tool |
| Progressive discovery vs one big context | `discovery.py`, `handbook.txt` | `python discovery.py` - the whole text in the prompt vs a table of contents and a read tool |
| A model scorecard: Haiku vs Sonnet | `scorecard.py` | `python scorecard.py` - correctness, median latency and cost per thousand requests |
| One trace across an agent run | `tracing.py` | `python tracing.py` - every call logged with a trace id; p50, p95, cost and the slowest step |
| Human-in-the-loop: a review queue | `hitl.py` | `python hitl.py`, then `cat review_queue.json`, then `python hitl.py review REV-1 approved` |
| Your first tool call, end to end | `tool_call.py` | `python tool_call.py` - `tools=`, `stop_reason == "tool_use"`, run the function, send `tool_result` back |
| Schema-valid output from Claude | `extract.py` | `python extract.py` - `output_config.format` with a JSON schema |
| Validate, then retry with the error | `validate.py` (start: `starting-points/p08_validate.start.py`) | `python validate.py` - schema-valid is not business-valid; retry once with the exact error |
| BUILD: ShopAssist extracts return requests | `build_extract.py`, `emails/` | `python build_extract.py` - five emails in, `return_requests.json` out |
| Structured tool errors in code | `tool_errors.py` (start: `starting-points/p11_tool_errors.start.py`) | `python tool_errors.py` - tools return `isError`, `errorCategory`, `isRetryable`, `customerMessage`, `developerMessage` |
| BUILD: backend tools and an MCP server | `mcp_server.py` | `claude mcp add shopassist -- python mcp_server.py`, then `claude -p "..." --allowedTools "mcp__shopassist__lookup_order,mcp__shopassist__start_return"` |
| The agentic loop in code | `agent_loop.py` | `python agent_loop.py` - loop while `stop_reason == "tool_use"` |
| Coordinator and subagents in code | `coordinator.py` | `python coordinator.py` - a coordinator calls two specialised agents as tools |
| BUILD: ShopAssist becomes an agentic system | `agent.py` (start: `starting-points/p13_agent.start.py`) | `python agent.py` - a gate before the loop, a hook before every tool, escalation as a tool |
| Prompt caching: pay for the policy once | `caching.py`, `policy.txt` | `python caching.py` - `cache_control` on the policy block; usage shows cache write once, then cache reads |
| Prompt injection through a tool result | `injection.py` (start: `starting-points/d29_injection.start.py`) | `python injection.py` - an order note demands a 500 refund; then `python -c "from injection import process_refund; print(process_refund('ORD-12345678', 500))"` shows the code gate |
| PII and secrets: redact before you send | `redact.py` (start: `starting-points/d30_redact.start.py`) | `python redact.py` - emails, card and phone numbers become placeholders before the request; `git check-ignore -v .env` |
| Claude Code hooks: block a destructive command | `.claude/settings.json`, `.claude/hooks/guard.py` | `claude -p "Show the first line of .env with head, then delete review.json with rm. Report what happened." --allowedTools Bash` - both commands are blocked by the hook |
| Stream the reply as it is written | `stream.py` (start: `starting-points/d07_stream.start.py`) | `python stream.py` - `messages.stream`, text pieces as they arrive, `get_final_message()` for stop_reason and usage |
| Errors and retries: 401, 400 and timeouts | `errors.py` | `python errors.py` - a wrong key, a removed parameter through `extra_body`, a 50 ms timeout; `max_retries` and `with_options` |
| Claude Code hands-on: CLAUDE.md, rules and a skill | `CLAUDE.md`, `.claude/rules/tools.md`, `.claude/skills/order-check/SKILL.md` | `claude -p "Use the order-check skill for order ORD-87654321." --allowedTools "Skill,Bash(python *)"` |
| Claude Code in CI: `-p` and JSON output | `ci_review.sh`, `review_schema.json` | break a tool convention in `mcp_server.py`, then `sh ci_review.sh` |

The retrieval lessons use a small BM25 keyword index in pure Python so that they run without extra services; in production the ranking is usually embeddings or a hybrid of both.

The order IDs the tools know are `ORD-12345678` (a jacket, delivered 20 days ago) and `ORD-87654321` (a lamp, delivered 45 days ago).
Everything else returns `found: false`, which is data, not an error.

The hook in `.claude/settings.json` is active in this folder: Claude Code cannot run `rm` or touch `.env` here, which is the point of
the hook lesson.

## The request trace

`shopassist/.trace/` is the small observer you see beside the editor in the videos. `sitecustomize.py` logs every HTTP exchange of
the Anthropic client to `trace.jsonl` when the folder is on `PYTHONPATH`; `trace.html` and `trace_mcp.html` draw it. It is optional
and the scripts never depend on it:

```bash
cd shopassist
export PYTHONPATH="$PWD/.trace"
(cd .trace && python3 -m http.server 8765 &)      # then open http://127.0.0.1:8765/trace.html
python first_request.py
```

## Labs

[`labs/`](labs/) holds hands-on Jupyter notebooks, a way to practise **without an API key**: a small simulator stands in for the SDK, and
the code you write is the code you would write against the real API. `labs/README.md` explains how to run them.

## License

The code is free to use for learning and in your own projects.
