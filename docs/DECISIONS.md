# Design decisions

Short records of the main choices: what was decided, why, the trade-off, and what would
change at larger scale.

## 1. A custom LangGraph `StateGraph` instead of a prebuilt agent

**Decision.** The agent loop is written explicitly: an `agent` node, a `ToolNode`, an
`approval` node, and a router function between them.

**Why.** The one rule that matters most here — *no write without human approval* — is a
control-flow rule. With an explicit graph it is a visible edge (`route_after_agent` sends
any batch containing a write to `approval`) that can be unit-tested, instead of
behaviour hidden inside a prebuilt agent. It also makes traces easy to read, because each
node is a named step.

**Trade-off.** More code than `langchain.agents.create_agent` with its human-in-the-loop
middleware, which would also work. The prebuilt agent is a good choice when the standard
loop is enough; a custom graph pays off once you need custom routing.

## 2. Approval is enforced by the graph, not by the prompt

**Decision.** Write tools are listed in `WRITE_TOOL_NAMES`. The router checks every tool
call; the prompt only tells the model *not* to ask for confirmation itself.

**Why.** Prompts are suggestions; the model can ignore them. Putting the rule in code
makes it deterministic and testable (`test_write_pauses_for_approval_then_runs`). If a
batch mixes reads and writes, the whole batch waits, so we never end up with half a plan
executed. On rejection, every tool call gets a `ToolMessage` saying it was rejected, so
the message history stays valid for every provider and the model can explain what
happened.

## 3. Durable state with the Postgres checkpointer

**Decision.** `PostgresSaver` stores graph state per `thread_id`.

**Why.** An approval can take hours. With in-memory state a deploy or a crash would lose
the paused run, and with several API workers the approval might hit a different process.
Postgres is already there for the CRM data, so there is no new infrastructure. A test
pauses a run, builds a completely new saver and service ("restart"), and then approves.

## 4. Model-agnostic layer via `init_chat_model`

**Decision.** All code depends on `BaseChatModel` and gets models from
`get_chat_model(provider, model)`.

**Why.** Different internal tools need different trade-offs (cost, latency, quality), and
models change every few months. Provider choice is configuration, so the eval harness can
run the identical agent against each candidate and the decision is based on data.
`DEALDESK_OPENAI_BASE_URL` covers any OpenAI-compatible endpoint (GitHub Models, Azure,
a gateway). Temperature can be unset because some reasoning models reject it.

## 5. Repository pattern + DTOs

**Decision.** All SQL lives in `CRMRepository`, which returns frozen Pydantic DTOs.

**Why.** The tools, the API and the MCP server share one data-access path with one set
of business rules (due date not in the past, contact must belong to the deal's company,
stage change writes an audit note). Replacing the mock database with HubSpot means
writing one new class. DTOs give the LLM a compact, stable JSON shape and avoid leaking
ORM sessions between layers.

## 6. Tools return errors as data

**Decision.** Domain errors become `{"error": "..."}` tool results instead of exceptions.

**Why.** The model can read the error and recover (for example, look the deal up again)
instead of the run failing. Unexpected exceptions still surface through `ToolNode`'s error
handling and in the trace.

## 7. One transaction per tool call

**Why.** A write either fully commits or fully rolls back. Long-lived sessions across LLM
calls would hold connections open while waiting for a slow model.

## 8. SQLAlchemy 2.0 + Alembic, enums as VARCHAR + CHECK

**Why.** Typed ORM models and versioned, reviewable migrations. Enums are stored as
`VARCHAR` with a `CHECK` constraint rather than native Postgres `ENUM`, because adding a
new pipeline stage is then a simple migration. Tests run `downgrade base` + `upgrade head`,
so the migrations are tested too.

## 9. HubSpot-shaped data model

**Why.** Owners, companies, contacts, deals with HubSpot's internal stage IDs, activities
(engagements) and tasks. That makes the agent's tools map directly onto HubSpot's API
later, and the data feels like real sales work. The seed data is hand-written so eval
answers are fixed.

## 10. Webhook: verify, sync, then let the agent propose

**Why.** The HMAC signature is checked over the raw body before parsing (never trust
unsigned input). The event's `event_id` makes processing idempotent, because webhook
senders retry. The handler returns `202` quickly and does the slow LLM work in the
background. The agent's proposals still go through the normal approval flow.

## 11. Evals: deterministic first, LLM-as-judge second

**Why.** Deterministic evaluators (facts, amounts, tool choice, approval behaviour) are
free, fast and reproducible, so they can run on every change. The LLM judge catches
things string checks miss (hallucinated details, wrong reasoning) but costs money and has
its own variance, so it is optional. Each model is a LangSmith experiment on the same
dataset, compared on quality, latency and tokens.

## 12. Synchronous code paths

**Decision.** Sync SQLAlchemy, sync graph invocation; FastAPI runs `def` endpoints in a
thread pool.

**Why.** Simpler to read and debug, and the bottleneck is LLM latency, not Python
concurrency. At higher load I would move to async SQLAlchemy and `AsyncPostgresSaver`,
and stream tokens to the client over SSE.

---

## Likely interview questions

**How do you stop the agent from changing data it shouldn't?** Writes are routed through
an approval interrupt by code, not by prompt; write tools are a small explicit list; the
repository validates business rules; every agent write is stored with `source='agent'`.

**How would you decide which model to use for an internal tool?** Run the same graph
against each candidate on the eval dataset, compare correctness, tool use, approval
behaviour, latency and tokens/cost in LangSmith, then pick the cheapest model that meets
the quality bar for that tool.

**How would you migrate existing tools onto this stack?** Inventory how each tool calls
an LLM; move model access to the shared factory first (smallest change, immediate
tracing); then move tool logic into typed tools/repositories; then add evals before
changing prompts or models.

**What does a trace show you when an answer is wrong?** Whether the model chose the wrong
tool, passed wrong arguments, got wrong data back, or misread correct data. Each needs a
different fix (tool description, schema, repository, prompt).

**What would you do differently in production?** Real auth and per-user permissions on
writes, the HubSpot adapter, streaming, rate limits and cost budgets per tool, online
evals on sampled traffic, and alerts on error rate and latency.
