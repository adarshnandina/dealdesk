"""Run the DealDesk eval suite against one or more models and compare them.

Examples
    # Compare two providers in LangSmith (creates the dataset on first run)
    uv run python -m evals.run --model anthropic:claude-haiku-5-5 --model openai:gpt-4.1-mini

    # Add an LLM-as-judge evaluator
    uv run python -m evals.run --model anthropic --judge anthropic:claude-sonnet-5-5

    # Run locally without uploading results to LangSmith
    uv run python -m evals.run --model anthropic --offline

The CRM is re-seeded before each run so every model sees identical data, and write
requests are never approved during evals, so runs do not change the data.
"""

from __future__ import annotations

import argparse
import inspect
import statistics
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.checkpoint.memory import InMemorySaver

from dealdesk.agent.service import AgentService, build_service
from dealdesk.config import Provider
from dealdesk.db.seed import seed
from dealdesk.db.session import session_scope
from dealdesk.llm.models import get_chat_model
from evals.dataset import DATASET_DESCRIPTION, DATASET_NAME, EXAMPLES
from evals.evaluators import DETERMINISTIC_EVALUATORS, make_judge

Evaluator = Callable[..., dict[str, Any]]
RESULTS_DIR = Path(__file__).parent / "results"
METRICS = ["correct_facts", "tool_selection", "approval_behavior", "llm_judge"]


def parse_spec(value: str) -> tuple[Provider, str | None]:
    provider, _, model = value.partition(":")
    if provider not in ("anthropic", "openai"):
        raise argparse.ArgumentTypeError(f"Unknown provider in '{value}'")
    return provider, model or None  # type: ignore[return-value]


def make_target(service: AgentService) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        with get_usage_metadata_callback() as usage:
            result = service.chat(inputs["question"], tags=["eval"])
        tokens = sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values())
        return {
            "answer": result.answer,
            "status": result.status,
            "tool_calls": result.tool_calls,
            "pending_actions": result.pending_actions,
            "latency_s": round(time.perf_counter() - start, 2),
            "total_tokens": tokens,
        }

    return target


def call_evaluator(
    fn: Evaluator, inputs: dict[str, Any], outputs: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    available = {"inputs": inputs, "outputs": outputs, "reference_outputs": reference}
    params = inspect.signature(fn).parameters
    return fn(**{k: v for k, v in available.items() if k in params})


def ensure_dataset(client: Any, recreate: bool) -> None:
    if client.has_dataset(dataset_name=DATASET_NAME):
        if not recreate:
            return
        client.delete_dataset(dataset_name=DATASET_NAME)
    dataset = client.create_dataset(DATASET_NAME, description=DATASET_DESCRIPTION)
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[e["inputs"] for e in EXAMPLES],
        outputs=[e["outputs"] for e in EXAMPLES],
        metadata=[{"category": e["outputs"]["category"]} for e in EXAMPLES],
    )
    print(f"Created LangSmith dataset '{DATASET_NAME}' with {len(EXAMPLES)} examples.")


Row = tuple[dict[str, Any], list[dict[str, Any]]]  # (outputs, evaluation results)


def run_offline(
    target: Callable[[dict[str, Any]], dict[str, Any]],
    evaluators: list[Evaluator],
    concurrency: int,
) -> list[Row]:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outputs = list(pool.map(target, [e["inputs"] for e in EXAMPLES]))
    rows: list[Row] = []
    for example, out in zip(EXAMPLES, outputs, strict=True):
        results = [
            call_evaluator(ev, example["inputs"], out, example["outputs"]) for ev in evaluators
        ]
        rows.append((out, results))
        failed = [r for r in results if r["score"] is not None and r["score"] < 1]
        if failed:
            print(f"  ✗ {example['inputs']['question']}")
            for r in failed:
                print(f"      {r['key']}={r['score']:.2f}: {r.get('comment', '')}")
    return rows


def run_langsmith(
    target: Callable[[dict[str, Any]], dict[str, Any]],
    evaluators: list[Evaluator],
    concurrency: int,
    label: str,
    metadata: dict[str, Any],
) -> list[Row]:
    from langsmith import Client

    client = Client()
    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix=f"dealdesk-{label.replace(':', '-')}",
        description=f"DealDesk agent with {label}",
        metadata=metadata,
        max_concurrency=concurrency,
    )
    rows: list[Row] = []
    for row in results:
        outputs = row["run"].outputs or {}
        evals = [
            {"key": r.key, "score": r.score, "comment": r.comment}
            for r in row["evaluation_results"]["results"]
        ]
        rows.append((outputs, evals))
    return rows


def summarize(label: str, rows: Iterable[Row]) -> dict[str, Any]:
    rows = list(rows)
    summary: dict[str, Any] = {"model": label, "n": len(rows)}
    for metric in METRICS:
        scores = [
            float(r["score"])
            for _, rs in rows
            for r in rs
            if r["key"] == metric and r["score"] is not None
        ]
        summary[metric] = statistics.mean(scores) if scores else None
    latencies = [o.get("latency_s", 0.0) for o, _ in rows]
    summary["avg_latency_s"] = statistics.mean(latencies) if latencies else None
    summary["p95_latency_s"] = (
        sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else None
    )
    summary["avg_tokens"] = statistics.mean(o.get("total_tokens", 0) for o, _ in rows)
    return summary


def to_markdown(summaries: list[dict[str, Any]]) -> str:
    def fmt(v: Any, pct: bool = False) -> str:
        if v is None:
            return "-"
        return f"{v:.0%}" if pct else f"{v:,.1f}" if isinstance(v, float) else str(v)

    header = (
        "| Model | Correct facts | Tool selection | Approval behaviour | LLM judge "
        "| Avg latency (s) | p95 latency (s) | Avg tokens |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    lines = [
        f"| {s['model']} | {fmt(s['correct_facts'], True)} | {fmt(s['tool_selection'], True)} "
        f"| {fmt(s['approval_behavior'], True)} | {fmt(s['llm_judge'], True)} "
        f"| {fmt(s['avg_latency_s'])} | {fmt(s['p95_latency_s'])} | {fmt(s['avg_tokens'])} |"
        for s in summaries
    ]
    return header + "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", action="append", type=parse_spec, required=True,
                        help="provider[:model], repeatable")  # fmt: skip
    parser.add_argument("--judge", type=parse_spec, help="provider[:model] for LLM-as-judge")
    parser.add_argument("--offline", action="store_true", help="do not upload to LangSmith")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--recreate-dataset", action="store_true")
    parser.add_argument("--no-seed", action="store_true", help="skip re-seeding the CRM")
    args = parser.parse_args(argv)

    if not args.no_seed:
        with session_scope() as session:
            seed(session)

    evaluators: list[Evaluator] = list(DETERMINISTIC_EVALUATORS)
    if args.judge:
        evaluators.append(make_judge(get_chat_model(*args.judge, temperature=0.0)))

    if not args.offline:
        from langsmith import Client

        ensure_dataset(Client(), args.recreate_dataset)

    summaries = []
    for provider, model in args.model:
        service = build_service(checkpointer=InMemorySaver(), provider=provider, model=model)
        label = service.spec.label
        print(f"\n▶ Evaluating {label} on {len(EXAMPLES)} examples")
        target = make_target(service)
        if args.offline:
            rows = run_offline(target, evaluators, args.concurrency)
        else:
            meta = {"llm_provider": provider, "llm_model": service.spec.model}
            rows = run_langsmith(target, evaluators, args.concurrency, label, meta)
        summaries.append(summarize(label, rows))

    table = to_markdown(summaries)
    print("\n" + table)
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    out = RESULTS_DIR / "results.md"
    with out.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## Run {stamp}\n\n{table}")
    print(f"Appended results to {out}")


if __name__ == "__main__":
    main()
