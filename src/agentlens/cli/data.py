"""Data commands: ``baseline``, ``experiments``, ``traces``, ``export`` and ``import``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from agentlens.cli._common import fail, open_store
from agentlens.report import fmt_value, table
from agentlens.schema import Experiment, Span, Trace
from agentlens.store import Store, export_jsonl, import_jsonl
from agentlens.suite import find_experiment, get_baseline, set_baseline, unset_baseline

__all__ = [
    "baseline_app",
    "experiments_app",
    "export_traces",
    "import_traces",
    "traces_app",
]

baseline_app = typer.Typer(
    help="Show or change the baseline of a suite (the experiment candidates are compared "
    "against by default).",
    no_args_is_help=True,
)
experiments_app = typer.Typer(help="List stored experiments.", no_args_is_help=True)
traces_app = typer.Typer(help="Browse stored traces.", no_args_is_help=True)

_EXPERIMENT_ARG = typer.Argument(help="Experiment id or name (most recent).", show_default=False)


def _when(trace_or_experiment: Trace | Experiment) -> str:
    moment = (
        trace_or_experiment.start_time
        if isinstance(trace_or_experiment, Trace)
        else trace_or_experiment.created_at
    )
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def _experiment_rows(experiments: list[Experiment]) -> str:
    rows = [
        [
            exp.id,
            exp.name,
            exp.suite,
            _when(exp),
            str(exp.config.get("model") or ""),
            "yes" if exp.is_baseline else "",
        ]
        for exp in experiments
    ]
    return table(["id", "name", "suite", "created (UTC)", "model", "baseline"], rows)


# -- baseline ----------------------------------------------------------------------


@baseline_app.command("set")
def baseline_set(ctx: typer.Context, experiment: Annotated[str, _EXPERIMENT_ARG]) -> None:
    """Mark an experiment as the baseline of its suite (unmarking the previous one)."""
    with open_store(ctx) as store:
        marked = set_baseline(store, experiment)
    typer.echo(f"baseline of suite {marked.suite!r}: {marked.name} ({marked.id})")


@baseline_app.command("unset")
def baseline_unset(ctx: typer.Context, experiment: Annotated[str, _EXPERIMENT_ARG]) -> None:
    """Remove the baseline mark from an experiment."""
    with open_store(ctx) as store:
        unmarked = unset_baseline(store, experiment)
    typer.echo(f"{unmarked.name} ({unmarked.id}) is not a baseline")


@baseline_app.command("show")
def baseline_show(
    ctx: typer.Context,
    suite: Annotated[
        str | None,
        typer.Argument(
            help="Suite name (default: every suite with a baseline).", show_default=False
        ),
    ] = None,
) -> None:
    """Show the baseline of a suite, or of every suite."""
    with open_store(ctx) as store:
        if suite is not None:
            found = get_baseline(store, suite)
            if found is None:
                fail(f"suite {suite!r} has no baseline (set one with `agentlens baseline set`)")
            baselines = [found]
        else:
            baselines = store.list_experiments(is_baseline=True)
    if not baselines:
        typer.echo("no baselines set")
        return
    typer.echo(_experiment_rows(baselines))


# -- experiments -------------------------------------------------------------------


@experiments_app.command("ls")
def experiments_ls(
    ctx: typer.Context,
    suite: Annotated[str | None, typer.Option("--suite", "-s", help="Only this suite.")] = None,
    name: Annotated[str | None, typer.Option("--name", "-n", help="Only this name.")] = None,
    limit: Annotated[int, typer.Option("--limit", "-l", min=1, help="At most this many.")] = 20,
) -> None:
    """List experiments, most recent first."""
    with open_store(ctx) as store:
        experiments = store.list_experiments(name=name, suite=suite)[:limit]
    if not experiments:
        typer.echo("no experiments")
        return
    typer.echo(_experiment_rows(experiments))


# -- traces ------------------------------------------------------------------------


def _experiment_id(store: Store, experiment: str | None) -> str | None:
    return None if experiment is None else find_experiment(store, experiment).id


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


@traces_app.command("ls")
def traces_ls(
    ctx: typer.Context,
    name: Annotated[
        str | None, typer.Option("--name", "-n", help="Only traces with this name.")
    ] = None,
    experiment: Annotated[
        str | None,
        typer.Option("--experiment", "-e", help="Only traces of this experiment (id or name)."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-l", min=1, help="At most this many.")] = 20,
) -> None:
    """List traces, most recent first."""
    with open_store(ctx) as store:
        traces = store.list_traces(
            name=name, experiment_id=_experiment_id(store, experiment), limit=limit
        )
    if not traces:
        typer.echo("no traces")
        return
    rows = [
        [
            trace.id,
            trace.name,
            _when(trace),
            str(trace.metadata.get("case_id") or ""),
            _ms(trace.latency_ms),
            str(trace.n_steps),
            fmt_value(trace.total_cost),
            "error" if trace.error else "ok",
        ]
        for trace in traces
    ]
    headers = ["id", "name", "started (UTC)", "case", "ms", "steps", "cost $", "status"]
    typer.echo(table(headers, rows))


def _preview(value: object, width: int = 100) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def _span_line(span: Span, depth: int) -> str:
    parts = [f"{'  ' * depth}- {span.kind.value} {span.name}", f"{_ms(span.latency_ms)} ms"]
    if span.usage is not None:
        parts.append(f"{span.usage.input_tokens}+{span.usage.output_tokens} tok")
        if span.usage.cost_usd is not None:
            parts.append(f"${fmt_value(span.usage.cost_usd)}")
    if span.error:
        parts.append(f"error: {span.error}")
    return "  ".join(parts)


def _span_tree(trace: Trace) -> list[str]:
    lines: list[str] = []

    def walk(span: Span, depth: int) -> None:
        lines.append(_span_line(span, depth))
        for child in trace.children(span.id):
            walk(child, depth + 1)

    for root in trace.roots():
        walk(root, 1)
    return lines


@traces_app.command("show")
def traces_show(
    ctx: typer.Context,
    trace_id: Annotated[str, typer.Argument(help="Trace id.", show_default=False)],
    as_json: Annotated[bool, typer.Option("--json", help="Print the full trace as JSON.")] = False,
) -> None:
    """Show one trace: input, output, spans and evaluation results."""
    with open_store(ctx) as store:
        trace = store.get_trace(trace_id)
        if trace is None:
            fail(f"no trace with id {trace_id!r}")
        results = store.get_eval_results(trace.id)
    if as_json:
        typer.echo(trace.model_dump_json(indent=2))
        return
    lines = [
        f"trace     {trace.id}",
        f"name      {trace.name}",
        f"started   {_when(trace)} UTC  ({_ms(trace.latency_ms)} ms)",
        f"steps     {trace.n_steps}  tokens {trace.total_tokens}  "
        f"cost ${fmt_value(trace.total_cost)}",
        f"input     {_preview(trace.input)}",
        f"output    {_preview(trace.output)}",
    ]
    if trace.error:
        lines.append(f"error     {trace.error}")
    if trace.metadata:
        lines.append(f"metadata  {_preview(trace.metadata, width=200)}")
    if trace.spans:
        lines += ["spans", *_span_tree(trace)]
    if results:
        lines.append("evaluations")
        lines += [
            f"  - {'pass' if r.passed else 'FAIL'} {r.evaluator} {fmt_value(r.score)}"
            + (f": {r.reason}" if r.reason else "")
            for r in results
        ]
    typer.echo("\n".join(lines))


# -- export / import ---------------------------------------------------------------


def export_traces(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="JSONL file to write.", show_default=False)],
    name: Annotated[
        str | None, typer.Option("--name", "-n", help="Only traces with this name.")
    ] = None,
    experiment: Annotated[
        str | None,
        typer.Option("--experiment", "-e", help="Only traces of this experiment (id or name)."),
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", "-l", min=1, help="Only the most recent N.")
    ] = None,
) -> None:
    """Export traces to a JSONL file (one trace per line, oldest first)."""
    with open_store(ctx) as store:
        count = export_jsonl(
            store, path, name=name, experiment_id=_experiment_id(store, experiment), limit=limit
        )
    typer.echo(f"exported {count} trace{'s' if count != 1 else ''} to {path}")


def import_traces(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="JSONL file to read.", show_default=False)],
    experiment: Annotated[
        str | None,
        typer.Option(
            "--experiment", "-e", help="Link the traces to this stored experiment (id or name)."
        ),
    ] = None,
) -> None:
    """Import traces from a JSONL file (validated first; re-importing is a no-op)."""
    with open_store(ctx, create=True) as store:
        count = import_jsonl(store, path, experiment_id=_experiment_id(store, experiment))
    typer.echo(f"imported {count} trace{'s' if count != 1 else ''} from {path}")
