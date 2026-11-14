"""The ``agentlens`` command: root options plus ``run``, ``compare`` and ``report``.

Data-management commands (``baseline``, ``experiments``, ``traces``,
``export``, ``import``) live in :mod:`agentlens.cli.data`, the web UI
(``ui``) in :mod:`agentlens.cli.ui`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from agentlens import __version__
from agentlens.cli._common import (
    DB_ENV,
    EXIT_GATE_FAILED,
    State,
    auto_format,
    db_exists,
    fail,
    open_store,
    user_errors,
)
from agentlens.cli.data import (
    baseline_app,
    experiments_app,
    export_traces,
    import_traces,
    traces_app,
)
from agentlens.cli.github import github_app
from agentlens.cli.ui import ui
from agentlens.compare import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    GateReport,
    Gates,
    compare_runs,
    evaluate_gates,
    experiment_runs,
    load_gates,
)
from agentlens.report import (
    ComparisonReport,
    ExperimentReport,
    HtmlRenderer,
    MarkdownRenderer,
    Renderer,
    ReportFormat,
    get_renderer,
    load_report,
    render,
)
from agentlens.schema import Experiment
from agentlens.store import DEFAULT_PATH, Store
from agentlens.suite import (
    RunResult,
    find_experiment,
    get_baseline,
    load_suite,
    run_suite,
    set_baseline,
)

__all__ = ["BASELINE_KEYWORD", "app", "main"]

BASELINE_KEYWORD = "baseline"
"""``compare --baseline baseline``: the marked baseline of the candidate's suite."""

app = typer.Typer(
    name="agentlens",
    help=(
        "Regression testing and evaluation for LLM agents: run suites, compare "
        "experiments with confidence intervals, gate CI on thresholds."
    ),
    epilog=(
        "Exit codes: 0 success; 1 a gate failed; 2 usage or configuration error "
        "(bad arguments, invalid suite or gates file, unknown experiment, unreadable file)."
    ),
    no_args_is_help=True,
    add_completion=False,
    # Plain tracebacks: rich's pretty ones can print local variables (API keys).
    pretty_exceptions_enable=False,
)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"agentlens {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    ctx: typer.Context,
    db: Annotated[
        Path,
        typer.Option(
            "--db",
            envvar=DB_ENV,
            help="SQLite database with traces and experiments.",
            show_envvar=True,
        ),
    ] = DEFAULT_PATH,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """Regression testing and evaluation for LLM agents."""
    ctx.obj = State(db=db)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _progress(result: RunResult, done: int, total: int) -> None:
    status = "pass" if result.passed else "FAIL"
    detail = f" ({result.error})" if result.error else ""
    typer.echo(f"[{done}/{total}] {result.case_id} #{result.repetition} {status}{detail}", err=True)


# -- run ---------------------------------------------------------------------------


@app.command()
def run(
    ctx: typer.Context,
    suite: Annotated[Path, typer.Argument(help="Suite YAML file.", show_default=False)],
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Experiment name (default: <suite>-<UTC time>)."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model under test ('provider:name'), overriding the suite's model "
            "and $AGENTLENS_MODEL.",
        ),
    ] = None,
    repetitions: Annotated[
        int | None,
        typer.Option("--repetitions", "-r", min=1, help="Runs per case (overrides the suite)."),
    ] = None,
    concurrency: Annotated[
        int | None,
        typer.Option("--concurrency", "-j", min=1, help="Runs at once (overrides the suite)."),
    ] = None,
    timeout: Annotated[
        float | None,
        typer.Option("--timeout", help="Seconds per agent call (overrides the suite)."),
    ] = None,
    gates: Annotated[
        Path | None,
        typer.Option(
            "--gates",
            "-g",
            help="Gates YAML file; absolute rules (min/max) are checked, delta rules "
            "need `agentlens compare`.",
        ),
    ] = None,
    mark_baseline: Annotated[
        bool,
        typer.Option(
            "--set-baseline",
            help="Mark the new experiment as its suite's baseline (not done if a gate fails).",
        ),
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="No per-run progress on stderr.")
    ] = False,
    json_out: Annotated[
        str | None,
        typer.Option(
            "--json",
            metavar="PATH",
            help="Also write the report (with the gate verdict) as JSON, for `agentlens "
            "report`; '-' prints JSON to stdout instead of text.",
        ),
    ] = None,
) -> None:
    """Run a suite, save it as an experiment and print a summary.

    Every case runs --repetitions times; each run is traced, scored by its
    evaluators and saved under a new experiment in the database. Exits with 1
    if a gate fails.
    """
    if timeout is not None and timeout <= 0:
        raise typer.BadParameter("must be > 0", param_hint="'--timeout'")
    with user_errors():
        loaded_gates = load_gates(gates) if gates is not None else None
        loaded_suite = load_suite(suite)
    with open_store(ctx, create=True) as store:
        result = run_suite(
            loaded_suite,
            store=store,
            name=name,
            model=model,
            repetitions=repetitions,
            concurrency=concurrency,
            timeout_s=timeout,
            progress=None if quiet else _progress,
        )
        gate_report = evaluate_gates(loaded_gates, run=result) if loaded_gates else None
        experiment = result.experiment
        marked = False
        if mark_baseline and (gate_report is None or gate_report.passed):
            experiment = set_baseline(store, experiment)
            marked = True
    report = ExperimentReport.from_runs(experiment, result, gates=gate_report)
    if json_out == "-":
        typer.echo(render(report, ReportFormat.JSON))
    else:
        typer.echo(render(report, auto_format()))
        typer.echo("")
        if json_out is not None:
            with user_errors():
                _write(Path(json_out), render(report, ReportFormat.JSON))
    # Status lines go to stderr when stdout carries JSON.
    to_err = json_out == "-"
    typer.echo(f"saved experiment {experiment.name!r} to {store.path}", err=to_err)
    if marked:
        typer.echo(f"marked as the baseline of suite {experiment.suite!r}", err=to_err)
    elif mark_baseline:
        typer.echo("not marked as the baseline: a gate failed", err=True)
    _exit_on_gates(gate_report)


def _exit_on_gates(report: GateReport | None) -> None:
    if report is not None and not report.passed:
        raise typer.Exit(EXIT_GATE_FAILED)


# -- compare -----------------------------------------------------------------------


def _resolve_baseline(store: Store, baseline: str, candidate: Experiment) -> Experiment:
    if baseline == BASELINE_KEYWORD:
        found = get_baseline(store, candidate.suite)
        if found is None:
            fail(
                f"suite {candidate.suite!r} has no baseline: mark one with "
                "`agentlens baseline set EXPERIMENT` or pass --baseline NAME|ID"
            )
        return found
    return find_experiment(store, baseline)


@app.command()
def compare(
    ctx: typer.Context,
    candidate: Annotated[
        str,
        typer.Option(
            "--candidate",
            "-c",
            help="Experiment under test: id or name (most recent).",
            show_default=False,
        ),
    ],
    baseline: Annotated[
        str,
        typer.Option(
            "--baseline",
            "-b",
            help="Reference experiment: id, name, or 'baseline' for the marked baseline "
            "of the candidate's suite.",
        ),
    ] = BASELINE_KEYWORD,
    gates: Annotated[
        Path | None, typer.Option("--gates", "-g", help="Gates YAML file to check.")
    ] = None,
    resamples: Annotated[
        int, typer.Option("--resamples", min=1, help="Bootstrap resamples per metric.")
    ] = DEFAULT_RESAMPLES,
    seed: Annotated[int, typer.Option("--seed", help="Bootstrap seed (reproducible CIs).")] = 0,
    confidence: Annotated[
        float,
        typer.Option("--confidence", help="Confidence level of the intervals, in (0, 1)."),
    ] = DEFAULT_CONFIDENCE,
    json_out: Annotated[
        str | None,
        typer.Option(
            "--json",
            metavar="PATH",
            help="Also write the comparison (and gate verdict) as JSON; '-' prints JSON "
            "to stdout instead of text.",
        ),
    ] = None,
    html_out: Annotated[
        Path | None,
        typer.Option(
            "--html",
            metavar="PATH",
            help="Also write a self-contained HTML report with both experiments' traces.",
        ),
    ] = None,
) -> None:
    """Compare a candidate experiment against a baseline.

    Cases are paired by id; every metric (pass rate, score per evaluator, cost,
    latency, steps) gets its change with a paired-bootstrap confidence
    interval. Exits with 1 if a gate fails.
    """
    if not 0.0 < confidence < 1.0:
        raise typer.BadParameter("must be between 0 and 1", param_hint="'--confidence'")
    with user_errors():
        loaded_gates: Gates | None = load_gates(gates) if gates is not None else None
    with open_store(ctx) as store:
        cand = find_experiment(store, candidate)
        base = _resolve_baseline(store, baseline, cand)
        cand_runs = experiment_runs(store, cand)
        base_runs = experiment_runs(store, base)
        comparison = compare_runs(
            base_runs,
            cand_runs,
            resamples=resamples,
            seed=seed,
            confidence=confidence,
            baseline_experiment=base,
            candidate_experiment=cand,
        )
        gate_report = (
            evaluate_gates(loaded_gates, comparison=comparison, run=cand_runs)
            if loaded_gates
            else None
        )
    report = ComparisonReport(comparison=comparison, gates=gate_report)
    if json_out == "-":
        typer.echo(render(report, ReportFormat.JSON))
    else:
        typer.echo(render(report, auto_format()))
        if json_out is not None:
            with user_errors():
                _write(Path(json_out), render(report, ReportFormat.JSON))
    if html_out is not None:
        with user_errors():
            _write(
                html_out, HtmlRenderer(runs=cand_runs, baseline_runs=base_runs).comparison(report)
            )
        typer.echo(f"wrote {html_out}", err=True)
    _exit_on_gates(gate_report)


# -- report ------------------------------------------------------------------------


@app.command()
def report(
    ctx: typer.Context,
    target: Annotated[
        str,
        typer.Argument(
            help="Experiment id or name, or a JSON file from `compare --json` or "
            "`report --format json`.",
            show_default=False,
        ),
    ],
    fmt: Annotated[
        ReportFormat | None,
        typer.Option(
            "--format",
            "-f",
            case_sensitive=False,
            help="Output format [default: rich on a terminal, text when piped, written to "
            "a file or with $NO_COLOR]. markdown is GitHub-flavoured, for PR comments; html "
            "is a self-contained page with the traces (from the database).",
            show_default=False,
        ),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write to this file instead of stdout.")
    ] = None,
    marker: Annotated[
        str | None,
        typer.Option(
            "--marker",
            help="markdown only: key of the hidden <!-- KEY --> first line that "
            "`agentlens github comment` finds to update its comment [default: agentlens-report].",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Report on an experiment, or re-render a saved comparison.

    An experiment report has its totals, metrics (mean over cases) and a table
    of cases with the evaluators that failed.
    """
    if fmt is None:
        fmt = auto_format() if output is None else ReportFormat.TEXT
    if marker is not None and fmt is not ReportFormat.MARKDOWN:
        raise typer.BadParameter("only applies to --format markdown", param_hint="'--marker'")
    with user_errors():
        # Fail before any work if the format is not available or the marker is invalid.
        renderer: Renderer = get_renderer(fmt)
        if marker is not None:
            try:
                renderer = MarkdownRenderer(marker=marker)
            except ValueError as exc:
                raise typer.BadParameter(str(exc), param_hint="'--marker'") from None
        path = Path(target)
        loaded: ExperimentReport | ComparisonReport
        if path.is_file():
            loaded = load_report(path)
            if fmt is ReportFormat.HTML:
                renderer = _html_renderer_for(ctx, loaded)
        elif path.suffix.lower() == ".json":
            fail(f"no such file: {target}")
        else:
            with open_store(ctx) as store:
                experiment = find_experiment(store, target)
                runs = experiment_runs(store, experiment)
            loaded = ExperimentReport.from_runs(experiment, runs)
            if fmt is ReportFormat.HTML:
                renderer = HtmlRenderer(runs=runs)
        if isinstance(loaded, ComparisonReport):
            text = renderer.comparison(loaded)
        else:
            text = renderer.experiment(loaded)
        if output is None:
            typer.echo(text)
        else:
            _write(output, text)
            typer.echo(f"wrote {output}", err=True)


def _html_renderer_for(
    ctx: typer.Context, report: ExperimentReport | ComparisonReport
) -> HtmlRenderer:
    """An HTML renderer with the traces of the report's experiments, if they are in the database.

    A report file may come from another machine (a CI artifact), so a missing
    database or experiment is not an error: the page shows the aggregates only.
    """
    if isinstance(report, ComparisonReport):
        roles = {"baseline": report.comparison.baseline, "candidate": report.comparison.candidate}
    else:
        roles = {"candidate": report.experiment}
    found: dict[str, list[RunResult]] = {}
    missing = [role for role, exp in roles.items() if exp is None]
    if db_exists(ctx):
        with open_store(ctx) as store:
            for role, exp in roles.items():
                if exp is None:
                    continue
                if store.get_experiment(exp.id) is None:
                    missing.append(role)
                else:
                    found[role] = experiment_runs(store, exp.id)
    else:
        missing = list(roles)
    if missing:
        typer.echo(
            "note: no traces for the "
            + " and ".join(sorted(missing))
            + " in the database; the HTML report shows the aggregates only",
            err=True,
        )
    return HtmlRenderer(runs=found.get("candidate"), baseline_runs=found.get("baseline"))


# Registered after run/compare/report so `agentlens --help` lists those first.
app.add_typer(baseline_app, name="baseline")
app.add_typer(experiments_app, name="experiments")
app.add_typer(traces_app, name="traces")
app.command("export")(export_traces)
app.command("import")(import_traces)
app.add_typer(github_app, name="github")
app.command("ui")(ui)


def main() -> None:
    """Console-script entry point (``agentlens``)."""
    app()
