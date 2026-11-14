"""``agentlens github``: GitHub helpers used by the AgentLens action (``action.yml``)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from agentlens.cli._common import fail, user_errors
from agentlens.integrations.github import GitHubContext, GitHubError, upsert_pr_comment
from agentlens.report import DEFAULT_MARKER

__all__ = ["github_app"]

github_app = typer.Typer(
    help="GitHub helpers: post the report as a pull-request comment.", no_args_is_help=True
)


@github_app.command("comment")
def comment(
    file: Annotated[
        Path,
        typer.Option(
            "--file",
            "-f",
            help="Markdown to post, e.g. from `agentlens report X --format markdown -o FILE`.",
            show_default=False,
        ),
    ],
    marker: Annotated[
        str,
        typer.Option(
            "--marker",
            help="Key of the hidden <!-- KEY --> marker: the PR comment carrying it is updated "
            "instead of posting a new one (added to the body if missing).",
        ),
    ] = DEFAULT_MARKER,
    pr: Annotated[
        int | None,
        typer.Option("--pr", min=1, help="Pull request number [default: from $GITHUB_EVENT_PATH]."),
    ] = None,
    repo: Annotated[
        str | None,
        typer.Option("--repo", help="OWNER/REPO [default: $GITHUB_REPOSITORY]."),
    ] = None,
) -> None:
    """Create or update the AgentLens comment on a pull request.

    Needs a token in $GITHUB_TOKEN with `pull-requests: write`. Comments of
    the PR are searched for the marker; the first match is edited, otherwise
    a new comment is posted. Exits with 2 if posting fails.
    """
    with user_errors():
        body = file.read_text(encoding="utf-8")
    try:
        context = GitHubContext.from_env(repository=repo, pr_number=pr)
        result = upsert_pr_comment(body, context=context, marker=marker)
    except (GitHubError, ValueError) as exc:
        fail(str(exc))
    where = f" {result.url}" if result.url else ""
    typer.echo(f"{result.action} comment {result.comment_id} on PR #{context.pr_number}{where}")
