"""``agentlens github comment``: the CLI around :func:`upsert_pr_comment` (HTTP mocked)."""

from __future__ import annotations

import json

import pytest
import respx

from agentlens.cli import EXIT_OK, EXIT_USAGE
from tests.cli.conftest import Toy, plain

COMMENTS = "https://api.github.com/repos/octo/demo/issues/7/comments"


@pytest.fixture
def github_env(toy: Toy, monkeypatch: pytest.MonkeyPatch) -> Toy:
    event = toy.write("event.json", json.dumps({"pull_request": {"number": 7}}))
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_secret")
    monkeypatch.setenv("GITHUB_REPOSITORY", "octo/demo")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.delenv("GITHUB_API_URL", raising=False)
    toy.write("report.md", "<!-- agentlens:toy -->\n### report\n")
    return toy


def test_comment_creates(github_env: Toy, respx_mock: respx.Router) -> None:
    respx_mock.get(COMMENTS).respond(json=[])
    create = respx_mock.post(COMMENTS).respond(
        201, json={"id": 3, "html_url": "https://github.com/octo/demo/pull/7#c3"}
    )
    result = github_env("github", "comment", "--file", "report.md", "--marker", "agentlens:toy")
    assert result.exit_code == EXIT_OK, result.output
    assert result.stdout.strip() == (
        "created comment 3 on PR #7 https://github.com/octo/demo/pull/7#c3"
    )
    assert json.loads(create.calls[0].request.content)["body"].startswith("<!-- agentlens:toy -->")


def test_comment_updates_with_explicit_pr_and_repo(
    github_env: Toy, respx_mock: respx.Router
) -> None:
    url = "https://api.github.com/repos/a/b/issues/9/comments"
    respx_mock.get(url).respond(json=[{"id": 4, "body": "<!-- agentlens-report -->\nold"}])
    respx_mock.patch("https://api.github.com/repos/a/b/issues/comments/4").respond(json={"id": 4})
    result = github_env("github", "comment", "-f", "report.md", "--pr", "9", "--repo", "a/b")
    assert result.exit_code == EXIT_OK, result.output
    assert result.stdout.strip() == "updated comment 4 on PR #9"


def test_comment_api_error_is_a_usage_error(github_env: Toy, respx_mock: respx.Router) -> None:
    respx_mock.get(COMMENTS).respond(403, json={"message": "Resource not accessible"})
    result = github_env("github", "comment", "--file", "report.md")
    assert result.exit_code == EXIT_USAGE
    assert "HTTP 403 Resource not accessible" in result.stderr
    assert "ghs_secret" not in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--file", "missing.md"], "missing.md"),
        (["--file", "report.md", "--marker", "bad marker"], "invalid marker"),
        (["--file", "report.md", "--repo", "nope"], "no repository"),
    ],
)
def test_comment_usage_errors(github_env: Toy, args: list[str], message: str) -> None:
    result = github_env("github", "comment", *args)
    assert result.exit_code == EXIT_USAGE
    assert message in plain(result.stderr)


def test_github_help_lists_comment(toy: Toy) -> None:
    result = toy("github", "--help", db=False)
    assert result.exit_code == EXIT_OK
    assert "comment" in result.stdout
