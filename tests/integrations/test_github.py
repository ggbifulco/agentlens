"""Posting the report as a PR comment: GitHub REST API mocked with respx."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from agentlens.integrations.github import (
    MAX_COMMENT_CHARS,
    CommentResult,
    GitHubContext,
    GitHubError,
    upsert_pr_comment,
    with_marker,
)

API = "https://api.github.com"
COMMENTS = f"{API}/repos/octo/demo/issues/7/comments"
CONTEXT = GitHubContext(repository="octo/demo", pr_number=7, token="ghs_secret")
BODY = "<!-- agentlens-report -->\n### ✅ AgentLens"


def comment(id_: int, body: str) -> dict[str, Any]:
    return {"id": id_, "body": body, "html_url": f"https://github.com/octo/demo/pull/7#c{id_}"}


def sent(route: respx.Route, index: int = -1) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(route.calls[index].request.content)
    return data


# -- upsert --------------------------------------------------------------------------


def test_creates_a_comment_when_none_has_the_marker(respx_mock: respx.Router) -> None:
    listing = respx_mock.get(COMMENTS).respond(json=[comment(1, "LGTM"), {"id": 2}, "junk"])
    create = respx_mock.post(COMMENTS).respond(201, json=comment(3, BODY))
    result = upsert_pr_comment(BODY, context=CONTEXT)
    assert result == CommentResult("created", 3, "https://github.com/octo/demo/pull/7#c3")
    request = listing.calls[0].request
    assert request.url.params["per_page"] == "100"
    assert request.headers["Authorization"] == "Bearer ghs_secret"
    assert request.headers["Accept"] == "application/vnd.github+json"
    assert request.headers["X-GitHub-Api-Version"] == "2022-11-28"
    assert request.headers["User-Agent"].startswith("agentlens/")
    assert sent(create) == {"body": BODY}


def test_updates_the_comment_with_the_marker(respx_mock: respx.Router) -> None:
    respx_mock.get(COMMENTS).respond(
        json=[comment(1, "hi"), comment(2, "<!-- agentlens-report -->\nold"), comment(4, BODY)]
    )
    patch = respx_mock.patch(f"{API}/repos/octo/demo/issues/comments/2").respond(
        json=comment(2, BODY)
    )
    result = upsert_pr_comment(BODY, context=CONTEXT)
    assert result.action == "updated"
    assert result.comment_id == 2  # the oldest one carrying the marker
    assert sent(patch) == {"body": BODY}


def test_unchanged_comment_is_not_patched(respx_mock: respx.Router) -> None:
    respx_mock.get(COMMENTS).respond(json=[comment(5, BODY)])
    result = upsert_pr_comment(BODY, context=CONTEXT)
    assert result == CommentResult("unchanged", 5, "https://github.com/octo/demo/pull/7#c5")
    assert len(respx_mock.calls) == 1


def test_follows_pagination(respx_mock: respx.Router) -> None:
    page2 = f"{API}/repositories/1/issues/7/comments?page=2&per_page=100"
    respx_mock.get(COMMENTS, params={"per_page": "100"}).respond(
        json=[comment(1, "a")], headers={"Link": f'<{page2}>; rel="next", <{page2}>; rel="last"'}
    )
    second = respx_mock.get(page2).respond(json=[comment(9, "<!-- k -->\nold")])
    patch = respx_mock.patch(f"{API}/repos/octo/demo/issues/comments/9").respond(json={"id": 9})
    result = upsert_pr_comment("new", context=CONTEXT, marker="k")
    assert second.called
    assert result == CommentResult("updated", 9, "")
    assert sent(patch) == {"body": "<!-- k -->\nnew"}  # the marker is added to the body


def test_marker_must_match_exactly(respx_mock: respx.Router) -> None:
    respx_mock.get(COMMENTS).respond(json=[comment(1, "<!-- agentlens-report:other -->")])
    create = respx_mock.post(COMMENTS).respond(201, json=comment(2, ""))
    assert upsert_pr_comment(BODY, context=CONTEXT).action == "created"
    assert create.called


def test_long_bodies_are_truncated(respx_mock: respx.Router) -> None:
    respx_mock.get(COMMENTS).respond(json=[])
    create = respx_mock.post(COMMENTS).respond(201, json=comment(1, ""))
    upsert_pr_comment(BODY + "x" * MAX_COMMENT_CHARS, context=CONTEXT)
    body = sent(create)["body"]
    assert len(body) == MAX_COMMENT_CHARS
    assert body.startswith(BODY)
    assert body.endswith("longer than a GitHub comment can hold.")


def test_enterprise_api_url_and_own_client(respx_mock: respx.Router) -> None:
    context = GitHubContext(
        repository="o/r", pr_number=1, token="t", api_url="https://ghe.example/api/v3"
    )
    respx_mock.get("https://ghe.example/api/v3/repos/o/r/issues/1/comments").respond(json=[])
    respx_mock.post("https://ghe.example/api/v3/repos/o/r/issues/1/comments").respond(
        201, json=comment(1, "")
    )
    with httpx.Client() as client:
        upsert_pr_comment("x", context=context, client=client)
        assert not client.is_closed  # a client passed in is left open


@pytest.mark.parametrize(
    ("status", "payload", "message"),
    [
        (403, {"message": "Resource not accessible by integration"}, "read-only token"),
        (401, {"message": "Bad credentials"}, "HTTP 401 Bad credentials"),
        (500, None, "HTTP 500 boom"),
    ],
)
def test_api_errors(
    respx_mock: respx.Router, status: int, payload: dict[str, str] | None, message: str
) -> None:
    response = (
        httpx.Response(status, json=payload) if payload else httpx.Response(status, text="boom")
    )
    respx_mock.get(COMMENTS).respond(json=[])
    respx_mock.post(COMMENTS).mock(return_value=response)
    with pytest.raises(GitHubError, match=message) as info:
        upsert_pr_comment(BODY, context=CONTEXT)
    assert "creating the comment" in str(info.value)
    assert "ghs_secret" not in str(info.value)


def test_listing_must_be_a_list(respx_mock: respx.Router) -> None:
    respx_mock.get(COMMENTS).respond(json={"message": "?"})
    with pytest.raises(GitHubError, match="returned no list"):
        upsert_pr_comment(BODY, context=CONTEXT)


def test_response_without_id(respx_mock: respx.Router) -> None:
    respx_mock.get(COMMENTS).respond(json=[])
    respx_mock.post(COMMENTS).respond(201, json=["?"])
    with pytest.raises(GitHubError, match="no comment id"):
        upsert_pr_comment(BODY, context=CONTEXT)


def test_network_errors(respx_mock: respx.Router) -> None:
    respx_mock.get(COMMENTS).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(GitHubError, match="request failed: refused"):
        upsert_pr_comment(BODY, context=CONTEXT)


def test_with_marker() -> None:
    assert with_marker("body") == "<!-- agentlens-report -->\nbody"
    assert with_marker(BODY) == BODY
    assert with_marker("x", "k") == "<!-- k -->\nx"
    with pytest.raises(ValueError, match="invalid marker"):
        with_marker("x", "bad marker")


# -- context -------------------------------------------------------------------------


def event(tmp_path: Path, data: object) -> str:
    path = tmp_path / "event.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize(
    "data",
    [{"pull_request": {"number": 7}}, {"issue": {"number": 7}}, {"number": 7, "action": "x"}],
)
def test_context_from_env(tmp_path: Path, data: object) -> None:
    env = {
        "GITHUB_TOKEN": "t",
        "GITHUB_REPOSITORY": "octo/demo",
        "GITHUB_EVENT_PATH": event(tmp_path, data),
        "GITHUB_API_URL": "https://ghe.example/api/v3/",
    }
    context = GitHubContext.from_env(env)
    assert context == GitHubContext("octo/demo", 7, "t", "https://ghe.example/api/v3")


def test_context_arguments_win_and_default_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GH_TOKEN", "from-gh")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_API_URL", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "a/b")
    context = GitHubContext.from_env(repository="c/d", pr_number=3)
    assert context == GitHubContext("c/d", 3, "from-gh", API)
    assert GitHubContext.from_env({}, repository="c/d", pr_number=3, token="x").token == "x"


def test_repr_hides_the_token() -> None:
    assert "ghs_secret" not in repr(CONTEXT)
    assert "token=***" in repr(CONTEXT)


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({}, "no GitHub token"),
        ({"GITHUB_TOKEN": "t"}, "no repository"),
        ({"GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "nope"}, "no repository"),
        ({"GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "/x"}, "no repository"),
        ({"GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "a/b"}, "no pull request number"),
        (
            {"GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "a/b", "GITHUB_EVENT_PATH": "/nope.json"},
            "cannot read the GitHub event file",
        ),
    ],
)
def test_context_errors(env: dict[str, str], message: str) -> None:
    with pytest.raises(GitHubError, match=message):
        GitHubContext.from_env(env)


@pytest.mark.parametrize("data", [{"ref": "refs/heads/main"}, [1], {"pull_request": {}}])
def test_event_without_a_pull_request(tmp_path: Path, data: object) -> None:
    env = {
        "GITHUB_TOKEN": "t",
        "GITHUB_REPOSITORY": "a/b",
        "GITHUB_EVENT_PATH": event(tmp_path, data),
    }
    with pytest.raises(GitHubError, match="not about a pull request"):
        GitHubContext.from_env(env)
