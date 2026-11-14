"""Post a report as a pull-request comment, updating it in place (FR-C3).

The GitHub Action (``action.yml``) calls ``agentlens github comment --file
report.md``, which runs :func:`upsert_pr_comment`:

1. list the PR's comments (``GET /repos/{owner}/{repo}/issues/{n}/comments``,
   every page);
2. if one contains the hidden marker (``<!-- agentlens-report -->``), edit it
   (``PATCH /repos/{owner}/{repo}/issues/comments/{id}``), unless its body is
   already the same;
3. otherwise create one (``POST .../issues/{n}/comments``).

So a PR gets one AgentLens comment that follows its latest push instead of a
new comment per push. :class:`GitHubContext.from_env` reads what a workflow
provides: ``GITHUB_TOKEN`` (pass ``secrets.GITHUB_TOKEN`` or the action's
``github-token``), ``GITHUB_REPOSITORY`` (``owner/repo``), ``GITHUB_EVENT_PATH``
(the event JSON, for the PR number) and ``GITHUB_API_URL`` (GitHub Enterprise).

The token needs ``pull-requests: write`` (or ``issues: write``). Pull requests
from forks get a read-only token on the ``pull_request`` event, so posting
fails there with a :class:`GitHubError` saying so.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from agentlens import __version__
from agentlens.report.markdown import DEFAULT_MARKER, marker_comment

__all__ = [
    "DEFAULT_API_URL",
    "MAX_COMMENT_CHARS",
    "CommentResult",
    "GitHubContext",
    "GitHubError",
    "upsert_pr_comment",
    "with_marker",
]

DEFAULT_API_URL = "https://api.github.com"

MAX_COMMENT_CHARS = 65_536
"""GitHub rejects longer comment bodies; longer reports are truncated with a note."""

_TIMEOUT = httpx.Timeout(30.0)
_TRUNCATED = "\n\n…report truncated: it is longer than a GitHub comment can hold."


class GitHubError(RuntimeError):
    """Missing configuration, or the GitHub API refused or failed a request."""


@dataclass(frozen=True)
class CommentResult:
    """What :func:`upsert_pr_comment` did."""

    action: Literal["created", "updated", "unchanged"]
    comment_id: int
    url: str
    """The comment's ``html_url`` (empty if the API did not return one)."""


@dataclass(frozen=True)
class GitHubContext:
    """Where to comment and with which token. ``repr`` hides the token."""

    repository: str
    """``owner/repo``."""
    pr_number: int
    token: str
    api_url: str = DEFAULT_API_URL

    def __repr__(self) -> str:
        return (
            f"GitHubContext(repository={self.repository!r}, pr_number={self.pr_number}, "
            f"api_url={self.api_url!r}, token=***)"
        )

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        repository: str | None = None,
        pr_number: int | None = None,
        token: str | None = None,
    ) -> GitHubContext:
        """The context of the running workflow; explicit arguments win over ``env``.

        ``env`` defaults to :data:`os.environ`. The PR number comes from the
        event file: ``pull_request.number`` (``pull_request`` /
        ``pull_request_target`` events), ``issue.number`` (comments) or
        ``number``.

        Raises:
            GitHubError: if the token, the repository or the PR number is missing.
        """
        env = os.environ if env is None else env
        token = token or env.get("GITHUB_TOKEN") or env.get("GH_TOKEN")
        if not token:
            raise GitHubError("no GitHub token: set GITHUB_TOKEN (e.g. to secrets.GITHUB_TOKEN)")
        repository = repository or env.get("GITHUB_REPOSITORY")
        if not repository or repository.count("/") != 1 or repository.startswith("/"):
            raise GitHubError(
                "no repository: set GITHUB_REPOSITORY or pass --repo OWNER/REPO "
                f"(got {repository!r})"
            )
        if pr_number is None:
            pr_number = _event_pr_number(env.get("GITHUB_EVENT_PATH"))
        api_url = (env.get("GITHUB_API_URL") or DEFAULT_API_URL).rstrip("/")
        return cls(repository=repository, pr_number=pr_number, token=token, api_url=api_url)


def _event_pr_number(path: str | None) -> int:
    if not path:
        raise GitHubError("no pull request number: pass --pr N or set GITHUB_EVENT_PATH")
    try:
        event = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GitHubError(f"cannot read the GitHub event file {path}: {exc}") from exc
    if isinstance(event, dict):
        for key in ("pull_request", "issue"):
            item = event.get(key)
            if isinstance(item, dict) and isinstance(item.get("number"), int):
                return int(item["number"])
        if isinstance(event.get("number"), int):
            return int(event["number"])
    raise GitHubError(
        f"the GitHub event in {path} is not about a pull request: run on `pull_request` "
        "or pass --pr N"
    )


def with_marker(body: str, marker: str = DEFAULT_MARKER) -> str:
    """``body`` starting with ``<!-- marker -->`` (added only if it is not in the body)."""
    tag = marker_comment(marker)
    return body if tag in body else f"{tag}\n{body}"


def _truncate(body: str) -> str:
    if len(body) <= MAX_COMMENT_CHARS:
        return body
    return body[: MAX_COMMENT_CHARS - len(_TRUNCATED)] + _TRUNCATED


def _check(response: httpx.Response, what: str) -> Any:
    if response.is_success:
        return response.json()
    try:
        detail = response.json().get("message", "")
    except (ValueError, AttributeError):
        detail = response.text[:200]
    hint = ""
    if response.status_code in (401, 403, 404):
        hint = (
            " (the token needs `pull-requests: write`; pull requests from forks get a "
            "read-only token on the `pull_request` event)"
        )
    raise GitHubError(f"GitHub API: {what} failed: HTTP {response.status_code} {detail}{hint}")


def _comments(client: httpx.Client, url: str, headers: dict[str, str]) -> Iterator[dict[str, Any]]:
    next_url: str | None = url
    params: dict[str, int] | None = {"per_page": 100}
    while next_url:
        response = client.get(next_url, params=params, headers=headers)
        page = _check(response, "listing the pull request comments")
        if not isinstance(page, list):
            raise GitHubError("GitHub API: listing the pull request comments returned no list")
        yield from (item for item in page if isinstance(item, dict))
        next_url = response.links.get("next", {}).get("url")
        params = None  # the next link carries them


def _result(action: Literal["created", "updated", "unchanged"], data: Any) -> CommentResult:
    if not isinstance(data, dict) or not isinstance(data.get("id"), int):
        raise GitHubError("GitHub API: unexpected response (no comment id)")
    url = data.get("html_url")
    html_url = url if isinstance(url, str) else ""
    return CommentResult(action=action, comment_id=data["id"], url=html_url)


def upsert_pr_comment(
    body: str,
    *,
    context: GitHubContext,
    marker: str = DEFAULT_MARKER,
    client: httpx.Client | None = None,
) -> CommentResult:
    """Create the PR's report comment, or update the one carrying ``marker``.

    Args:
        body: The comment (e.g. ``render(report, "markdown")``). The marker is
            added if missing; a body longer than :data:`MAX_COMMENT_CHARS` is
            truncated.
        context: Repository, PR number and token (:meth:`GitHubContext.from_env`).
        marker: Key of the hidden ``<!-- marker -->`` identifying the comment;
            use one per report when a PR gets several.
        client: An ``httpx.Client`` to use (tests, proxies); by default one is
            created and closed.

    Returns:
        Whether the comment was ``created``, ``updated`` or already ``unchanged``,
        with its id and URL. If several comments carry the marker, the oldest
        is updated.

    Raises:
        GitHubError: if a request fails or GitHub answers with an error.
        ValueError: if ``marker`` is not a valid key.
    """
    text = _truncate(with_marker(body, marker))
    tag = marker_comment(marker)
    base = f"{context.api_url}/repos/{context.repository}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {context.token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": f"agentlens/{__version__}",
    }
    own = client is None
    http = httpx.Client(timeout=_TIMEOUT) if client is None else client
    try:
        comments_url = f"{base}/issues/{context.pr_number}/comments"
        comments = _comments(http, comments_url, headers)
        existing = next((c for c in comments if tag in str(c.get("body") or "")), None)
        if existing is None:
            created = http.post(comments_url, json={"body": text}, headers=headers)
            return _result("created", _check(created, "creating the comment"))
        if existing.get("body") == text:
            return _result("unchanged", existing)
        updated = http.patch(
            f"{base}/issues/comments/{existing['id']}", json={"body": text}, headers=headers
        )
        return _result("updated", _check(updated, "updating the comment"))
    except httpx.HTTPError as exc:
        raise GitHubError(f"GitHub API request failed: {exc}") from exc
    finally:
        if own:
            http.close()
