"""Sync `status: *` issue labels to the Status field of a GitHub Projects (v2) board.

Env:
  PROJECT_TOKEN   classic PAT with `project` + `repo` scopes (or fine-grained with Projects r/w)
  PROJECT_OWNER   user login that owns the project
  PROJECT_NUMBER  project number (from the project URL)
  GITHUB_REPOSITORY, GITHUB_EVENT_NAME, GITHUB_EVENT_PATH  (set by Actions)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

API = "https://api.github.com"

# label suffix / "closed" -> accepted option names on the board (normalised: lowercase, no spaces/dashes)
ALIASES: dict[str, list[str]] = {
    "backlog": ["backlog", "todo"],
    "todo": ["ready", "todo", "nextup"],
    "in-progress": ["inprogress", "doing"],
    "review": ["inreview", "review"],
    "done": ["done", "closed", "completed"],
}


def norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def request(method: str, url: str, token: str, body: dict | None = None) -> object:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (fixed https host)
        return json.loads(resp.read() or b"null")


def graphql(token: str, query: str, **variables: object) -> dict:
    out = request("POST", f"{API}/graphql", token, {"query": query, "variables": variables})
    assert isinstance(out, dict)
    if out.get("errors"):
        raise RuntimeError(out["errors"])
    return out["data"]


def project_info(token: str, owner: str, number: int) -> tuple[str, str, dict[str, str]]:
    q = """query($owner:String!,$number:Int!){ user(login:$owner){ projectV2(number:$number){ id
      field(name:"Status"){ ... on ProjectV2SingleSelectField { id options { id name } } } } } }"""
    p = graphql(token, q, owner=owner, number=number)["user"]["projectV2"]
    field = p["field"]
    return p["id"], field["id"], {norm(o["name"]): o["id"] for o in field["options"]}


def desired_status(issue: dict) -> str | None:
    if issue.get("state") == "closed":
        return "done"
    for label in issue.get("labels", []):
        name = label["name"] if isinstance(label, dict) else label
        if name.startswith("status: "):
            return name.removeprefix("status: ")
    return None


def sync(token: str, project: tuple[str, str, dict[str, str]], issue: dict) -> None:
    project_id, field_id, options = project
    status = desired_status(issue)
    if status is None:
        return
    option_id = next((options[a] for a in ALIASES.get(status, [norm(status)]) if a in options), None)
    if option_id is None:
        print(f"::warning::No board column matches status '{status}' (columns: {list(options)})")
        return
    add = """mutation($p:ID!,$c:ID!){ addProjectV2ItemById(input:{projectId:$p,contentId:$c}){ item { id } } }"""
    item_id = graphql(token, add, p=project_id, c=issue["node_id"])["addProjectV2ItemById"]["item"]["id"]
    upd = """mutation($p:ID!,$i:ID!,$f:ID!,$o:String!){ updateProjectV2ItemFieldValue(input:{
      projectId:$p,itemId:$i,fieldId:$f,value:{singleSelectOptionId:$o}}){ projectV2Item { id } } }"""
    graphql(token, upd, p=project_id, i=item_id, f=field_id, o=option_id)
    print(f"#{issue['number']} -> {status}")


def main() -> int:
    token = os.environ.get("PROJECT_TOKEN", "")
    owner = os.environ.get("PROJECT_OWNER", "")
    number = os.environ.get("PROJECT_NUMBER", "")
    if not (token and owner and number):
        print("::notice::PROJECT_TOKEN / PROJECT_NUMBER not configured, skipping board sync.")
        return 0
    project = project_info(token, owner, int(number))
    event = os.environ["GITHUB_EVENT_NAME"]
    if event == "issues":
        with open(os.environ["GITHUB_EVENT_PATH"]) as f:
            sync(token, project, json.load(f)["issue"])
    else:  # workflow_dispatch / schedule: resync every issue
        repo = os.environ["GITHUB_REPOSITORY"]
        page = 1
        while True:
            issues = request("GET", f"{API}/repos/{repo}/issues?state=all&per_page=100&page={page}", token)
            assert isinstance(issues, list)
            if not issues:
                break
            for issue in issues:
                if "pull_request" not in issue:
                    sync(token, project, issue)
            page += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
