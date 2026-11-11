# AGENTS.md — How work happens in this repo

This project is built incrementally by an AI agent (Claude) in scheduled sessions, twice a day,
and reviewed by the maintainer (@ggbifulco). Every session starts with **no memory**:
everything needed lives in this repo. Read this file fully before doing anything.

## Sources of truth
- `docs/requirements.md` — what we build (FR-*/NFR-* ids)
- `docs/architecture.md`, `docs/design.md`, `docs/adr/` — how we build it
- GitHub issues + labels + milestones — the backlog / board
- `docs/PROGRESS.md` — session log (append-only)

## Board conventions
- Status labels: `status: backlog` → `status: todo` → `status: in-progress` → `status: review` → closed (Done).
  A workflow (`project-sync.yml`) mirrors these onto the GitHub Project board.
- Priority: `P0` (must) > `P1` (should) > `P2` (could). Size: `S` / `M` / `L`.
- Milestones `M0`…`M7` are phases; prefer finishing the lowest open milestone.

## Session procedure (one task per session)
1. **Sync**: `git fetch origin main && git checkout main && git reset --hard origin/main`.
2. **Check feedback first**: read new comments on open/recently merged PRs and issues by the maintainer.
   Requested changes on an open PR take priority over new work. Feedback on merged PRs becomes a new issue.
3. **Pick a task**: if an issue is `status: in-progress` (an interrupted session), resume it.
   Otherwise pick among open `status: todo` issues whose *Depends on* issues are all closed, ordered by
   lowest milestone, then priority (P0 first), then lowest issue number.
   If none are ready, promote ready `status: backlog` issues of the next milestone to `status: todo` and pick again.
4. **Start**: set label `status: in-progress` (remove the old status). Create branch `feat/<issue>-<slug>`
   (`chore/`, `docs/`, `test/` for other types).
5. **Implement** to the acceptance criteria only. Small, logical commits using Conventional Commits
   (`feat(tracing): ...`, `test(store): ...`). Keep the design docs updated if a decision changes
   (add an ADR for any significant new decision).
6. **Verify locally**: `uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest`.
   Never push red code. Never commit secrets; real API keys only via env vars.
7. **PR**: push the branch, open a PR titled like the issue with body: summary, *what & why*,
   *how to review* (files to read first), *what you can learn here* (concepts/patterns used),
   test evidence, and `Closes #<n>`. Set the issue to `status: review`.
8. **Merge policy**: see `MERGE_POLICY` below.
9. **Log**: append an entry to `docs/PROGRESS.md` (on the PR branch) and comment the same summary on the issue.
10. **Report** to the maintainer: issue, PR link, commits (sha + message), what was done, key design
    choices, how to try it, what's next.

`MERGE_POLICY: auto` — after CI is green, squash-merge the PR yourself and delete the branch
(the maintainer reviews afterwards; follow-ups come as comments).
If this is changed to `MERGE_POLICY: approval`, never merge: wait for the maintainer's approval and,
if the next task depends on an unmerged PR, pick an independent task instead.

## Engineering standards
- Python >= 3.11, src layout, full type hints, `mypy --strict` on `src/`.
- Tests next to behaviour in `tests/` mirroring `src/agentlens/`; coverage target 85%.
- CI must pass with **no API keys**: use `MockLLM` / mocked HTTP. Live tests use `@pytest.mark.live`.
- Public API documented with docstrings; examples in README kept runnable.
- Keep dependencies minimal; justify new ones in the PR.
