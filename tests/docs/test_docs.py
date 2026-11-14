"""The docs site and the README stay consistent with the code (issue #33).

- every page in the ``mkdocs.yml`` nav exists (``mkdocs build --strict`` runs in CI too, but
  only with the ``docs`` dependency group; this check runs with the normal test suite);
- the README's images exist and the demo GIF stays small;
- the README's "Your own agent in three files" example works when copied as is;
- the MkDocs hook that tidies docstrings for the API reference.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from agentlens.cli import app
from agentlens.compare import EXIT_GATE_FAILED, EXIT_OK

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
README = (ROOT / "README.md").read_text(encoding="utf-8")


class _MkDocsLoader(yaml.SafeLoader):
    """Safe loader that accepts the ``!!python/name:`` tags of ``mkdocs.yml``."""


_MkDocsLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/name:", lambda loader, suffix, node: suffix
)


def _nav_pages(node: Any) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from _nav_pages(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _nav_pages(value)


def _load_hooks() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mkdocs_hooks", ROOT / "scripts/mkdocs_hooks.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mkdocs_config_points_at_existing_pages() -> None:
    config = yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), Loader=_MkDocsLoader)
    assert config["site_url"] == "https://ggbifulco.github.io/agentlens/"
    assert config["strict"] is True
    pages = list(_nav_pages(config["nav"]))
    assert "index.md" in pages
    assert "getting-started.md" in pages
    missing = [page for page in pages if not (DOCS / page).is_file()]
    assert missing == []
    for hook in config["hooks"]:
        assert (ROOT / hook).is_file()


def test_every_doc_page_is_in_the_nav() -> None:
    config = yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), Loader=_MkDocsLoader)
    pages = set(_nav_pages(config["nav"]))
    on_disk = {path.relative_to(DOCS).as_posix() for path in DOCS.rglob("*.md")}
    assert on_disk - pages == set()


def test_readme_images_exist_and_the_demo_gif_is_small() -> None:
    images = re.findall(r"!\[[^\]]*\]\((docs/images/[^)]+)\)", README)
    assert "docs/images/demo.gif" in images
    for image in images:
        assert (ROOT / image).is_file(), image
    assert (ROOT / "docs/images/demo.gif").stat().st_size < 2 * 1024 * 1024


def test_readme_quickstart_refers_to_real_files() -> None:
    quickstart = README.split("## Quickstart", 1)[1].split("\n## ", 1)[0]
    paths = set(re.findall(r"examples/[\w/.-]+\.yaml", quickstart))
    assert paths == {"examples/support_agent/suite.yaml", "examples/support_agent/gates.yaml"}
    for path in paths:
        assert (ROOT / path).is_file()
    assert "SUPPORT_PROMPT_VERSION=v2" in quickstart  # the regression the e2e test checks


def _readme_block(first_line: str) -> str:
    """The README code block whose first line is ``first_line``."""
    for block in re.findall(r"```\w*\n(.*?)```", README, flags=re.DOTALL):
        if block.startswith(first_line):
            return str(block)
    raise AssertionError(f"no README block starts with {first_line!r}")


@pytest.fixture
def own_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENTLENS_DB", raising=False)
    monkeypatch.delenv("AGENTLENS_MODEL", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name, first_line in [
        ("my_agent.py", "# my_agent.py"),
        ("suite.yaml", "# suite.yaml"),
        ("gates.yaml", "# gates.yaml"),
    ]:
        (tmp_path / name).write_text(_readme_block(first_line), encoding="utf-8")
    yield tmp_path
    sys.modules.pop("my_agent", None)


def test_readme_own_agent_example_runs(own_agent: Path) -> None:
    runner = CliRunner()
    db = ["--db", str(own_agent / "runs.db")]
    first = runner.invoke(app, [*db, "run", "suite.yaml", "--set-baseline", "-q"])
    assert first.exit_code == EXIT_OK, first.output
    second = runner.invoke(app, [*db, "run", "suite.yaml", "--name", "my-change", "-q"])
    assert second.exit_code == EXIT_OK, second.output
    compared = runner.invoke(
        app, [*db, "compare", "--candidate", "my-change", "--gates", "gates.yaml"]
    )
    assert compared.exit_code == EXIT_OK, compared.output
    assert "gates passed" in compared.output


def test_readme_own_agent_gates_fail_on_a_regression(own_agent: Path) -> None:
    runner = CliRunner()
    db = ["--db", str(own_agent / "runs.db")]
    assert runner.invoke(app, [*db, "run", "suite.yaml", "--set-baseline", "-q"]).exit_code == 0
    agent = own_agent / "my_agent.py"
    agent.write_text(agent.read_text().replace(".content", ".content.upper()"), encoding="utf-8")
    sys.modules.pop("my_agent", None)
    assert runner.invoke(app, [*db, "run", "suite.yaml", "--name", "bad", "-q"]).exit_code == 0
    compared = runner.invoke(app, [*db, "compare", "--candidate", "bad", "--gates", "gates.yaml"])
    assert compared.exit_code == EXIT_GATE_FAILED, compared.output


def test_hook_tidies_sphinx_roles_on_api_pages_only() -> None:
    hooks = _load_hooks()
    html = (
        "<p>See :func:<code>compare_runs</code> and :class:<code>~agentlens.suite.Suite</code>"
        " or :meth:<code>Trace.model_dump</code>::</p>"
    )
    tidy = (
        "<p>See <code>compare_runs</code> and <code>Suite</code>"
        " or <code>Trace.model_dump</code>:</p>"
    )

    class _File:
        def __init__(self, src_uri: str) -> None:
            self.src_uri = src_uri

    class _Page:
        def __init__(self, src_uri: str) -> None:
            self.file = _File(src_uri)

    assert hooks.on_page_content(html, page=_Page("api/compare.md")) == tidy
    assert hooks.on_page_content(html, page=_Page("concepts/tracing.md")) == html
