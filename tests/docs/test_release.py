"""Release tooling stays consistent (issue #34).

- ``scripts/release.py``: tag/version check and CHANGELOG section extraction used by
  ``.github/workflows/release.yml``;
- ``CHANGELOG.md`` has a section for the current version, is in the docs site, and the
  packaging metadata points at it;
- the release workflow uses trusted publishing with a pinned publish action.
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path
from types import ModuleType

import pytest
import yaml

import agentlens

ROOT = Path(__file__).resolve().parents[2]
CHANGELOG_TEXT = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def _load_release() -> ModuleType:
    spec = importlib.util.spec_from_file_location("release", ROOT / "scripts/release.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = _load_release()

SAMPLE = """# Changelog

## [Unreleased]

## [1.1.0] - 2027-01-02

### Fixed
- Something.

## [1.0.0] - 2027-01-01

### Added
- Everything.

[Unreleased]: https://example.com/compare/v1.1.0...HEAD
[1.1.0]: https://example.com/v1.1.0
[1.0.0]: https://example.com/v1.0.0
"""


def test_changelog_section_extracts_the_body_without_heading_or_link_definitions() -> None:
    assert release.changelog_section(SAMPLE, "1.1.0") == "### Fixed\n- Something.\n"
    assert release.changelog_section(SAMPLE, "1.0.0") == "### Added\n- Everything.\n"


def test_changelog_section_errors() -> None:
    with pytest.raises(release.ReleaseError, match=r"no '## \[2.0.0\]' section"):
        release.changelog_section(SAMPLE, "2.0.0")
    with pytest.raises(release.ReleaseError, match="empty"):
        release.changelog_section(SAMPLE, "Unreleased")
    with pytest.raises(release.ReleaseError, match="empty"):
        release.changelog_section("## [0.1.0]", "0.1.0")


@pytest.mark.parametrize(
    ("tag", "version"), [("v0.1.0", "0.1.0"), ("v1.2.3rc1", "1.2.3rc1"), (" v10.0.0\n", "10.0.0")]
)
def test_tag_version(tag: str, version: str) -> None:
    assert release.tag_version(tag) == version


@pytest.mark.parametrize("tag", ["0.1.0", "v0.1", "v0.1.0-beta", "release-0.1.0", "v0.1.0.1"])
def test_tag_version_rejects_other_tags(tag: str) -> None:
    with pytest.raises(release.ReleaseError, match="not of the form"):
        release.tag_version(tag)


def _project(tmp_path: Path, version: str = "1.1.0") -> tuple[Path, Path]:
    init = tmp_path / "__init__.py"
    init.write_text(f'"""Doc."""\n\n__version__ = "{version}"\n', encoding="utf-8")
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(SAMPLE, encoding="utf-8")
    return init, changelog


def test_check_tag(tmp_path: Path) -> None:
    init, changelog = _project(tmp_path)
    assert release.check_tag("v1.1.0", init=init, changelog=changelog) == "1.1.0"
    with pytest.raises(release.ReleaseError, match=r"does not match .*'1\.1\.0'"):
        release.check_tag("v1.0.1", init=init, changelog=changelog)
    init, changelog = _project(tmp_path, version="2.0.0")
    with pytest.raises(release.ReleaseError, match="section"):
        release.check_tag("v2.0.0", init=init, changelog=changelog)
    init.write_text("VERSION = 1\n", encoding="utf-8")
    with pytest.raises(release.ReleaseError, match="no __version__"):
        release.check_tag("v2.0.0", init=init, changelog=changelog)


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    init, changelog = _project(tmp_path)
    paths = ["--changelog", str(changelog)]
    assert release.main(["check-tag", "v1.1.0", *paths, "--init", str(init)]) == 0
    assert "matches agentlens 1.1.0" in capsys.readouterr().out
    assert release.main(["check-tag", "v9.9.9", *paths, "--init", str(init)]) == 1
    assert "error: tag v9.9.9 does not match" in capsys.readouterr().err
    assert release.main(["notes", "1.0.0", *paths]) == 0
    assert capsys.readouterr().out == "### Added\n- Everything.\n"
    out = tmp_path / "notes.md"
    assert release.main(["notes", "1.1.0", *paths, "-o", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == "### Fixed\n- Something.\n"
    assert release.main(["notes", "1.1.0", "--changelog", str(tmp_path / "missing.md")]) == 1
    assert "error:" in capsys.readouterr().err


def test_the_current_version_is_released_from_this_changelog() -> None:
    assert release.package_version() == agentlens.__version__
    notes = release.changelog_section(CHANGELOG_TEXT, agentlens.__version__)
    assert "### Added" in notes
    assert release.check_tag(f"v{agentlens.__version__}") == agentlens.__version__
    heading = re.search(
        rf"^## \[{re.escape(agentlens.__version__)}\] - (\S+)$", CHANGELOG_TEXT, re.M
    )
    assert heading is not None
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", heading.group(1))
    assert "## [Unreleased]" in CHANGELOG_TEXT
    assert f"[{agentlens.__version__}]: https://github.com/ggbifulco/agentlens/" in CHANGELOG_TEXT


def test_packaging_metadata_is_release_ready() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["name"] == "agentlens-toolkit"
    assert set(project["dynamic"]) == {"version", "readme"}
    assert project["license"] == "Apache-2.0"
    assert project["urls"]["Documentation"] == "https://ggbifulco.github.io/agentlens/"
    assert project["urls"]["Changelog"].endswith("/CHANGELOG.md")
    assert "otel" in project["optional-dependencies"]


def test_changelog_is_in_the_docs_site() -> None:
    page = (ROOT / "docs/changelog.md").read_text(encoding="utf-8")
    assert '--8<-- "CHANGELOG.md"' in page
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "Changelog: changelog.md" in nav
    assert "Releasing: releasing.md" in nav


def test_release_workflow_uses_trusted_publishing() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    triggers = workflow[True]  # YAML 1.1 reads the `on:` key as a boolean
    assert triggers["push"]["tags"] == ["v*.*.*"]
    publish = workflow["jobs"]["publish"]
    assert publish["environment"]["name"] == "pypi"
    assert publish["permissions"] == {"id-token": "write"}
    action = publish["steps"][-1]["uses"]
    assert re.fullmatch(r"pypa/gh-action-pypi-publish@[0-9a-f]{40}", action)
    assert "with" not in publish["steps"][-1]  # no password: OIDC only
    assert workflow["jobs"]["github-release"]["needs"] == ["build", "publish"]
    assert workflow["permissions"] == {"contents": "read"}
