"""Release helpers used by `.github/workflows/release.yml` (see docs/releasing.md).

    python3 scripts/release.py check-tag v0.1.0    # tag == __version__, CHANGELOG entry exists
    python3 scripts/release.py notes 0.1.0 [-o F]  # the CHANGELOG section (GitHub release notes)

Stdlib only, so it runs before anything is installed. Exit code 1 with a message on stderr when
the check fails or the CHANGELOG has no section for the version.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
INIT = ROOT / "src" / "agentlens" / "__init__.py"

_VERSION_RE = re.compile(r'^__version__ = "([^"]+)"$', re.MULTILINE)
_TAG_RE = re.compile(r"^v(\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?)$")
# "## [0.1.0] - 2026-11-14" (Keep a Changelog); the link reference definitions at the end
# ("[0.1.0]: https://...") are not part of any section.
_HEADING_RE = re.compile(r"^## \[(?P<version>[^\]]+)\]", re.MULTILINE)
_LINK_DEF_RE = re.compile(r"^\[[^\]]+\]: \S+\s*$", re.MULTILINE)


class ReleaseError(Exception):
    """A release check failed."""


def package_version(init: Path = INIT) -> str:
    """``agentlens.__version__`` read from the source, without importing the package."""
    match = _VERSION_RE.search(init.read_text(encoding="utf-8"))
    if match is None:
        raise ReleaseError(f"no __version__ in {init}")
    return match.group(1)


def tag_version(tag: str) -> str:
    """``v0.1.0`` -> ``0.1.0``; anything else is an error."""
    match = _TAG_RE.match(tag.strip())
    if match is None:
        raise ReleaseError(f"tag {tag!r} is not of the form vX.Y.Z (optionally aN/bN/rcN)")
    return match.group(1)


def changelog_section(text: str, version: str) -> str:
    """Body of the ``## [version]`` section of a Keep a Changelog file (heading excluded)."""
    headings = list(_HEADING_RE.finditer(text))
    for i, heading in enumerate(headings):
        if heading.group("version") != version:
            continue
        line_end = text.find("\n", heading.start())
        start = len(text) if line_end == -1 else line_end + 1
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = _LINK_DEF_RE.sub("", text[start:end]).strip()
        if not body:
            raise ReleaseError(f"the CHANGELOG section for {version} is empty")
        return body + "\n"
    raise ReleaseError(f"no '## [{version}]' section in the CHANGELOG")


def check_tag(tag: str, *, init: Path = INIT, changelog: Path = CHANGELOG) -> str:
    """Check that ``tag`` matches the package version and has a CHANGELOG section."""
    version = tag_version(tag)
    package = package_version(init)
    if version != package:
        raise ReleaseError(
            f"tag {tag} does not match agentlens.__version__ = {package!r} "
            f"(bump src/agentlens/__init__.py or tag v{package})"
        )
    changelog_section(changelog.read_text(encoding="utf-8"), version)
    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-tag", help="tag == __version__ and a CHANGELOG entry exists")
    check.add_argument("tag")
    notes = commands.add_parser("notes", help="print the CHANGELOG section of a version")
    notes.add_argument("version")
    notes.add_argument("-o", "--output", type=Path, help="write to this file instead of stdout")
    for command in (check, notes):
        command.add_argument("--changelog", type=Path, default=CHANGELOG)
    check.add_argument("--init", type=Path, default=INIT)
    args = parser.parse_args(argv)
    try:
        if args.command == "check-tag":
            version = check_tag(args.tag, init=args.init, changelog=args.changelog)
            print(f"tag {args.tag} matches agentlens {version}; CHANGELOG entry found")
            return 0
        body = changelog_section(args.changelog.read_text(encoding="utf-8"), args.version)
    except (ReleaseError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.output is None:
        sys.stdout.write(body)
    else:
        args.output.write_text(body, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
