"""Screenshot the local web UI: the trace list, then a failing trace's span tree.

Not a project dependency: run it with any Python that has ``playwright``,
against a running UI with the support-agent regression story in its database::

    agentlens --db /tmp/ui.db run examples/support_agent/suite.yaml --name main --set-baseline
    SUPPORT_PROMPT_VERSION=v2 agentlens --db /tmp/ui.db run examples/support_agent/suite.yaml \\
        --name prompt-v2
    agentlens --db /tmp/ui.db ui --no-open --gates examples/support_agent/gates.yaml &
    python scripts/screenshot_web_ui.py http://127.0.0.1:8765 docs/images/web-ui.png

writes ``web-ui.png`` (the candidate's traces, failing runs flagged) and
``web-ui-trace.png`` (the regressed ``refund-over-limit`` run, its refund call selected),
reduced to 256 colours when Pillow is installed (README images stay under 400 KB).
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main(base: str, png: str) -> None:
    base = base.rstrip("/")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1280, "height": 860}, device_scale_factor=2, color_scheme="light"
        )
        page.goto(f"{base}/traces?experiment=prompt-v2&limit=25")
        page.screenshot(path=png, type="png")
        page.goto(f"{base}/traces?experiment=prompt-v2&case_id=refund-over-limit")
        page.locator("table.list tbody a").first.click()
        page.wait_for_load_state()
        page.locator("#trace-viewer .tv-row", has_text="issue_refund").first.click()
        trace_png = Path(png).with_name(Path(png).stem + "-trace.png")
        page.screenshot(path=trace_png, type="png")
        browser.close()
    for path in (Path(png), trace_png):
        _shrink(path)
    print(f"wrote {png} and {trace_png}")


def _shrink(path: Path) -> None:
    try:
        from PIL import Image
    except ImportError:  # optional
        return
    image = Image.open(path).convert("RGB")
    image.quantize(colors=256, dither=Image.Dither.NONE).save(path, optimize=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
