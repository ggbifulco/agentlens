"""Screenshot an HTML report, then its trace viewer (Playwright + Chromium).

Not a project dependency: run it with any Python that has ``playwright``::

    python scripts/screenshot_html_report.py report.html docs/images/html-report.png

writes ``html-report.png`` (the report) and ``html-report-trace.png`` (a
regressed case's trace open in the viewer) next to each other.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main(html: str, png: str) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1280, "height": 860}, device_scale_factor=2, color_scheme="light"
        )
        page.goto(Path(html).resolve().as_uri())
        page.screenshot(path=png, type="png")
        page.locator("details.case.st-regression a.open-trace").nth(1).click()
        page.locator("#trace-viewer .tv-row").nth(2).click()  # the tool call of the first step
        trace_png = Path(png).with_name(Path(png).stem + "-trace.png")
        page.screenshot(path=trace_png, type="png")
        browser.close()
    print(f"wrote {png} and {trace_png}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
