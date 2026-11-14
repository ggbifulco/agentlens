"""Generate ``docs/images/demo.gif``: ``agentlens run`` and ``compare`` catching a regression.

There is no terminal recorder involved: the script runs the real commands of the
[regression story](../docs/regression-story.md) in a pseudo-terminal (so the CLI prints its
coloured ``rich`` report, exactly as in your terminal), parses the ANSI output with ``rich`` and
draws every frame with Pillow (typing the command, output scrolling, pauses). Re-run it whenever
the terminal report changes::

    uv run --with pillow python scripts/make_demo_gif.py            # writes docs/images/demo.gif
    uv run --with pillow python scripts/make_demo_gif.py out.gif --font /path/to/Mono.ttf

Pillow is only needed here, not by AgentLens. Unix only (``pty``). The commands use a temporary
database, so ``.agentlens/`` is not touched. The model is the deterministic mock, but latency is
real wall-clock time, so its row can differ between recordings.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import pty
import re
import struct
import subprocess
import sys
import tempfile
import termios
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rich.cells import cell_len
from rich.color import Color
from rich.console import Console
from rich.style import Style
from rich.terminal_theme import TerminalTheme
from rich.text import Text

ROOT = Path(__file__).resolve().parent.parent
COLS, ROWS = 108, 47
FONT_SIZE = 14
PAD, BAR = 16, 30

# Catppuccin-like dark palette: background, foreground, 8 normal + 8 bright ANSI colours.
BG, FG = (30, 30, 46), (205, 214, 244)
ANSI = [  # 8 normal + 8 bright colours
    (69, 71, 90), (243, 139, 168), (166, 227, 161), (249, 226, 175),
    (137, 180, 250), (245, 194, 231), (148, 226, 213), (186, 194, 222),
    (88, 91, 112), (243, 139, 168), (166, 227, 161), (249, 226, 175),
    (137, 180, 250), (245, 194, 231), (148, 226, 213), (166, 173, 200),
]  # fmt: skip
THEME = TerminalTheme(BG, FG, ANSI[:8], ANSI[8:])
BAR_BG, TITLE = (24, 24, 37), (166, 173, 200)
DOTS = [(243, 139, 168), (249, 226, 175), (166, 227, 161)]
PROMPT = Style(color="green", bold=True)
COMMENT = Style(color="bright_black", italic=True)

FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
    ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Menlo.ttc"),
    ("C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/consolab.ttf"),
]  # fmt: skip

# The scenario: (kind, text, environment). "comment" lines are typed but not run.
SCRIPT: list[tuple[str, str, dict[str, str]]] = [
    ("comment", "# baseline: the reviewed prompt, 27 golden cases, offline mock model", {}),
    ("run", "agentlens run examples/support_agent/suite.yaml --name main --set-baseline", {}),
    ("clear", "", {}),
    ("comment", '# candidate: a "simplified" prompt that drops the $200 refund limit', {}),
    (
        "run",
        "SUPPORT_PROMPT_VERSION=v2 agentlens run examples/support_agent/suite.yaml"
        " --name prompt-v2",
        {"SUPPORT_PROMPT_VERSION": "v2"},
    ),
    ("clear", "", {}),
    (
        "run",
        "agentlens compare --candidate prompt-v2 --gates examples/support_agent/gates.yaml",
        {},
    ),
    ("run", "echo $?", {}),
]

_CSI_NOT_SGR = re.compile(r"\x1b\[[0-9;?]*[A-Za-ln-z]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)")


@dataclass
class Frame:
    """One distinct screen and how long it stays (milliseconds)."""

    lines: list[Text]
    duration: int


@dataclass
class Recorder:
    """Accumulates screens; consecutive identical screens are merged."""

    frames: list[Frame] = field(default_factory=list)
    screen: list[Text] = field(default_factory=list)

    def snap(self, duration: int, extra: Text | None = None) -> None:
        lines = [*self.screen, *([extra] if extra is not None else [])][-ROWS:]
        key = [line.markup for line in lines]
        if self.frames and [line.markup for line in self.frames[-1].lines] == key:
            self.frames[-1].duration += duration
        else:
            self.frames.append(Frame([line.copy() for line in lines], duration))


def run_in_pty(command: str, extra_env: dict[str, str], db: Path) -> tuple[str, int]:
    """Run one ``agentlens ...`` demo command in a pseudo-terminal: (what it printed, exit code)."""
    tokens = command.split()
    while "=" in tokens[0]:  # leading VAR=value assignments come from ``extra_env``
        tokens.pop(0)
    if tokens[0] != "agentlens":
        raise ValueError(f"not an agentlens command: {command}")
    env = {
        **os.environ,
        **extra_env,
        "AGENTLENS_DB": str(db),
        "COLUMNS": str(COLS),
        "LINES": "200",
        "TERM": "xterm-256color",
    }
    env.pop("NO_COLOR", None)
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 200, COLS, 0, 0))
    argv = [sys.executable, "-m", "agentlens", *tokens[1:]]
    proc = subprocess.Popen(argv, stdin=slave, stdout=slave, stderr=slave, env=env, cwd=ROOT)
    os.close(slave)
    chunks: list[bytes] = []
    while True:
        try:
            data = os.read(master, 65536)
        except OSError:  # EIO: the child closed the terminal
            break
        if not data:
            break
        chunks.append(data)
    os.close(master)
    return b"".join(chunks).decode("utf-8", "replace"), proc.wait()


def to_lines(output: str) -> list[Text]:
    """Terminal output -> one rich Text per line (SGR colours kept, other escapes dropped)."""
    output = _CSI_NOT_SGR.sub("", output.replace("\r\n", "\n"))
    lines = [line.rsplit("\r", 1)[-1] for line in output.split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    return [Text.from_ansi(line) for line in lines]


def record(db: Path) -> list[Frame]:
    """Play the scenario and capture its frames."""
    rec = Recorder()
    exit_code = 0
    rec.snap(600, Text.assemble(("$ ", PROMPT)))
    for kind, command, extra_env in SCRIPT:
        if kind == "clear":
            rec.screen = []
            rec.snap(300, Text.assemble(("$ ", PROMPT)))
            continue
        style = COMMENT if kind == "comment" else Style(bold=True)
        for end in range(0, len(command) + 1, 2):  # typing, two characters a frame
            typed = Text.assemble(("$ ", PROMPT), (command[:end], style), "█")
            rec.snap(35, typed)
        line = Text.assemble(("$ ", PROMPT), (command, style))
        rec.snap(350 if kind == "run" else 500, line + "█")
        rec.screen.append(line)
        if kind == "comment":
            continue
        if command == "echo $?":
            rec.screen.append(Text(str(exit_code), style="bold red" if exit_code else "green"))
            rec.snap(6000, Text.assemble(("$ ", PROMPT), "█"))
            continue
        printed, exit_code = run_in_pty(command, extra_env, db)
        output = to_lines(printed)
        progress = [line for line in output if re.match(r"\[\d+/\d+\]", line.plain)]
        rest = output[len(progress) :]
        for i in range(0, len(progress), 3):  # progress lines stream in
            rec.screen.extend(progress[i : i + 3])
            rec.snap(60)
        rec.screen.extend(rest)  # the report is printed at once
        hold = 6500 if "compare" in command else 2600
        rec.snap(hold, Text.assemble(("$ ", PROMPT), "█"))
    return rec.frames


def load_fonts(path: str | None) -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    """Regular and bold monospace fonts (``--font`` or the first one found)."""
    if path:
        return ImageFont.truetype(path, FONT_SIZE), ImageFont.truetype(path, FONT_SIZE)
    for regular, bold in FONT_CANDIDATES:
        if Path(regular).exists():
            bold = bold if Path(bold).exists() else regular
            return ImageFont.truetype(regular, FONT_SIZE), ImageFont.truetype(bold, FONT_SIZE)
    raise SystemExit("no monospace font found: pass --font /path/to/Mono.ttf")


def rgb(color: Color | None, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if color is None or color.is_default:
        return default
    triplet = color.get_truecolor(THEME)
    return (triplet.red, triplet.green, triplet.blue)


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        round(a[0] + (b[0] - a[0]) * t),
        round(a[1] + (b[1] - a[1]) * t),
        round(a[2] + (b[2] - a[2]) * t),
    )


# Box-drawing characters: which of (up, down, left, right) each one connects.
BOX = {
    "─": (0, 0, 1, 1), "│": (1, 1, 0, 0), "╭": (0, 1, 0, 1), "╮": (0, 1, 1, 0),
    "╰": (1, 0, 0, 1), "╯": (1, 0, 1, 0), "┌": (0, 1, 0, 1), "┐": (0, 1, 1, 0),
    "└": (1, 0, 0, 1), "┘": (1, 0, 1, 0),
}  # fmt: skip


def _box(
    pen: ImageDraw.ImageDraw,
    char: str,
    x: float,
    y: int,
    cell_w: float,
    cell_h: int,
    color: tuple[int, int, int],
) -> None:
    up, down, left, right = BOX[char]
    cx, cy = round(x + cell_w / 2), y + cell_h // 2
    if up:
        pen.line((cx, y, cx, cy), fill=color)
    if down:
        pen.line((cx, cy, cx, y + cell_h), fill=color)
    if left:
        pen.line((round(x), cy, cx, cy), fill=color)
    if right:
        pen.line((cx, cy, round(x + cell_w), cy), fill=color)


def draw(frame: Frame, fonts: tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]) -> Image.Image:
    """Render one screen: a window bar, then the terminal text cell by cell."""
    regular, bold = fonts
    cell_w = regular.getlength("M")
    cell_h = round(FONT_SIZE * 1.3)
    width = round(PAD * 2 + COLS * cell_w)
    height = BAR + PAD * 2 + ROWS * cell_h
    image = Image.new("RGB", (width, height), BG)
    pen = ImageDraw.Draw(image)
    pen.rectangle((0, 0, width, BAR), fill=BAR_BG)
    for i, dot in enumerate(DOTS):
        x = PAD + i * 20
        pen.ellipse((x, BAR // 2 - 6, x + 12, BAR // 2 + 6), fill=dot)
    title = "agentlens: catching a prompt regression"
    pen.text((width / 2, BAR / 2), title, fill=TITLE, font=regular, anchor="mm")
    console = Console(width=COLS, color_system="truecolor", force_terminal=True)
    for row, line in enumerate(frame.lines):
        y = BAR + PAD + row * cell_h
        col = 0
        for segment in line.render(console):
            style = segment.style or Style()
            fg = rgb(style.color, FG)
            bg = rgb(style.bgcolor, BG) if style.bgcolor else None
            if style.reverse:
                fg, bg = bg or BG, fg
            if style.dim:
                fg = blend(fg, BG, 0.45)
            font = bold if style.bold else regular
            for char in segment.text:
                x = PAD + col * cell_w
                if bg is not None:
                    pen.rectangle((x, y, x + cell_w * cell_len(char), y + cell_h), fill=bg)
                if char == "█":
                    pen.rectangle((x, y + 1, x + cell_w - 1, y + cell_h - 2), fill=FG)
                elif char in BOX:  # drawn as lines, so borders join across rows
                    _box(pen, char, x, y, cell_w, cell_h, fg)
                elif not char.isspace():
                    pen.text((x, y + cell_h - 4), char, fill=fg, font=font, anchor="ls")
                col += cell_len(char)
    return image


def fixed_palette() -> Image.Image:
    """A ``P`` image whose palette has each ink colour at 6 blend levels over the background."""
    inks = [FG, *ANSI]
    inks += [blend(ink, BG, 0.45) for ink in inks]  # dim text
    colors: list[tuple[int, int, int]] = [BG, BAR_BG, TITLE, *DOTS]
    for ink in inks:
        for level in range(1, 7):
            color = blend(BG, ink, level / 6)
            if color not in colors:
                colors.append(color)
    if len(colors) > 256:
        raise RuntimeError(f"palette too large: {len(colors)} colours")
    image = Image.new("P", (1, 1))
    image.putpalette([value for color in colors for value in color])
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("output", nargs="?", default=str(ROOT / "docs/images/demo.gif"))
    parser.add_argument("--font", help="a monospace TrueType font (default: DejaVu/Menlo/Consolas)")
    args = parser.parse_args()
    fonts = load_fonts(args.font)
    with tempfile.TemporaryDirectory() as tmp:
        frames = record(Path(tmp) / "demo.db")
    images = [draw(frame, fonts) for frame in frames]
    # One fixed palette for every frame, so colours do not flicker and unchanged pixels stay
    # identical (Pillow then stores only each frame's changed area). A palette computed from the
    # pixels would drop thin coloured lines; this one has every colour of the theme blended
    # into the background at a few levels, which is what anti-aliased text is made of.
    palette = fixed_palette()
    gif = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    gif[0].save(
        args.output,
        save_all=True,
        append_images=gif[1:],
        duration=[frame.duration for frame in frames],
        loop=0,
        optimize=True,
    )
    size = Path(args.output).stat().st_size
    print(f"wrote {args.output}: {len(gif)} frames, {size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
