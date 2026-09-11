"""Close out a rail-UX task file directly (five-digit IDs break `backlog task edit`).

Usage: close_rail_ux_task.py <task-id> <notes-file-or-'-'> [--plan <text>]
Reads close-out notes from the file (or stdin for '-') and writes them into
the task's Implementation Notes, checks every AC box, and sets status Done.
"""

from __future__ import annotations

import sys
from pathlib import Path

if len(sys.argv) < 3:
    raise SystemExit(__doc__)

tid = sys.argv[1].removeprefix("TASK-")
notes_src = sys.argv[2]
plan_text = ""
if "--plan" in sys.argv:
    plan_text = sys.argv[sys.argv.index("--plan") + 1]

notes = (
    sys.stdin.read() if notes_src == "-" else Path(notes_src).read_text()
).strip()

tasks = Path("backlog/tasks")
matches = list(tasks.glob(f"task-{tid} - *.md"))
if len(matches) != 1:
    raise SystemExit(f"task-{tid}: expected one file, found {matches}")
path = matches[0]
text = path.read_text()

if plan_text and "## Implementation Plan" not in text:
    anchor = "## Acceptance Criteria"
    plan_block = (
        "## Implementation Plan\n\n<!-- SECTION:PLAN:BEGIN -->\n"
        + plan_text.strip()
        + "\n<!-- SECTION:PLAN:END -->\n\n"
    )
    text = text.replace(anchor, plan_block + anchor, 1)

block = "\n### Close-out (2026-09-10)\n\n" + notes + "\n"
if "<!-- SECTION:NOTES:BEGIN -->" in text:
    text = text.replace(
        "<!-- SECTION:NOTES:BEGIN -->",
        "<!-- SECTION:NOTES:BEGIN -->" + block,
        1,
    )
else:
    text += (
        "\n## Implementation Notes\n<!-- SECTION:NOTES:BEGIN -->"
        + block
        + "<!-- SECTION:NOTES:END -->\n"
    )

import re

text = re.sub(r"- \[ \] (#\d+)", r"- [x] \1", text)
text = text.replace("status: In Progress", "status: Done").replace(
    "status: To Do", "status: Done"
)
path.write_text(text)
print(f"closed task-{tid} ({path.name})")
