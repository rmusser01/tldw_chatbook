"""Read exact base source for the static timer census without editing product files."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
BASE = "b1ca1b11fa44a8e0ff0a09dfbffe940072861ba4"
original_read = Path.read_text
sources = {}
for name in subprocess.check_output(
    [
        "git",
        "diff",
        "--name-only",
        BASE,
        "--",
        "tldw_chatbook",
        "Tests/Architecture/test_timer_path_static_update_inventory.py",
    ],
    cwd=ROOT,
    text=True,
).splitlines():
    if not name.endswith(".py"):
        continue
    result = subprocess.run(
        ["git", "show", f"{BASE}:{name}"], cwd=ROOT, capture_output=True
    )
    if result.returncode == 0:
        sources[(ROOT / name).resolve()] = result.stdout.decode("utf-8")


def base_read(self, *args, **kwargs):
    if self.resolve() in sources:
        return sources[self.resolve()]
    return original_read(self, *args, **kwargs)


Path.read_text = base_read
