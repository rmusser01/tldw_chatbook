"""Named hotspot counts and source provenance for the presentation follow-up."""

from collections import Counter
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).parent
ROOT = OUT.resolve().parents[3]
names = (
    "_ordered_candidates",
    "_build_console_inspector_state",
    "_with_console_conversation_browser_state",
    "get_conversation_appearances",
    "_unread_notification_count",
)
result = {}
for variant in ("before", "after", "control", "final"):
    path = OUT / f"plain-ready-presentation-{variant}-samples.json"
    if not path.exists():
        continue
    counts = Counter()
    for row in json.loads(path.read_text()):
        if row["stage"].startswith("console-resume-"):
            for name in {name for _filename, name, _line in row["stack"]}:
                counts[name] += row["count"]
    result[variant] = {name: counts[name] for name in names}
result["source_sha256"] = {
    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for name in (
        "tldw_chatbook/Utils/textual_css_fastpath.py",
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "tldw_chatbook/UI/Console_Modules/workspace.py",
    )
}
text = json.dumps(result, indent=2)
(OUT / "presentation-hotspots.json").write_text(text, encoding="utf-8")
print(text)
