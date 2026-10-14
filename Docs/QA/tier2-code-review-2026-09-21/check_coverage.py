# Tier-2 coverage completeness check.
# Every .py under tldw_chatbook/, minus the v1 PROMPT.md always-excluded paths,
# must be claimed by exactly one row in slice_paths.txt.
import pathlib
EXCLUDE = (".venv", "Third_Party", "Tests", "__pycache__")
claimed = [pathlib.Path(l.split("#")[0].strip())
           for l in pathlib.Path("qa/tier2-code-review-2026-09-21/slice_paths.txt").read_text().splitlines()
           if l.split("#")[0].strip()]
unclaimed = [f for f in sorted(pathlib.Path("tldw_chatbook").rglob("*.py"))
             if not any(x in f.parts for x in EXCLUDE)
             and not any(f == c or c in f.parents for c in claimed)]
for f in unclaimed:
    print("  ", f)
print(f"UNCLAIMED: {len(unclaimed)}")
