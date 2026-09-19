# TASK-32601 no-new-static-debt qualification — 2026-09-15

User approval: after the final review, the user approved the recommendation to
fix newly introduced static findings and retain documented baseline failures.
ADR-138's task-specific qualification amendment records that policy. This is not
a whole-repository exception or a claim that whole-file lint/format pass.

## Scope and result

- Clean dev base: `77eb2601a63ba473318b8ec1e4edb53f8ac5899e`.
- Qualified code: `62ab9ebc04996f2f59aaea10890bfe1126726a64`.
- Tool: Ruff 0.16.6, default settings, Python 3.12 inferred from unchanged
  `pyproject.toml`. Both sides used the same executable, cwd and settings.
- All 34 changed Python files: 25 new, one rewritten screen and eight incumbent
  integration files. All 25 new files and the rewritten screen are lint/format clean.
- **711 remaining lint findings; all 711 map to identical baseline source spans,
  rule/message and columns. Zero unmatched findings.**
- **Five formatter-failing files, 64 edit hunks; every hunk maps to the exact
  baseline source range and replacement text. Zero unmatched formatter edits.**
- No production, dependency, lint-config or suppression changes in this round.
  Only two test import blocks were sorted; reviewer disposition is recorded separately.

Machine-readable per-file results: `static-gate-results.json` in this directory.

| Incumbent file | Baseline lint | Current lint | Baseline/current format edits |
| --- | ---: | ---: | ---: |
| `tldw_chatbook/app.py` | 485 | 485 | 37 / 37 |
| `tldw_chatbook/config.py` | 164 | 164 | 18 / 18 |
| `tldw_chatbook/DB/private_sqlite.py` | 25 | 25 | 0 / 0 |
| `Tests/DB/test_private_sqlite_inventory.py` | 14 | 14 | 0 / 0 |
| `Tests/UI/test_destination_shells.py` | 11 | 11 | 4 / 4 |
| `Tests/UI/test_destination_visual_parity_correction.py` | 11 | 10 | 2 / 2 |
| `Tests/UI/test_console_live_work_handoffs.py` | 1 | 0 | 0 / 0 |
| `tldw_chatbook/css/build_css.py` | 2 | 2 | 3 / 3 |

The previously reported 713 was the eight incumbent files. Including the old
Workflows screen's two findings gives 715 at the full branch base; rewriting that
screen already removed its two findings. This round removes the two I001 import
findings, leaving 711. This is not an equal-count comparison masking new errors.

## Source-attribution method

One-off coordinator diagnostic used Python's stdlib `subprocess`, `json`,
`collections.Counter` and `difflib.SequenceMatcher(autojunk=False)`; no installed
tooling or CI changes. Sources came from immutable Git blobs, not another checkout.

1. Enumerate exactly `git diff --name-only --diff-filter=AM BASE HEAD -- '*.py'`.
2. Read each side with `git show REF:path`; missing baseline files have no baseline
   diagnostics (do not lint invented empty files).
3. For each existing source, run the same `ruff check --output-format=json
   --stdin-filename PATH -` with that source on stdin; accept only exit 0 or 1.
   Compute a current-to-base line map from identical source blocks.
4. Match each current diagnostic against a counted baseline signature of rule,
   message, every mapped row of its inclusive reported span, start column and end
   column. A changed/unmapped row or exhausted duplicate count is unmatched.
5. Run `ruff format --stdin-filename PATH -` on each source without writing files.
   Diff original/formatted lines. Match each current edit against the counted
   baseline tuple `(operation, mapped_start, mapped_end, exact_replacement_lines)`.
   Nonempty source spans must map contiguously. For pure insertions, both adjacent
   source lines must map to adjacent baseline lines; beginning/end use boundaries.
6. Emit per-file results and assert that both unmatched lists are empty. The final
   diagnostic exited 0 on the pinned qualified code. It does not mask the nonzero
   whole-file Ruff/formatter results in the table.

Initial comparison at `3fd794c376` found 711 exact-mapped lint findings and two
unmatched I001 spans in the import blocks changed by this slice. The implementer
sorted those two blocks, then the final all-file comparison above found zero
unmatched findings. Formatter insertions initially needed boundary-aware matching;
all seven such insertions match the exact baseline edits, including their text.

## Verification commands and evidence

Interpreter/tool prefix: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/`.
All commands run from the authoring worktree with no live profile/network/model use.

```text
ruff check --select I001 Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_visual_parity_correction.py
All checks passed! (after two I001 fixes; exit 0)

ruff check Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_visual_parity_correction.py
Found 10 errors. (source-attributed baseline; exit 1)

ruff format --check Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_visual_parity_correction.py
1 file would be reformatted, 1 file already formatted. (baseline; exit 1)

PYTHONPATH=. python -m pytest Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_visual_parity_correction.py -k workflows -q --timeout=60 --tb=short --show-capture=no -p no:randomly
15 passed, 164 deselected, 1 warning in 16.44s. (implementer; exit 0)

PYTHONPATH=. python -m pytest Tests/Workflows/test_authoring.py Tests/DB/test_workflows_authoring_storage.py -q --timeout=60 --tb=short --show-capture=no -p no:randomly
35 passed, 1 warning in 20.79s. (coordinator; exit 0)
```

The two disjoint selections total 50 fresh passes. Earlier 374-pass, visual and
final integration review evidence remains separately recorded; it was not rerun
or relabeled as fresh here. Existing Requests/Kokoro cleanup warnings persist.
The nonblocking capture-cleanup minor remains tracked; no capture or cleanup ran.
