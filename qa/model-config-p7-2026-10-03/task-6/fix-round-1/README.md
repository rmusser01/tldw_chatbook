# TASK-33007.6 fix round 1: the evidence behind the Task 6 review's open claims (2026-10-04)

The Task 6 review listed five claims it could not check from the diff. Each one was
re-run against the Task 6 base `d9b73fc412`. Finding 4 was a real regression, fixed in
`1db0f78198`; the other four hold and are recorded here.

All test runs used the venv's Python with `PYTHONPATH` set to the tree under test, so
the venv's editable install (which points at the main checkout) was not measured. They
ran under `env -i`, with `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME` and
`TLDW_CONFIG_PATH` in a scratch directory, the null keyring, `-p no:randomly` and the
scratch admission plugin (lessons-testing-evidence, "A local red wall of
`RecoveryRequired`"). Base, Task 6 head, fix and dev merge-base trees were detached
worktrees in scratch, removed afterwards. The probe, runner and live-driver scripts
lived only in scratch and are not committed.

| File | Finding | What it shows |
|---|---|---|
| `snapshot-and-search-index.txt` | 1, 2 | `test_llamacpp_snapshot_settings.py` fails 11 of 15 in one process on both sides, because the plugin's shared profile leaks one test's config writes into the next; one node per process, all 15 pass at base and at head. `test_settings_search_index.py`: the same 2 failure names on dev, base and head, and head's missing-from-index list is base's minus the two snapshot fields |
| `covering-parity.txt` | 3 | the 70-file covering set (plus the new file at head), base and fix commit only, both at once with 6 workers each and no other suite of this session; counts, every head-only and base-only name, each head-only name re-run alone twice per side |
| `compact-rows.txt` | 4 | at <=100 columns the card-wide descendant selector added a margin under every Catalog refresh provider row (open group 71 -> 96 rows at 100x40); the fix names the containers instead, and every row matches base again apart from Task 6's intended one-row hours input; RED-first output of the new guard |
| `gutter.txt`, `gutter-captures/` | 5 | every category that fits the pane at 211x44 (21) and 235x52 (22) loses one blank column and no painted text; live captures of Overview and Network at base and at the fix |
| `preflight-1db0f78198.txt` | - | `PYTHON=<venv> ./scripts/preflight.sh` at the fix commit, unpiped, with its exit code (rc 0) |

The controller rulings named in finding 4 (R13, R15, R16, R17, R20, R22, R23) are in the
SDD progress log's preflight section, outside the diff. R20 is the one whose effect was
not what it said. The other six were re-read against the diff:
- R13 (order 6 before 5) was not followed: Task 5 ran first, as dispatched. Task 6's
  AC#13 makes up for it. Captures `../01` and `../02` re-check Task 5's AC#9 after the
  fold, and the fix commit changes only rules for 100 columns or fewer.
- R15 holds. The six ids the vLLM late-ack test snapshots (`#settings-model-value`,
  `#settings-provider-api-key`, `-credential-env-var`, `-endpoint-value`, `-save-result`,
  `#settings-model-profile-temperature`) are not among the ids the diff moves. Every
  moved id is still composed.
- R16 holds. Only this card and its instant-apply groups lose their frames, and the
  ADR-033 test keeps its class assertion.
- R17 holds. `#settings-mc-auto-*` and `#settings-mc-write-*` stay Checkboxes with
  On/Off labels.
- R22 holds. See `../verification.txt` §6-7. The fix adds no dimension literal, and
  `test_component_pattern_governance.py` is in the covering set.
- R23 holds. Reasoning replay override stays in Console Behavior.
