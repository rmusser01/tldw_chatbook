# Baseline at origin/dev d0face3ebe (2026-09-21) — measured before any tier-2 fix landed

Anything red below is PRE-EXISTING. Do not "fix" it as part of a tier-2 batch unless the
batch's own task says to, and never let it be mistaken for a regression you caused.

## preflight.sh — GREEN (all 8 checks pass)
Run as: `PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python ./scripts/preflight.sh`
Notable, and exactly the report's structural finding — these are green WITHOUT PROVING ANYTHING:
- `timestamp writers: 0 datetime.utcnow() site(s), 0 naive ... (0 pinned)` — **empty census reads as OK
  while seven review slices found live writers.** The check's predicate is too narrow (ADR-173 contract).
- `Canvas Mermaid assets reproduce: 6 outputs` — **2 of the 6 are copied OUT of the directory the check
  then compares against**, so they self-certify.
- `textual worker contract: ... 269 post-await DOM lookup(s) in 136 function(s), none new` — the W002
  section predicate treats a `finally:`/`except:` body as guarded by its own statement's handlers,
  hiding 59 sites.

## Size ratchets — 5 RED (pre-existing; TASK-32809.1's re-pin is outstanding)
```
FAILED Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/Chat/console_chat_controller.py]
FAILED Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/UI/MCP_Modules/mcp_workbench.py]
FAILED Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/personas_screen.py]
FAILED Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/Widgets/Console/console_transcript.py]
FAILED Tests/Architecture/test_module_size_ratchet.py::test_budget_is_not_left_slack[tldw_chatbook/app.py]
```
Command: `.venv/bin/python -m pytest Tests/Architecture/test_module_size_ratchet.py Tests/Architecture/test_library_modules_size_ratchet.py -q` -> 5 failed, 58 passed

## ruff — 25,500 errors on tldw_chatbook/ (project does not gate on ruff)
Do not run `ruff --fix` across the tree in a fix batch; it would bury the real diff.

## ADR-126 recovery gate
`Tests/UI/*` and other suites raise `RecoveryRequired` at fixture setup in a clean worktree
(`Backup_Recovery/raw_participants.py` via `app.py` `APP_CONFIG = load_settings()`).
Gate-free unit tests run locally; integration tests are CI-only. Do not report this as your regression.
