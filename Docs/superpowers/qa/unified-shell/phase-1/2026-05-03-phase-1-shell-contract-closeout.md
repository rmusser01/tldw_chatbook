# Phase 1 Shell Contract Closeout

Date: 2026-05-03
Task: `TASK-2.4`
Parent Task: `TASK-2`
Branch: `codex/unified-shell-phase1-closeout-replay`
Base: `origin/dev` at `8acf1ed4`

## Current Baseline

Phase 1 is replayed after the merged Phase 1 slices:

- `TASK-2.1` - QA walkthrough harness and evidence template.
- `TASK-2.2` - destination action ownership audit.
- `TASK-2.3` - false Console-launch affordance fix.
- `TASK-2.4` - final shell-contract replay and closeout.

## Replay Matrix

| Destination | Primary route | Destination ID | Replay result | Primary action state |
| --- | --- | --- | --- | --- |
| Home | `home` | `home` | Navigation destination is present; Home dashboard sections and next-best-action routing are covered. | Active-work controls remain honest placeholder hooks until Phase 2 adapters are wired. |
| Console | `chat` | `console` | Navigation opens Console and pending launch cards render when live-work context exists. | Console remains the live-work hub; provider/model execution is Phase 3 scope. |
| Library | `library` | `library` | Library routes to Notes, Media, Conversations, Import/Export, Search/RAG, and staged Console context. | Working compatibility wrapper over legacy source surfaces. |
| Artifacts | `artifacts` | `artifacts` | Artifacts routes to Chatbooks and stages artifact context into Console. | Working compatibility wrapper; generated output service adoption remains Phase 4. |
| Personas | `personas` | `personas` | Personas routes to the character/persona/prompt surface and stages persona context into Console. | Working compatibility wrapper over existing persona management. |
| W+C | `watchlists_collections` | `watchlists_collections` | W+C route, title, Watchlists/Collections sections, and Watchlists route are covered. | Console follow is disabled with recovery copy until actionable W+C payloads exist. |
| Schedules | `schedules` | `schedules` | Schedules route and timing/recovery sections are covered. | Console recovery is disabled with recovery copy until schedule run payloads exist. |
| Workflows | `workflows` | `workflows` | Workflows route and recipe/dry-run/approval/output sections are covered. | Console launch is disabled with recovery copy until workflow execution payloads exist. |
| MCP | `mcp` | `mcp` | MCP route and `tools_settings` alias are covered; management unavailable state is explicit. | MCP management button is disabled with recovery copy. |
| ACP | `acp` | `acp` | ACP route, runtime-unconfigured state, and session boundaries are covered. | ACP launch and Console follow are disabled until runtime/session payloads exist. |
| Skills | `skills` | `skills` | Skills route, SKILL.md boundary, local directory copy, and Console context staging are covered. | Import is disabled with recovery copy; attach-to-Console stages skills context. |
| Settings | `settings` | `settings` | Settings route and Appearance customization route are covered. | Working appearance route; Settings excludes MCP/tool-control ownership. |

## Focused Verification

Commands run from `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-unified-shell-phase1-closeout-replay`:

- Red closeout test: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_unified_shell_phase1_closeout.py -q`
- Red result before closeout evidence existed: `3 failed`
- Closeout contract: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_unified_shell_phase1_closeout.py -q`
- Closeout contract result: `3 passed`
- Replay suite: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_unified_shell_phase1_closeout.py Tests/UI/test_master_shell_navigation.py Tests/UI/test_destination_shells.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_home_screen.py Tests/UI/test_shell_destinations.py Tests/UI/test_unified_shell_qa_protocol.py Tests/UI/test_unified_shell_destination_action_audit.py -q`
- Replay suite result: `73 passed, 1 warning`
- Closeout contract evidence: `Tests/UI/test_unified_shell_phase1_closeout.py`
- Navigation and labels: `Tests/UI/test_master_shell_navigation.py`, `Tests/UI/test_shell_destinations.py`
- Destination primary actions and honest blocked states: `Tests/UI/test_destination_shells.py`, `Tests/UI/test_console_live_work_handoffs.py`
- Home dashboard controls and next-best-action routing: `Tests/UI/test_home_screen.py`

Warning boundary: the remaining dependency-version warning is not a shell-contract failure.

## Residual Risk

- Phase 1 proves shell contract usability, route ownership, action honesty, and focused mounted Textual behavior. It does not claim live provider, scheduler, workflow, ACP runtime, MCP server, or service-backed destination completion.
- Home active-work controls intentionally remain placeholder hooks until `TASK-4`.
- Console live-agent execution remains `TASK-3`.
- Destination service adoption remains `TASK-5`.

## Phase 1 Closeout Decision

No unresolved Phase 1 shell-contract blockers remain.

Phase 1 is verified for shell contract completion because all top-level destinations have honest status/action ownership, the replay suite covers navigation/layout labels/focusable primary action states, changed shell seams have focused automated tests, and durable QA evidence exists under this directory.
