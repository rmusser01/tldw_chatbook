# Character chat model and modal-resume qualification

Date: 2026-10-29 (America/Los_Angeles).
Branch: `codex/fix-character-chat-selection`, based on dev `d36affec9b`.
Task: TASK-37600.

## Running-app UAT

The real `TldwCli` ran through the production served terminal on Windows.
`live_uat.py` used a private disposable profile and a loopback HTTP fixture on
127.0.0.1:19877; the served UI listened on 127.0.0.1:19876. No provider gateway,
handoff, model-Apply, workspace-reconciliation, or transcript method was mocked.
The helper seeded a worked named-workspace chat using llama.cpp/model A and
older vLLM control mirrors through the existing first-chat selection seam.

Browser actions and independent state/HTTP receipts established:

1. Open Roleplay, select bundled pixel-migu, click Chat now. The new global chat
   used llama.cpp/model A despite the older vLLM controls. The registry aligned
   to Default. See [launch receipt](02-launch.json) and [screen](02-launch.png).
2. Click Model above the composer, find model B, press Enter to Apply. The same
   character session stayed active and used model B.
   See [Apply receipt](03-apply.json) and [screen](03-apply.png).
3. Reopen the picker, enter model A, cancel through the backdrop. The same
   session stayed active with model B.
   See [Cancel receipt](04-cancel.json) and [screen](04-cancel.png).
4. Send “Hello pixel-migu. Confirm this character chat.” The actual HTTP
   request selected model B and carried Pixel Migu's system prompt. The reply
   marker visibly rendered in that same chat. The source chat's two messages
   were unchanged. See [request](provider-request.json),
   [reply receipt](05-reply.json), and [screen](05-reply.png).

All four receipts name the same character session. The HTTP fixture provides a
deterministic completion; this verifies routing and presentation, not llama.cpp
inference quality. Native terminal rendering was not exercised.

## Automated verification

- Pre-change baseline: 12 targeted workspace/character-default cases passed.
- Red: all six new cases failed on the expected provider or registry assertion.
- Green: 24 character-launch cases passed, including pristine convergence,
  source preservation, duplicate handoffs and both launch routes.
- Final focused run after test cleanup: all 19 new launch, character-controller,
  and workspace-reconciliation checks passed.
- Adjacent controller/prompt/workspace run: 21 passed, five existing fixture
  failures. The failures occur in app-factory config loading with
  `RecoveryRequired("raw_source_selection_changed")`, before entering changed
  code. The first case was also run in the unchanged pre-fix checkout and
  produced the identical exception.
- Ruff is clean for character.py, wiring.py and the UAT helper. Comparing
  session.py and test_console_session_controller.py against the base found zero
  introduced diagnostics (10 and 22 existing diagnostics respectively).
- Task-ID/path guard passed across 5,924 task files. Diff whitespace check passed.
- Independent read-only code review found no Critical or Important findings.

The pre-existing fixture failures are:

- `test_personas_start_chat_seeds_template_provenance_with_global_name`
- `test_character_picker_new_chat_seeds_template_provenance`
- `test_character_picker_keeps_raw_identity_but_sanitizes_notification`
- `test_character_picker_swap_uses_override_and_only_greets_empty_chat`
- `test_character_swap_surfaces_refused_durable_projection`

The full suite was not run, following the repository's targeted-verification rule.

## Reproduce the live setup

Create a private disposable directory under an independently verified private
parent, then run from this worktree:

```powershell
python Docs/superpowers/qa/2026-10-29-character-chat-selection/live_uat.py --profile C:/path/to/private/disposable-profile
```

Open http://127.0.0.1:19876/?fontsize=12 and follow the browser actions above.
Close the browser terminal and stop the helper after the check.

ADR required: no. Existing ADR-006/095 and the TASK-15120 workspace ownership
ruling apply; no schema, permissions, or provider-ownership contract was added.
