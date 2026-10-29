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
- Initial adjacent controller/prompt/workspace run: 21 passed, five fixture
  failures. The failures occurred in app-factory config loading with
  `RecoveryRequired("raw_source_selection_changed")`, before entering changed
  code. The first case was also run in the unchanged pre-fix checkout and
  produced the identical exception. The merge follow-up repaired those five
  app-backed tests using the existing private-profile subprocess helper and
  explicit app-resource retirement. All 13 cases in the prompt module now pass
  their original assertions; no production admission guard was changed.
- Ruff is clean for character.py, wiring.py and the UAT helper. Comparing
  session.py and test_console_session_controller.py against the base found zero
  introduced diagnostics (10 and 22 existing diagnostics respectively).
- Task-ID/path guard passed across 5,924 task files. Diff whitespace check passed.
- Independent read-only code review found no Critical or Important findings.

Merge follow-up verification before the final dev rebase:

- The prompt module's 13 cases and the combined 32-case character/workspace
  set pass after repairing the isolated-profile lifetime.
- Reusing the existing active-session lookup in character and persona handoffs
  preserves behavior and keeps session.py at its unchanged 7,000-line budget.
  The ratchet rejected 7,016 lines before this cleanup; both ratchet checks now
  pass. All 19 targeted launch, pristine-session and persona checks pass afterward.
- All 400 targeted CI-contract, census, lane-placement and size checks pass.
  Six launch variants run in the admission-sensitive lane; five repaired prompt
  cases run individually in the UI census, which passes at 1,150 entries.
- Scoped Ruff checks and the Textual worker guard pass. The CI-contract file's
  existing formatting debt is outside the changed lines. Local preflight passes
  every guard except Canvas Mermaid, which requires Python 3.12.11; this Windows
  environment has 3.12.10. Hosted CI pins the required 3.12.11 build.
- Final independent review of the helper reuse, fixture repair and CI enrollment
  found no Critical, Important or Minor findings; nothing in that scope was
  declined. Qodo lacked workspace credits, and the user explicitly authorized
  independent review to replace it for PR #3306. Required CI and thread resolution
  remain enforced by the merge queue.

## Rebase onto the workspace activation repair

The reviewed head `a43cc847e8` passed every hosted required CI check, including
the pinned Canvas guard. Before auto-merge, dev advanced to `ed7481a30c` by
merging PR #3307 / TASK-37620. That change supplies the workspace alignment
callbacks for character and persona launches and adds eight real model-picker
Apply/Cancel/pristine regressions.

The rebase retains that implementation and removes this branch's now-redundant
optional workspace callback and wiring. The remaining production fix supplies
character defaults from the current worked chat pair, using the established
callback for activation. Both PRs' CI entries remain enrolled; the combined
census has 1,157 entries. The controller is 6,989 lines under dev's 6,990-line
budget. Original UAT receipts describe the pre-rebase run; combined automated
model-picker checks qualify the integration after rebase.

Post-rebase verification passes all 31 combined character, prompt and real
model-picker checks, including all eight TASK-37620 cases and all six new
stale-provider/workspace variants. All 400 CI/census/size contracts pass again.
Scoped Ruff, task-ID/path and whitespace guards pass. Independent review of
`ed7481a30c..51de16779e` found no Critical, Important or Minor findings and
confirmed one activation callback per path. Native rendering and real inference
remain outside the evidence. Required hosted CI must run on the rebased head.

The next reviewed head `d0ae6d68be` also passed every hosted check. Dev then
advanced to `ff815f8559` through PR #3302 (speech controls and modal theming).
The second rebase preserves both lesson additions and all census entries (1,161).
Session production logic is byte-identical to the previously qualified head;
the only wiring diff from current dev is still the character-default callback.
Independent conflict-resolution review found no findings; all 31 character,
prompt and real model-picker cases and all 400 CI/census/size contracts pass
again. Scoped Ruff, task IDs across 5,928 files and whitespace guards pass.
Further ordinary queue updates are left to the queue;
disabling/re-enabling auto-merge resets its FIFO position.

The repaired fixture cases are:

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
