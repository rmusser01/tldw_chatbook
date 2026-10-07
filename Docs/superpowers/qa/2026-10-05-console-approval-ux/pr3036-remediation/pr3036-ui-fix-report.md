# PR3036 UI remediation

Base: 93f639956cc3b66e0bb5e7ea31259a2eeeedca55.
Owned scope: the three source/test files listed below. No CSS, token, generated bundle, guard, runtime permission owner, dependency, task metadata, commit, or development reference was changed by this agent.

ADR required: no new ADR.
ADR paths: backlog/decisions/221-console-approval-interaction-and-feedback.md, plus governing ADR-150/161.
Reason: direct repair of the accepted captured-copy and disclosure geometry contracts.

## Changes

- The card now consumes each captured row's withheld_scope_copy as literal text under that row's disclosed choices. The slot uses existing w-fill/h-auto utility tokens, stays hidden while More options is closed or the captured copy is empty, and refreshes/clears with reused rows. Ordinary same-round re-sync keeps the open disclosure.
- Original captured legal choices, one-time fallback eligibility, raw-shell default/deliberate review, staged selections, count/scope summaries, generation and round guards are untouched.
- The retained production geometry test checks initially closed text/action geometry, opens More options via a real click, then checks the actual Select, text ordering, no overlap, complete containment, approved 28-cell width, unchanged row-height/container/gap bounds, hit-tested and focusable actions, a real scope choice, and Escape preserving that staged choice.
- New painted regressions use the real capture seam for repeated MCP tool stamps at 80x24 and 120x40. They check each affected explanation, viewport containment, complete pinned actions, noncommit, re-sync, Escape, literal bracket rendering, row identity reuse, and absence of old copy after restoration.

## Test commands and receipts

Interpreter: C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe, Python 3.12.10.
Launcher: Docs/superpowers/qa/2026-10-05-console-approval-ux/private_control.py.
Every receipt reports verified checkout module origins, original-home protection before redirection, and startup admission [true, "startup_allowed"]. The launcher and profile guards were unchanged.

Command shape: <python> Docs/superpowers/qa/2026-10-05-console-approval-ux/private_control.py <target> [-k <selection>]

| Target / selection | Result | Log in this report directory |
| --- | --- | --- |
| Tests/UI/test_approval_interaction.py -k withheld_scope, before product fix | RED: 2 failed because explanation slots were missing | pr3036-ui-withheld-red.log |
| Tests/UI/test_approval_batch_geometry.py, retained original test | RED: 1 failed at the initially hidden zero-size Select, matching CI line 57 | pr3036-ui-geometry-red.log |
| Tests/UI/test_approval_interaction.py | GREEN: 28 passed, exit 0 | pr3036-ui-interaction-green.log |
| Tests/UI/test_approval_batch_geometry.py | GREEN: 1 passed, exit 0 | pr3036-ui-geometry-green.log |
| Tests/UI/test_approval_action_ownership.py | GREEN: 25 passed, exit 0 | pr3036-ui-ownership-green.log |
| Tests/UI/test_console_approval_compact_layout.py | GREEN: 3 passed, exit 0 | pr3036-ui-compact-green.log |

During test authoring, two checks were corrected: Textual Static has no public markup property, and SVG screenshots encode spaces as nonbreaking-space entities. The final tests assert actual literal bracket text and decode visible screenshot text. No product behavior was changed to satisfy either authoring mistake.

57 targeted tests passed across the four files. The production compact journeys cover the existing size/theme/Inspect matrix, row reuse, resize, neutral focus/composer preservation, and real batch disclosure.

## Static checks

- C:/Python312/Scripts/ruff.exe check --select F821 on all three owned files: passed.
- Ruff format --check on the two changed test files: passed.
- Scoped git diff --check: passed.
- Formatting the complete card would change five pre-existing unrelated blocks. An exact base-source git show piped as bytes through ruff format --diff --stdin-filename ... - reproduced those same five blocks and exit 1. They are deliberately untouched; full-card formatter cleanliness is not claimed.
- Black is absent from this interpreter; existing Ruff was used without installing anything.

## Source SHA-256

- tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py: 36ebd863ed9ef18b12c067c17fab0b36926ec61e334883e62a76db8976abe584
- Tests/UI/test_approval_interaction.py: f5d5a8de8127ac2e4debe635089bfa07c8c5b290021c985d04c9938c0c65eb16
- Tests/UI/test_approval_batch_geometry.py: 1abf875b08fd8c4ce17fa6d1eb030642c603b17aeca28389e4c0e7e37fe463cf

## Qualifications

These are targeted headless mounted-product and painted-harness checks. Native/browser timing, actual Windows tool dispatch, the broader UI matrix, whole-project static analysis and full test-suite/DoD qualification remain outside this scoped repair. Existing pytest-asyncio and Pydantic deprecation warning noise remains. No full sweep was run.

Root owns task notes, independent final review, index/commit, push and CI. This agent has not marked any Backlog task Done.
