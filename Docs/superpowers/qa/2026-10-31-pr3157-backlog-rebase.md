# PR #3157 backlog integration — 2026-10-31

Base: `57e330e735fff25515778e839f31ed85602c34f2`. Original remote head: `ebba583bd1ae2ceea1e87e78f3d07597cc76a7e6`. Rebased head before repairs: `574bad359b4fc3d8f1aa33f3ec1448980a810944`.

ADR required: no new ADR. Existing ADR-237 owns the local board/service boundary; ADR-150/161 govern tokens; ADR-031/152 govern navigation and ADR-240 retires standalone MCP. These repairs implement those existing contracts. User explicitly waived hosted CI and size limits for backlog integration and authorized independent requesting-code-review instead of Qodo. No release work.

## Repairs

- Preserve the retired MCP route mapping while adding Kanban. Update the touched stale shell-order expectation to the current destinations.
- Reserve literal column names before allocating duplicate suffixes. Lists named `Done`, `Done`, `Done (2)`, `Done (3)`, `Done (2)` retain five distinct columns and the correct store IDs. Real-store move/create operations address the duplicate column correctly.
- Supply independent UUID client IDs to board, list, and card creation. The existing recording fake accepted incomplete requests, but the real LocalKanbanService rejected every first-use create for missing `client_id`.
- Render user column names literally in the new-card modal.
- Use existing spacing, sizing, boundary and focus design tokens; rebuild generated styles. Mounted board harnesses now load the production consolidated CSS.

## Evidence

Private receipts are under sibling `test-runtime/`:

- `regression-red-1793471747570015300/pytest.log`: 13 original cases passed; literal label regression failed as intended. Initial collision setup failed for missing required client ID (not collision proof).
- `real-store-red-1793471952058413100/pytest.log`: original loader produced four columns for five real stored lists, losing the second list; mounted keyboard bootstrap produced zero boards. The original loader was temporarily restored for this control and the repair then restored byte-for-byte.
- `real-store-green-1793471997977066700/pytest.log`: all 16 focused cases passed in 37.11 seconds, including mounted keyboard bootstrap/create through actual SQLite, duplicate-name move/create, and literal modal rendering. No skips or teardown errors.
- Targeted Ruff check passed. `navigation-tokens-1793472123370456600/pytest.log`: five touched shell order/shortcut/palette and token contracts passed in51.29seconds. Final CSS rebuild and `git diff --check` passed. Independent review remains pending.

## Limits

No full test sweep, hosted CI, release, server-backed board, scheduling feed or external Kanban qualification. The UI census accepts only Tests/UI paths; the board tests remain in their existing Tests/Kanban suite. F8 is the next free shortcut on this reviewed base; if earlier Plugins PR #3148 lands first, final rebase must allocate Kanban the next free slot and re-review that integration. No existing destination shortcut may move.


## Independent review repairs

Full review of `c02dc559b117ee22130670d26bcdc603aabce7e6` found four Important keyboard/access defects. Normal terminal Shift+H/L emits uppercase keys; horizontal overflow consumed arrow bindings; rebuilding columns lost selected-card visibility; fixed Static details could not scroll. All four were reproduced in actually pushed screens with production CSS (`keyboard-review-red-1793472836246736300/pytest.log`:4failed).

Repairs bind uppercase H/L, give board arrows priority while declining them for Input/inspector focus, update selection classes without rebuilding mounted columns, reveal selection after layout, and use a keyboard-scrollable full-detail container. Escape returns to board navigation. Existing store and mutation paths are unchanged.

`keyboard-review-green-1793472941878317000/pytest.log`:19passed; one Input caret assertion failed because its fixture placed the caret before focus/select-all settled. After adding a focus settlement pause, only that test reran: `keyboard-input-final-1793473077128288200/pytest.log`:1passed11.16seconds. This is combined targeted evidence, not a claim that the earlier20-case invocation was wholly green. Ruff/format/whitespace and final CSS regeneration were verified. The same independent reviewer must check the repair before merge.
