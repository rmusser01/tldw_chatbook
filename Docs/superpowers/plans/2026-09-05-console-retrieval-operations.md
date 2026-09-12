# Console retrieval operations ownership (TASK-31773)

> Execute with the executing-plans skill; root reviews the scoped diff before commit.

**Goal:** Move dictionary/world-book send appliers and picker orchestration to the existing retrieval controller without changing behavior or raising screen ceilings.

**Architecture:** `ConsoleRetrievalController` already owns retrieval summaries and the current-conversation dependency. Move six method bodies there. Keep `@on` handlers, dialog admission flags and `run_worker(..., group="console-io")` on `ChatScreen`; wire two named completion callbacks that clear those flags. Retain the existing dictionary-service accessor and callback. No new controller, proxy, state descriptor, or UI geometry change.

**Tech stack:** Python, Textual 8, pytest, existing explicit late-bound wiring.

ADR required: no. ADR path: N/A. This is direct implementation of DESIGN.md §7 and the existing [screen-decomposition design](../specs/2026-08-02-screen-decomposition-design.md), not a new interface policy.

## 1. Characterize the exact seam

- Census all six method callers and three standalone controller constructor fixtures. The screen chat-controller construction and rebind paths both supply the two appliers; four event handlers schedule pickers. Three world-book test calls reach private workers directly; one world-info test binds the old class method and patches its module-local config lookup.
- Add no-mount tests proving the retrieval owner provides all six operations, and all picker early-return paths release the appropriate screen guard through an explicit callback.
- Verify the new ownership test is RED before production edits. Run whole existing dictionary/world-book integration and retrieval files as the baseline.

## 2. Move only approved operations

- Move six bodies from `UI/Screens/chat_screen.py` to `UI/Console_Modules/retrieval.py` with imports/constants.
- Normalize only the current-conversation call, same-owner summary calls, and finally-block completion callbacks. Preserve exception handling, notification strings, local imports and picker awaits.
- Add required named completion callbacks in `wiring.py`; update three standalone constructor fixtures. Retarget screen applier bindings and event scheduling to `self._retrieval` without changing worker arguments.
- Retarget the private-call and monkeypatch fixtures to the actual owner; keep their assertions intact.

## 3. Verify and review

- Compare every moved function AST to pre-move HEAD, allowing only the three explicit dependency rewrites above. Census callers again; confirm construction callbacks remain late-bound.
- Run whole Console dictionary, world-book, retrieval, automatic-library-preparation, auto-RAG, RAG settings, and native-chat-flow test files. Run Console reuse coverage and relevant architecture ownership checks.
- Run Ruff lint and changed-region formatting; measure exact screen lines/direct AST methods without editing shared ceilings or inventories.
- Report evidence, any unrelated failures, exact method/line gains and scoped diff to root. Commit only after review; mark the task Done only when its acceptance criteria are satisfied.
