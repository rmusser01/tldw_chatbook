# TASK-2703 Console Edit Message Action Paint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep every Console Edit Message action visible, hit-testable, and visibly focused in USER and non-USER modals under the real stylesheet and real terminal drivers.

**Architecture:** Preserve the modal's outer geometry, copy, DOM, and handlers. Correct only the inner height allocation by letting the editor consume remaining space with an eight-row minimum; treat any button-face styling as a separate evidence-gated fallback. Pin the result at compositor-cell and hit-test level, then verify it through tmux and a separate PTY before removing the documented workaround.

**Tech Stack:** Python 3.12, Textual 8.x, TCSS, pytest/pytest-asyncio, tmux, `expect`, Backlog.md CLI.

**Design:** `Docs/superpowers/specs/2026-08-13-task-2703-console-edit-modal-paint-design.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This is a localized Textual layout/rendering correction with no storage, ownership, interface, security, dependency, or long-lived UX decision.

---

## File map

- Modify: `Tests/Chat/test_console_edit_message_modal.py` — real-bundle paint, containment, hit-test, focus, contrast, and Enter-activation contract.
- Modify: `tldw_chatbook/Widgets/Console/console_edit_message_modal.py` — minimal editor height allocation.
- Modify only if separately proven necessary: `tldw_chatbook/css/components/_agentic_terminal.tcss` — modal-scoped button paint/focus escape.
- Regenerate only if the source CSS module changes: `tldw_chatbook/css/tldw_cli_modular.tcss`.
- Modify after live verification: `Docs/User_Guide/console/branching-and-rewind.md` — remove the obsolete invisible-button workaround.
- Modify only if the incident adds new reusable knowledge: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`.
- Modify at closeout: `backlog/tasks/task-2703 - Console-Edit-Message-modal-action-buttons-invisible-in-real-terminals.md` — ACs, notes, evidence, and Done status.
- Temporary and remove: one `/tmp/task2703-live.*` root containing the live harness, Expect driver, scratch profile, and runtime state.

### Task 1: Pin the real-bundle rendering failure

**Files:**
- Modify: `Tests/Chat/test_console_edit_message_modal.py`

- [ ] **Step 1: Add the real stylesheet harness and exact region helpers**

Add `Path`, a repository-root/bundle constant, and a harness that differs from the incumbent bare `_ModalHost` only by loading the application bundle:

```python
REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_STYLESHEET = REPO_ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


class _StyledModalHost(App[None]):
    CSS_PATH = str(BUNDLED_STYLESHEET)
```

Adapt the already-proven compositor text helper in
`Tests/UI/test_model_artifact_widgets.py:574-590`. Materialize
`render_strips()` once, walk every row from `region.y` through
`region.bottom`, track each segment's cell interval, and split overlapping
segments at cell boundaries before collecting them. The helper returns a
`tuple[tuple[Segment, ...], ...]` for exactly `region.x:region.right`; a second
helper joins only those cropped segments' text. Task 3 adds style/contrast
helpers after geometry and hit-testing have independently gone GREEN.

Do not slice Python strings by cell offsets and do not search the whole
SVG/frame: wide cells can make string offsets wrong, and USER context prose
itself contains `Save` and `Edit & resend`.

- [ ] **Step 2: Write the failing containment/paint/hit-test matrix**

Parameterize the reported sizes `(200, 50)` and `(235, 52)`, `can_resend`
false/true, and each expected action. Push
`ConsoleEditMessageModal(content="Synthetic edit body", can_resend=...)` and
pause through layout. Use three separately collected parameterized tests so a
containment failure cannot prevent paint/hit evidence from being reported:

```python
containment = {
    "actions": actions.content_region.contains_region(button.region),
    "modal": modal_root.content_region.contains_region(button.region),
    "viewport": app.screen.region.contains_region(button.region),
}
assert containment == {"actions": True, "modal": True, "viewport": True}

hit, _ = app.screen.get_widget_at(*button.region.center)
assert hit is button

paint = {
    "label": expected_label in _painted_region_text(app, button),
    "visible": button in app.screen._compositor.visible_widgets,
}
assert paint == {"label": True, "visible": True}
```

Name the tests `real_bundle_action_containment`,
`real_bundle_action_hit_test`, and `real_bundle_action_painted_label`. Their
docstrings record why non-zero regions and whole-frame label searches are
insufficient. Preserve the baseline distinction in assertion messages: USER
paint/hit/containment are expected RED; non-USER paint/hit remain controls
while full containment may be RED by one row. Run without `-x` and retain the
full three-oracle failure list.

- [ ] **Step 3: Run RED and inspect the actual failure dimensions**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Chat/test_console_edit_message_modal.py -q \
  -k 'real_bundle_action_containment or real_bundle_action_hit_test or real_bundle_action_painted_label'
```

Expected: current fixed-height USER layout fails paint/hit/containment; non-USER paint/hit controls remain green, with any one-row containment failure recorded rather than hidden. If the test errors instead of reaching these assertions, fix the harness and rerun until the intended RED is observed.

- [ ] **Step 4: Commit the honest RED tests**

```bash
git add Tests/Chat/test_console_edit_message_modal.py
git commit -m "test(console): reproduce edit-modal action clipping"
```

### Task 2: Correct only the inner height allocation

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_edit_message_modal.py:65-68`
- Test: `Tests/Chat/test_console_edit_message_modal.py`

- [ ] **Step 1: Replace the fixed editor height with remaining space**

Change only the body sizing rule:

```tcss
#console-edit-message-body {
    width: 100%;
    height: 1fr;
    min-height: 8;
}
```

Do not change the 92x28 modal, context/error/action rows, margins, copy, variants, DOM, bindings, or handlers.

- [ ] **Step 2: Run the containment and hit-test matrix to GREEN**

Run only `real_bundle_action_containment or real_bundle_action_hit_test`.
Expected: hit-test and full containment pass for both shapes and both sizes.
The already-written `real_bundle_action_painted_label` oracle is evaluated
independently with contrast/focus in Task 3, so a residual paint RED remains
reachable rather than blocking the evidence-gated fallback.

- [ ] **Step 3: Run incumbent behavior and keystroke tests**

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_edit_message_modal.py \
  Tests/UI/test_console_edit_modal_keystroke_guard.py \
  Tests/UI/test_console_edit_resend_wiring.py -q \
  -k 'not real_bundle_action_painted_label and not real_bundle_action_ordinary_contrast and not real_bundle_focus and not enter_activates_focused_action'
```

Expected: PASS; Save, Cancel, resend, stale-key protection, and USER/non-USER wiring are unchanged.

- [ ] **Step 4: Mutation-prove the layout regression with a fresh bytecode root**

Create `MUTATION_ROOT=$(mktemp -d)`. Temporarily restore `height: 16` and
remove `min-height: 8`; run the containment and hit-test selections in a new
process with `PYTHONPYCACHEPREFIX="$MUTATION_ROOT/pycache"` and `-B`. Expected:
the named USER hit/containment assertions fail and every applicable non-USER
containment assertion fails. Restore `1fr`/`8`, use a second empty pycache
directory, rerun those two selections without `-x`, and require GREEN. Capture
the complete failure list so no sequential assertion can mask another oracle.
Task 3 repeats the fixed-height mutation against all three final oracles after
the paint decision. Remove `MUTATION_ROOT` after confirming the source diff is
restored.

- [ ] **Step 5: Commit the minimal production correction**

```bash
git add tldw_chatbook/Widgets/Console/console_edit_message_modal.py
git commit -m "fix(console): keep edit-modal actions visible"
```

### Task 3: Evidence-gate any button styling

**Files:**
- Test: `Tests/Chat/test_console_edit_message_modal.py`
- Modify only if RED remains: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate only if source CSS changes: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Add the independent focused-cell and keyboard contract**

After containment is GREEN, adapt `_relative_luminance`, `_contrast`, and the
label-segment style resolver from
`Tests/UI/test_model_artifact_widgets.py:591-625`. Run the already-written
`real_bundle_action_painted_label` oracle and add a
`real_bundle_action_ordinary_contrast` oracle requiring the actual composited
label foreground/background to reach 3:1.

Then add `real_bundle_focus` tests. For each action in DOM order and each
shape/size, first record its unfocused cropped label segments. Focus the
button, await refresh, crop the same region again, and require all of the
following:

- the exact label survives in the focused region;
- focused label foreground/background contrast is at least 3:1;
- the focused label cells differ from the ordinary label cells in their Rich
  style;
- the focused state has a focus-specific non-color cue that the ordinary state
  did not have: underlined label segments or a newly painted outline/edge row;
- `modal.focused` follows Cancel, Save, Edit & resend for USER and Cancel, Save
  for non-USER when the pilot presses Tab.

Add an explicit mounted keyboard behavior test that focuses Save and presses
Enter, then asserts the callback receives
`ConsoleEditResult(text="edited", resend=False)`. Existing mouse-click tests
remain the mouse control.

- [ ] **Step 2: Run the independent focus RED/GREEN decision**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Chat/test_console_edit_message_modal.py -q \
  -k 'real_bundle_action_painted_label or real_bundle_action_ordinary_contrast or real_bundle_focus or enter_activates_focused_action'
```

If it is already GREEN with surviving ordinary/focused labels, >=3:1 ordinary
and focused contrast, and a focus-specific non-color cue, make no CSS source or
bundle change and record that result in the task notes. If ordinary paint or
focus is RED, retain the exact failing compositor cells as evidence before
changing CSS.

- [ ] **Step 3: If and only if a separate focus/paint RED remains, add the scoped escape**

Use selectors descended from `#console-edit-message-actions`; do not touch global `Button`. Reuse design-system tokens and require the focused rule to preserve the label:

```tcss
#console-edit-message-actions Button:focus {
    background: $ds-focus-bg;
    color: $ds-focus-fg;
    outline: heavy $ds-action-focus;
    text-style: bold underline;
}
```

Add ordinary surface/edge declarations only for a separately observed ordinary-paint RED. Keep primary/default variants distinct.

- [ ] **Step 4: Rebuild and verify bundle parity only when CSS source changed**

```bash
../../.venv/bin/python tldw_chatbook/css/build_css.py
../../.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
```

Do not hand-edit the generated bundle. Re-run
`real_bundle_action_painted_label or real_bundle_action_ordinary_contrast or
real_bundle_focus or enter_activates_focused_action`, plus the containment and
hit-test controls.

- [ ] **Step 5: Mutation-prove final layout paint and focus independently**

First, with the final CSS decision in place, temporarily restore the editor's
fixed `height: 16`, run all three Task 1 oracles with a fresh pycache root, and
require USER paint/hit/containment plus every applicable non-USER containment
assertion to fail. Restore flexible sizing and require all three GREEN.

Then prove the focus oracle is non-vacuous whether it relies on incumbent
global CSS or a new scoped rule:

1. Back up `_agentic_terminal.tcss`, `_buttons.tcss`, and the generated bundle
   under one `mktemp -d` root.
2. Whether a scoped rule was retained or incumbent global CSS sufficed,
   temporarily append a later modal-scoped `Button:focus` override to
   `_agentic_terminal.tcss` that keeps `$ds-focus-bg`/`$ds-focus-fg` but
   explicitly sets **both** `outline: none` and `text-style: none`. This beats
   the scoped and global rules and neutralizes every non-color cue without
   removing the label or focus palette.
3. Run `build_css.py`, then `check_bundle_sync.py`, then `-k real_bundle_focus`.
   Expected: the non-color-cue assertion fails while the label remains painted.
4. Restore the source module from the backup, rebuild the bundle, run the sync
   checker, and require the focus selection GREEN.

For every retained ordinary paint declaration, remove it, rebuild, and require
`real_bundle_action_painted_label or real_bundle_action_ordinary_contrast` to
fail for paint/contrast; then restore, rebuild, sync, and require GREEN. If any
removal stays green, delete that declaration as YAGNI. Finish by checking that
the source modules and generated bundle contain no mutation-only diff.

- [ ] **Step 6: Commit only a substantive CSS correction**

If source CSS was required:

```bash
git add tldw_chatbook/css/components/_agentic_terminal.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  Tests/Chat/test_console_edit_message_modal.py
git commit -m "fix(console): clarify edit-modal button focus"
```

If no CSS was required, commit only the substantive Task 3 test additions:

```bash
git add Tests/Chat/test_console_edit_message_modal.py
git commit -m "test(console): verify edit-modal keyboard focus"
```

Do not create an empty CSS commit.

### Task 4: Verify tmux and non-tmux terminal drivers

**Files:**
- Temporary: `$LIVE_ROOT/task2703_console_edit_live.py`
- Temporary: `$LIVE_ROOT/task2703_console_edit_live.exp`
- Evidence: `.superpowers/sdd/2026-08-13-task-2703-console-edit-modal-paint/live/`
  (covered by the repository's existing `.superpowers/` ignore rule)

- [ ] **Step 1: Create one bounded scratch root and fingerprint real state**

Create `LIVE_ROOT=$(mktemp -d /tmp/task2703-live.XXXXXX)` and put the harness,
config, data, cache, captures, and JSON evidence beneath it. Before any
application import, record path, existence, mode, size, mtime, and recursive
SHA-256 manifest for these real locations:

- `$HOME/.config/tldw_cli/config.toml`;
- `$HOME/.config/tldw_cli`;
- `$HOME/.local/share/tldw_cli`;
- `$HOME/.cache/tldw_cli`;
- `$HOME/Library/Application Support/tldw_cli` and
  `$HOME/Library/Caches/tldw_cli` when present.

Every launch command must export, before Python starts,
`HOME=$LIVE_ROOT/home`, `XDG_CONFIG_HOME=$LIVE_ROOT/xdg-config`,
`XDG_DATA_HOME=$LIVE_ROOT/xdg-data`,
`XDG_CACHE_HOME=$LIVE_ROOT/xdg-cache`, and
`TLDW_CONFIG_PATH=$LIVE_ROOT/xdg-config/tldw_cli/config.toml`. Pre-create that
TOML with `[paths].data_dir = "$LIVE_ROOT/data"` and model-catalog/network
refresh disabled. Use only `TASK2703 SYNTHETIC USER BODY` and
`TASK2703 SYNTHETIC ASSISTANT BODY`.

- [ ] **Step 2: Create the complete temporary modal/live-cell harness**

Write `$LIVE_ROOT/task2703_console_edit_live.py`. It loads the real generated
bundle, pushes USER or non-USER `ConsoleEditMessageModal`, and accepts
`--shape`, `--evidence`, `--ready`, `--ack-dir`, and `--verify` arguments. A
temporary system-priority F12 binding increments a snapshot sequence and uses
`call_after_refresh` to record one JSONL snapshot containing that sequence,
terminal size, shape, focused widget id, each button's region and label-region
cropped Rich segments (`text`, foreground, background, bold, underline),
containment booleans, and center hit-test widget id. Only after the refreshed
snapshot is fsynced and atomically replaced does it create
`$ACK_DIR/<sequence>`. After the modal's first complete layout, the harness
atomically touches `--ready`; F12 never doubles as the readiness signal. A
system-priority F10 binding exits with status zero. The snapshot code uses the
same cell-cropping/contrast helpers as the pytest oracle. It adds no production
API and writes no message body.

`--verify EVIDENCE_DIR` must fail unless every expected driver/size/shape/focus
record exists, every target label is present in its own cropped cells, all
three containment booleans and hit tests are true, ordinary and focused label
contrast are >=3:1, and each focused record has the focus palette plus the
same focus-specific non-color cue proven in Task 3. This JSON verifier is the
targeted-cell oracle for both drivers; raw SGR captures are retained as
secondary proof that the real terminal rendered the states.

- [ ] **Step 3: Drive the harness in tmux with exact captures**

For each `(200x50, 235x52) × (user, nonuser)`, launch the harness under a
private `tmux -L task2703-$PPID` socket with the complete scratch environment.
Wait for the ready file. Send F12, wait for the exact next sequence's ack, and use
`tmux capture-pane -ep -t SESSION -S 0 -E <last-row>` to write the ordinary
SGR capture. Send Tab one key at a time and F12 after each; wait for that F12's
ack before reading the matching JSON sequence, capturing, or sending another
key. Stop after the bounded sequence reaches Cancel → Save → Edit & resend for
USER or Cancel → Save for non-USER. Capture `-ep` after each action focus. Use
`tmux display-message -p '#{pane_width}x#{pane_height}'` to prove dimensions,
then send F10 and destroy the private socket. No grep/byte offset is used to
infer button columns.

- [ ] **Step 4: Drive the same harness through a direct Expect PTY**

Write `$LIVE_ROOT/task2703_console_edit_live.exp` with parameters for rows,
columns, shape, evidence path, and ready path. Before `spawn`, call
`stty rows $rows columns $columns`; use `log_file -noappend` to retain the raw
ANSI/SGR stream; spawn `env` with the same five scratch environment variables and
`../../.venv/bin/python $LIVE_ROOT/task2703_console_edit_live.py ...` directly
(no tmux and no `script`). Wait for the ready file, send F12 as `\033\[24~`,
wait for that exact snapshot ack, then send `\t` one key at a time followed by
F12. After each F12, use `exec` to poll only the matching ack file before
reading its JSON sequence or sending the next key. Stop at the same bounded
USER/non-USER focus sequence. Send F10 as `\033\[21~` and require EOF with exit
status zero. Repeat both shapes and both dimensions.

Run the harness verifier over the combined tmux/Expect evidence and require
PASS. The direct PTY's targeted cells come from compositor JSON recorded only
after actual Expect keystrokes; its `log_file` proves the corresponding raw
terminal stream retained SGR rather than being a headless-only capture.

- [ ] **Step 5: Prove isolation and remove temporary code**

Regenerate the real-path manifest and require byte-for-byte equality with the
pre-run manifest. Copy only privacy-safe JSON/SGR captures and their SHA-256
manifest to
`.superpowers/sdd/2026-08-13-task-2703-console-edit-modal-paint/live/`; prove
that directory is ignored with `git check-ignore`. Remove the scratch
config/data/cache, temporary harness, Expect script, private tmux socket, and
any session. Confirm no `/tmp/task2703-live.*` runtime directory remains.

### Task 5: Documentation and task closeout

**Files:**
- Modify: `Docs/User_Guide/console/branching-and-rewind.md:160-164`
- Modify: `backlog/tasks/task-2703 - Console-Edit-Message-modal-action-buttons-invisible-in-real-terminals.md`

- [x] **Step 1: Remove the obsolete workaround**

Delete only the bullet claiming the Edit Message actions may be invisible and advising blind Tab/Enter operation. Do not rewrite unrelated Console guidance.

- [x] **Step 2: Run the final bounded behavior matrix**

```bash
../../.venv/bin/python -m pytest \
  Tests/Chat/test_console_edit_message_modal.py \
  Tests/UI/test_console_edit_modal_keystroke_guard.py \
  Tests/UI/test_console_edit_resend_wiring.py \
  Tests/integration/test_console_edit_resend_e2e.py -q
```

Run the exact native Console nodes rather than the whole monolithic file:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_native_chat_flow.py::test_console_selected_message_edit_action_opens_modal_and_saves_content \
  Tests/UI/test_console_native_chat_flow.py::test_console_edit_resend_clears_replaced_descendant_original_attempt \
  Tests/UI/test_console_native_chat_flow.py::test_console_selected_message_edit_action_cancel_preserves_content \
  Tests/UI/test_console_native_chat_flow.py::test_console_selected_message_edit_action_blank_save_stays_open_with_error
```

- [x] **Step 3: Run static and generated-artifact checks**

```bash
../../.venv/bin/python -m ruff check \
  Tests/Chat/test_console_edit_message_modal.py \
  tldw_chatbook/Widgets/Console/console_edit_message_modal.py
../../.venv/bin/python -m ruff format --check \
  Tests/Chat/test_console_edit_message_modal.py \
  tldw_chatbook/Widgets/Console/console_edit_message_modal.py
../../.venv/bin/python -m py_compile \
  tldw_chatbook/Widgets/Console/console_edit_message_modal.py
../../.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
git diff --check origin/dev...HEAD
```

If a whole-file formatter failure is inherited, prove it against `origin/dev` and range-check every changed hunk; do not accept unrelated formatter churn.

- [x] **Step 4: Run UI hardening and bounded code review**

Use the Impeccable hardening detector/review on the exact changed production
and test files, then request a bounded correctness/accessibility review. Resolve
TASK-2703 findings; document proven inherited findings without expanding
scope. Commit any substantive review correction separately. Any production,
CSS, or test edit from review invalidates prior evidence: rerun all final Task 1
containment/hit/paint oracles, the Task 3 ordinary-contrast/focus/Enter
selection, both Task 4 live drivers plus isolation manifests, and the Task 5
bounded behavior/native/static gates before proceeding.

- [x] **Step 5: Self-review cumulative scope and make the lessons decision**

Review `git diff --stat`, `git diff --check origin/dev...HEAD`, and the complete
`origin/dev...HEAD` diff. Confirm no outer-modal, copy, DOM, handler, global
Button, dependency, config, or logging change slipped in. Explicitly decide
whether the USER-overflow diagnosis or live-driver technique generalizes into
`backlog/docs/lessons-testing-evidence.md` or
`backlog/docs/lessons-live-verification.md`; add a concise incident-based lesson
only if it adds knowledge not already recorded.

- [x] **Step 6: Freeze the reviewed candidate and apply the user-directed scoped gate**

The reviewed candidate was frozen at `8c450c8b4`. The user explicitly limited
this PR's verification to the touched modal and its behavior/wiring surfaces,
superseding the originally planned full-suite gate. The full suite had reached
28% when that direction arrived; it was terminated, is non-authoritative for
TASK-2703, and its unrelated failures belong in a separate follow-up PR from
the latest `dev`.

The scoped matrix in Step 2 produced 68 passes and one inherited integration
failure. The exact failing node,
`Tests/integration/test_console_edit_resend_e2e.py::test_console_edit_and_resend_full_lifecycle_persist_resume_swipe`,
failed identically at the pre-TASK-2703 base `0d718e7fb`; the user explicitly
approved closing TASK-2703 with that exception. The exact native nodes passed
4/4. The owned 61-test modal file, mutation evidence, Task 4 live matrix,
static gates, and independent reviews complete the approved scoped gate.

- [x] **Step 7: Complete task hygiene only after approved scoped evidence is green**

Use Backlog CLI to check all four ACs, add concise Implementation Notes
containing RED/GREEN, mouse and Enter behavior, live-driver, isolation, scoped
verification, static, review, lessons, and ADR evidence, then set TASK-2703 Done.
Re-read with:

```bash
backlog task 2703 --plain
```

Confirm all ACs render checked and the plan/notes survived the CLI update.

- [x] **Step 8: Commit docs and closeout**

If Step 5 added a lessons entry, first stage that exact lessons file with
`git add backlog/docs/lessons-testing-evidence.md` or
`git add backlog/docs/lessons-live-verification.md`; otherwise do not touch
either lessons file. Then run:

```bash
git add Docs/User_Guide/console/branching-and-rewind.md \
  'backlog/tasks/task-2703 - Console-Edit-Message-modal-action-buttons-invisible-in-real-terminals.md' \
  Docs/superpowers/plans/2026-08-13-task-2703-console-edit-modal-paint.md
git commit -m "docs(console): complete TASK-2703"
```

- [x] **Step 9: Final branch verification**

Re-run `git diff --check origin/dev...HEAD`, verify `git status --short` is
clean, and audit the cumulative diff for scope. Do not push/create a PR until
the approved scoped verification and review evidence above are complete;
PR/rebase/Qodo handling remains a later user-authorized integration step.
