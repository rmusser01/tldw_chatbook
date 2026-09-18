# Library Skills browsing and editor review — TASK-32646

This pass repairs keyboard return from Save, Back, Discard and new-draft
Cancel. Editor exits reveal the Skills list even after a manual collapse.
Cancel uses the shared route transition, so the rail and content agree.
A pending row handoff waits for loading and the canvas’s queued rebuild to finish;
it uses the filter only when the settled result is empty.

A completed write preserves text or invocation changes entered during I/O as
an unsaved draft, adopting the committed version and trust metadata for the next
Save. Save focus moves to the available lifecycle action only while the outgoing
Save still owns focus (or its removal cleared focus). The name-shadow warning
now covers twenty newer runtime tools and Console commands reported by the
existing four-source drift guard. No tokens, stylesheet values, storage schema,
trust authority or execution permissions changed.

## Verification

**277 targeted checks pass**, recorded in [test results](test-results.json). The selection covers the
new production-CSS journeys, existing Skills canvas/reader cases, state and local
service behavior, controller wiring, token governance and CSS bundle reproduction.
No full repository suite was run.

The five new journeys cover both themes at 170×48 and 80×24, literal Unicode and
markup-like text, Basic/Advanced and invocation changes without field replacement,
exact duplicate/unavailable allowlist entries, held writes, dirty Escape, saved
content after Discard, a forced loading/ready recomposition race after Back, and a held browse reload
after Cancel. The late-edit case
uses actual keyboard input during the write, verifies the first saved description,
then saves the remaining draft at version 3. Fresh-profile admission is enabled
in the main journey. Setup and most draft values are assigned programmatically;
actions use focused keyboard activation.

Nine initial canvas assertions were stale: the five-cell grip floor, layout
readiness before a programmatic grip press, CSS ownership/token spellings,
focus-aware footer hints, and a constructor-bypassing fake. These were corrected
to current contracts rather than changing the interface to satisfy old literals.
A separate state drift test exposed the missing reserved names and passed after
the warning set was updated.

New journey and runner files pass Ruff and formatting. Existing-file diagnostics
are compared against HEAD in [lint results](lint-results.json); unrelated baseline
warnings are retained. Historical comments in the changed controller paths were
shortened while keeping the documented invariants. The inherited Library screen
size ratchet remains above its existing budget (35,203 lines against 33,204;
HEAD already had 35,204); this pass does not increase its
baseline size or raise the budget. Controller size is within its existing budget.

Independent review reproduced late-edit data loss, manually collapsed returns,
ready-result/DOM timing and zero-width compact layout. The fixes and the
second-save version handoff received follow-up review.

## Native evidence

The [runner](native_check.py) uses actual TldwCli with LinuxDriver in an exclusive
private profile. Both 170×48 dark and 80×24 light exercise Overview → Edit,
Basic/Advanced, save, Back, dirty Escape, Discard and new-draft Cancel. Browse
selection and fixture creation use the normal local service/route helpers;
control actions use key events. The checked allowlist preserves duplicate
`fs_read` and unavailable `unknown_tool` entries exactly.

[Native results](native-results.json) record two committed version-2 Skills and
focus on a saved row after Cancel. [Read-only persistence](persistence.json)
confirms both SKILL.md bodies and allowlists, absence of the cancelled draft,
SQLite integrity and zero messages. Normal Ctrl+Q returned exit 0 to the observed
shell, which was then closed. Trust setup, approval, import, provider execution and restart qualification
are outside this native journey.

| State | Wide dark | Compact light |
| --- | --- | --- |
| Saved | [Back focused](saved-170.svg) | [Back focused](saved-80.svg) |
| Cancelled new draft | [Browse row](cancelled-170.svg) | [Browse row](cancelled-80.svg) |
| Additional compact states | — | [Advanced](advanced-80.svg), [Discarded](discarded-80.svg) |

Six rendered captures were inspected. Compact fields scroll; long row names
truncate within the list pane. The existing dirty-veto notification may remain
visible briefly during later actions, including Advanced and Discard/Cancel,
without covering the focused action or row. Earlier
failed probes are not qualification: they exposed hidden Save focus, collapsed
returns and Cancel's route/focus timing. Two failed probes required termination
after normal quit did not finish. The final successful profile is independent
and uses the committed runner without diagnostic focus wrappers.

ADR required: no. Existing ADR-009, ADR-076, ADR-086, ADR-150 and ADR-161 apply.
Next: Skills import and trust journeys. Integration into dev remains pending.
