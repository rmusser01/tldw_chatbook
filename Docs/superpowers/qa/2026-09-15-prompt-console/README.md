# Prompt Use in Console review — TASK-32638

Direct insertion now decodes escaped braces even when no variable dialog is
needed. `{{name}}` inserts literal `{name}` from Library, Console's `/prompt`
command and its picker. Both paths render the already compiled plan once.
The explicit **Use original placeholders** action still inserts the exact
source. The System checkbox now paints only its glyph beside the full label,
and Console refreshes its System status chip after authorized replacement.
No CSS, token, handoff lifetime, source-record or System authority contract changed.

## Targeted verification

**369 distinct targeted checks passed.** Commands use `.venv/bin/python -m pytest`, with
`-q --disable-warnings --tb=short --show-capture=no`:

| Selection | Result |
| --- | --- |
| `Tests/UI/test_library_prompt_console_journeys.py Tests/UI/test_console_prompts_controller.py Tests/UI/test_prompt_variables_dialog.py Tests/State/test_pending_handoff_store.py` | 168 passed before paint follow-up |
| `Tests/UI/test_prompt_variables_dialog.py Tests/UI/test_console_prompts_controller.py` after paint fixes | 63 passed, including 61 repeated cases and two new status-chip cases |
| `Tests/Prompt_Management/test_prompt_variables.py Tests/UI/test_console_command_composer.py -k 'prompt'` | 118 passed, 74 deselected |
| `Tests/UI/test_design_token_governance.py Tests/UI/test_css_bundle_sync_guard.py Tests/Architecture/test_library_prompts_wiring.py Tests/UI/test_library_prompts_canvas.py -k 'governance or bundle or wiring or insert_console or use_recipe or use_in_console or dialog_rechecks_projection'` | 37 passed, 331 deselected |
| `Tests/UI/test_library_prompt_console_journeys.py Tests/UI/test_design_token_governance.py Tests/UI/test_css_bundle_sync_guard.py Tests/Architecture/test_library_prompts_wiring.py` after paint fixes | 33 repeated cases passed |
| `Tests/UI/test_console_controller_wiring.py` | 44 passed |

Four escape regressions failed before the two insertion fixes, covering
Library and Console direct insertion. Four additional paint cases reproduced
the duplicate checkbox letter and stale System chip before their repairs. Seven new Library journeys use production
CSS and real SQLite, exercising the header, Cancel, System checkbox toggles,
retained values, Apply and original-source keyboard actions. Dialog journeys
run at 170×48 and 80×24 in both themes; direct insertion has three text cases.
These tests assign values programmatically and observe staged payloads. The
native check separately exercises the actual destination and consumer.

Existing tests cover missing/stale targets, unavailable staging, expiry,
one-shot claims, transient composer remount, draft preservation, guarded System
changes, failure recovery and detached Recipe conversion. Source records remain
unchanged. Two existing structural assertions were stale at HEAD: one counted
physical signature lines, the other required history to belong to the former
controller. The replacement checks one delegation statement to the current
owner and requires `await` for asynchronous forwarders.

New journey/runner files pass Ruff and formatting. The changed ranges in the
existing test are formatted; [baseline comparison](lint-results.json) records
no new diagnostics in existing files. No full repository suite was run. An
independent code review identified the AST helper's missing Await handling;
the correction passed all six delegation cases. The follow-up paint fixes also
received an independent review with no outstanding findings.

## Native verification

The [runner](native_check.py) uses actual TldwCli and LinuxDriver with an
exclusive synthetic profile in an owned tmux session. Saved Prompt fixtures are
seeded through the real local database, then opened in Library. Browse selection
is setup; the header, dialog and navigation actions use native key events.
Variable values are assigned programmatically. No Send action is invoked.

The journey checks existing-draft append, escaped literals, Cancel, empty values
on reopen, explicit System opt-in, single-pass value rendering, exact original
source, composer focus, and no duplicate insert after another navigation.
[Results](native-results.json) and [read-only persistence](persistence.json)
record the observed states. All four source Prompts remain at version 1 with
unchanged source text, and the message table remains empty. Native Ctrl+Q
returned exit 0 to an observed zsh shell. Live session System replacement is qualified;
provider execution and durable System restoration after restart are not.
The owned terminal session was closed after the final exit check.

The six initial renders exposed the duplicate checkbox letter and stale chip.
One confirmation batch inspected the four corrected dialog/applied captures.
At 80×24, variable entry scrolls while the three actions stay visible; the
System status chip is beyond the initial horizontal status-strip viewport.
Transient context-rail notices were allowed to expire before final captures.

| State | Wide dark | Compact light |
| --- | --- | --- |
| Direct insertion | [Draft](direct-170.svg) | [Draft](direct-80.svg) |
| Authorized System and values | [Dialog](authorized-170.svg) | [Dialog](authorized-80.svg) |
| Applied values | [Draft](applied-170.svg) | [Draft](applied-80.svg) |

Early probe attempts failed because their setup raced Basic-mode field
replacement, omitted private storage directories, used the wrong browse-row
identity, or tried to focus the hidden compact rail. Those attempts are not
handoff qualification. Final evidence uses saved fixtures and the shared browse
selection entry point. Native fault injection and Recipe conversion are covered
by targeted tests, not by this native journey.

ADR required: no. Existing ADR-040, ADR-053, ADR-086, ADR-094, ADR-150 and
ADR-161 apply. Next component: Skills. Integration into dev remains pending.
