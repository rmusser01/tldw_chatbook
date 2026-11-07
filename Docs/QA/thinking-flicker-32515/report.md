# TASK-32522: streaming thinking flicker verification

The Console now retains the expanded thinking text widget and updates its
literal content in place. This removes the empty frame between removing an old
body and mounting its replacement. Disclosure ownership, wrapping, lazy body
mounting, and automatic/manual collapse semantics remain governed by ADR-090.

## Automated evidence

- The original renderer failed the body-identity regression and painted blank
  thinking frames at both 60 and 100 columns. Repeating the negative control
  with the corrected production stylesheet stack recorded 16 and 15 blank
  frames respectively. The negative control loaded the original renderer in
  the test process without rewriting the working tree.
- The fixed renderer passed all **74 tests** across
  `test_console_thinking_disclosures.py`, `test_console_thinking_edit_wiring.py`,
  and `test_console_assistant_turn.py`. Coverage includes live identity and
  painted continuity, focus, wrapped growth, disclosure transitions, terminal
  failures, proprietary notices, saved thinking edits, and themed geometry.
- The styled harnesses had omitted the split Console stylesheets. The original
  renderer also had 32 resulting geometry/paint failures; using the existing
  shared production stylesheet helpers made the checks representative again.
- Both changed test files pass Ruff and whole-file formatting. The production
  edit passes range formatting and introduces no Ruff diagnostics. The large
  existing transcript module retains its **27 pre-existing Ruff diagnostics**;
  whole-file lint/format cleanup was outside this rendering fix.
- Independent read-only review found no actionable correctness issues.

Targeted command:

```sh
.venv/bin/python -m pytest Tests/UI/test_console_thinking_disclosures.py Tests/UI/test_console_thinking_edit_wiring.py Tests/UI/test_console_assistant_turn.py -q --tb=short --show-capture=no
```

The full test suite was not run. The environment emitted its existing
requests-dependency warning and cleanup warnings for an unrelated old pytest
temporary directory.

## Native UAT

On September 12, 2026 (America/Los_Angeles), the real `TldwCli` ran in a
140×45 native terminal on a dedicated tmux socket, with a disposable profile,
real SQLite persistence, the normal Console composer/Enter send path, and the
local llama.cpp server at `127.0.0.1:9099`. The loaded model was Gemma 4 26B A4B.
No provider or transcript renderer was mocked. A pass-through observer recorded
the actual compositor display callbacks during live expanded thinking; deliberate
collapse/reopen transitions were excluded from streaming-blank counts.

| Scenario | Live frames | Empty body frames | Blank painted frames | Body replacements | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| Automatic disclosure, then answer | 173 | 0 | 0 | 0 | Reply complete; auto-collapsed; reopened full thinking |
| Manual collapse/reopen during thinking | 42 | 0 | 0 | 0 | Reply complete; manual expansion preserved |

Both actual answers completed: `2 + 2 is 4.` and
`7 * 8 is 56. The answer is 56.` The first turn provided 92 observed thinking
text lengths, and the second provided 26. The total was **215 observed live
frames without blanking or replacement**. Native terminal captures and exported
screens were inspected for readable thinking, wrapping, and the final answers.

An earlier longer arithmetic prompt produced 458 clean thinking frames but
ended in a provider HTTP 502 before an answer. That run was not counted as a
successful end-to-end UAT. The two shorter prompts above then completed normally.

Evidence:

- [Live UAT measurements](live-uat.json)
- [Thinking while streaming](turn-1-thinking.svg)
- [Automatically collapsed thinking and answer](turn-1-answer.svg)
- [Full thinking reopened after the answer](turn-1-reopened.svg)
- [Manual expansion retained with the second answer](turn-2-answer.svg)
- [Targeted test results](targeted-tests.xml)
- [Negative-control regression results](regression-red.xml)

The native UAT launcher and isolated profiles were disposable diagnostics under
`/private/tmp/thinking-flicker-uat`; the normal user profile was not used.

PR preparation reran all 74 targeted tests on fresh `origin/dev` base
`a3142cb356`; all passed. [Dev-based test results](pr-tests.xml).
The task was renumbered to TASK-32522 because dev already owned TASK-32515;
the evidence directory retains its original name for link continuity.
