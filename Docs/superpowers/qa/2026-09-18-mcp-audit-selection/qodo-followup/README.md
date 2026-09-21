# Audit selection: Qodo follow-up

Qodo reviewed `919ae96fb58a7546ec8cc92e6e7b5d097c1d921c` with zero bugs and
three rule findings. Nullable selection arguments now have Google-style Args
documentation. The existing duplicated old/new uniqueness calculation is shared
by cursor and selection restoration, with eight isolated identity/key-retirement
cases alongside the fifteen mounted selection regressions. Findings routing is
unchanged; its docstring now accurately distinguishes its cached index contract.

The request to move all QA runtime imports to module scope was declined with a
[technical reply](https://github.com/rmusser01/tldw_chatbook/pull/2720#discussion_r4063218793).
Checkout pinning, CLI/private-profile admission, environment selection and the
network guard must run before application imports. The required stdlib,
third-party, local ordering is retained within that intentional boundary.
Extracting another runtime module would add structure without repairing a defect.
See the recorded checkout-import incident in `backlog/docs/lessons-live-verification.md`.

Verification: **37 focused tests pass** (8 isolated cases, 15 mounted selection
regressions, 14 architecture checks), all nine artifact guards pass, and no Ruff
findings are introduced. New tests and modified production ranges pass formatting
checks. The meaningful red run was seven missing-helper failures plus one passing
key-retirement characterization. An earlier collection attempt lacked the repo's
required private-profile declaration and is not counted as a behavioral red run.
Independent review found no actionable issues. No full suite ran.

A fresh real-TTY TldwCli replay covers dark/light at 80x24 and 170x48, twenty
captures, two real local stdio catalogs, synthetic JSONL executions, and no tool
calls or external network. All twenty terminal captures match the owner's approved
content after fixture timestamp normalization. Seventeen SVGs match after Rich ID
normalization; the remaining three differ only in the input caret blink, including
one zero-width rectangle. Their retained SVGs were rendered and visually inspected.
The approved layout and behavior remain unchanged; source hashes correctly record
the helper/docstring follow-up rather than claiming unchanged production bytes.

Normal app and fixture exits, lock release, ten healthy databases, zero chats and
messages, unchanged defaults/sentinels, and source provenance are verified in
`lifecycle.json`. The owned terminal was closed after process absence verification.
Current-head CI, final accumulated review and latest-dev review remain merge gates.
