# Audit navigation ownership — TASK-32837 / PR2724

Audit's Open tool and Adjust permission buttons retain the tool/profile identity
shown when they were rendered. Retired controls are invalidated before async
pruning. Held events are ignored if their control or owning view has become
hidden, invisible, disabled or covered by another screen. Current controls remain
repeatable. Both destinations honor failed row selection, clear stale detail and
show the existing unavailable-tool warning.

Rebased onto merged dev `29b0a31df4701160a3c805e1bf490c76b9353964` (PR2740).
The [conflict review](CURRENT-DEV-REVIEW.md) records both documentation choices;
application source merged automatically. No CSS, token values, permission policy,
storage or runtime authority changed. Existing ADR-150/161 apply; no new ADR.

## Current verification

[145 distinct targeted checks pass](current-dev/qualified-cases.json):
30 Audit navigation cases, 23 adjacent permission-navigation cases, 24 existing
inspector/workbench/table cases, 26 design/component governance cases and 42
native-runner argument/path checks. No full suite ran.

The [12-case red run](current-dev/red-tests.txt) reproduced both actions navigating
from unavailable controls/views. The [95-case green run](current-dev/green-tests.txt)
includes all 30 Audit cases, queued replacement/clear, retirement before pending
pruning, immutable identity, repeatability and both filtered destination routes.

The [adjacent run](current-dev/adjacent-tests.txt) passed 47 cases; three older
inspector cases stopped during fixture setup with `raw_source_selection_changed`.
Their per-test fixture changed the profile after collection had bound the config
source (the documented TASK-32749 issue). Applying the existing private-profile
process wrapper preserves their assertions and [all three pass](current-dev/setup-fix-tests.txt).
[Log export hashes](current-dev/log-export-manifest.json) record trailing-whitespace
normalization of saved pytest output. No production recovery guard changed.
Cleanup warnings concern unrelated old
pytest temporary folders, which were left untouched.

All [seven preflight guards](current-dev/preflight.txt) pass. Ruff adds
[no diagnostics](current-dev/static-analysis.json), new files and changed ranges
[pass formatting](current-dev/formatting.txt), and [independent read-only review](current-dev/independent-review.txt)
found no blockers, including the fixture follow-up.

## Current native evidence

The [24-capture gallery](GALLERY.md) covers both focused Audit controls, Tools and
Permissions destinations after clearing a nonmatching filter, and each action's
missing-tool warning. Dark/light themes run at 120×40 and 170×48. Every activated
button is focused, fully inside its compositor clip, paints its complete label
and owns the center hit target. The normal Enter handlers perform navigation.
All 24 screenshots were rendered and inspected. Existing Audit filter layout
and inspector guidance are visible; their separate PRs remain outside this repair.

The [native receipt](current-dev/native/result.json) records the real TldwCli with
LinuxDriver and TTY streams, a fresh validated private profile, two synthetic
metadata records, correct destination row identities, unchanged permission
profiles and zero network attempts. No tool executes or external server connects.
Destination loss during an await is covered by deterministic tests rather than
a native timing race.

[Lifecycle verification](current-dev/lifecycle.json) confirms exit 0, normal app
return, absent process, released instance lock, ten healthy private databases,
zero conversations/messages, unchanged default-profile files, no app errors and
empty faulthandler output. All eleven source hashes and both runner/journey hashes
match. [Native export hashes](current-dev/native-export-manifest.json) record only
trailing-whitespace normalization of SVG exports; visual content is unchanged.

## History and bounds

The original `native/`, `tests/`, lifecycle and verification receipts in this
folder are historical evidence from commit `b0eca0deb0`; they do not qualify the
current source. The runner was refreshed for the supported terminal warm-up API,
shared argument validation and network guard. Current receipts are exclusively
under `current-dev/`; [current export hashes](current-dev/export-manifest.json)
cover those files, the runners and review documents.

Audit selection PR2720, filter layout PR2721, inspector guidance PR2722 and
same-ID catalog freshness PR2726 remain separate. Compact 80×24 inspector
reachability, connected-runtime journeys and the wider component review remain
open. PR2724 stays draft pending current-head CI/review and its own final owner
visual approval. PR2740 approval does not authorize this merge.
