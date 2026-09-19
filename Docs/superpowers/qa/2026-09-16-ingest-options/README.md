# Library Parakeet folder picker — TASK-32664

Reviewed on `feat/component-pattern-library`, based on `6144043362`.

Selecting a local Parakeet model folder rebuilt the entire import form. The
production-CSS journeys reproduced title cursor reset (4 → 13) and an 80×24
return focused on Browse at y=36, outside the viewport. The callback now updates
the existing folder Input, allowing the normal option event to refresh the
receipt, validation and gate while retaining the form and its editing context.
Cancel continues to preserve the draft and return to Browse. Selecting the same
folder explicitly refreshes the gate after disarming consent, so unchanged input
values cannot leave stale confirmation text or styling.

The earlier overflow report was a harness error. Its consolidated widget CSS
omitted app-tier utility classes. The [paired probe](css-probe.txt) shows the same
input resolving to `100w` without those sheets and `1fr` with them. Loading
APP_STYLESHEETS repairs the geometry test; no row CSS or token value changed.

ADR required: no. Existing
[014](../../../../backlog/decisions/014-library-ingest-service-authority-and-recovery.md),
[150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [161](../../../../backlog/decisions/161-component-pattern-library.md) apply.

## Evidence

[Verification commands](verification.json) record **279 distinct passing checks**:
208 neighboring canvas, structural, token/bundle and file-picker checks, plus
16 new journeys and 55 existing consent checks. The twelve full Library cases use 170×48/80×24 and both themes, real temporary
media SQLite, real preflight and the real directory picker. Four further cases
check disabled controls and painted missing-package explanations. Selection and
cancellation use keyboard activation; source and picker paths are assigned to
fixture inputs. Tab/Shift+Tab, visible return focus, immediate typing, unchanged
draft fields and title cursor are asserted. The same-folder cases use a deterministic tooling-warning fixture and exactly
one Start press to arm consent. No import is submitted.

[Static comparison](static-comparison.json) adds no Ruff diagnostics; inherited
existing-file diagnostics remain. New files pass full Ruff and formatting;
modified existing ranges are formatted. [Sizes](size-comparison.json) record the
small callback change with no raised budget. [Focused review](review.md) caught the same-folder edge and found no remaining
actionable findings after repair. No full suite was run. Two inherited
pytest temporary-directory cleanup warnings remain.

## Native inspection

[native_check.py](native_check.py) ran actual TldwCli/LinuxDriver with an
[exclusive private profile](isolation.json), private database paths and null
keyring. The real environment lacks `audio_processing` and `parakeet_onnx`.
The [unavailable capture](unavailable-170.svg) shows the real disabled state and
package reason. For enabled journeys, only the widget's availability probe is
simulated. The picker constructor starts the actual picker inside the private
fixture directory. Nothing installs or loads a model, transcribes audio, changes
Lab Models, or contacts a provider/server.

Seven run-002 SVGs were rendered and inspected together. Final run-003 repeats
the journey with same-folder consent coverage; its compact return was rendered
and inspected as the confirmation pass. Linked captures below are from run-003:

| State | 170×48 dark | 80×24 light |
|---|---|---|
| Browse focused | [capture](browse-170.svg) | [capture](browse-80.svg) |
| Real picker, Select focused | [capture](picker-170.svg) | [capture](picker-80.svg) |
| Selected folder, visible return | [capture](returned-170.svg) | [capture](returned-80.svg) |

Both sizes preserve title text, cursor, field identity, provider and staged source
through Select and Cancel. Keyboard re-entry changes the retained option.
[Results](result.json) and [read-only persistence](persistence.json) record ten
healthy SQLite databases, zero Media/messages/ingest jobs, an unchanged synthetic
source, an empty selected folder, no ERROR/CRITICAL log lines and normal exit 0.
Normal terminal Quit returned to the shell, which was observed before the owned
session was closed. Initial run-001 failed before UI startup because its
configured private database parent had not been created; run-002 creates it and
passes. Final run-003 also passes the same-folder confirmation regression at both
sizes. No production profile was used.

## Remaining review

At this checkpoint the compact picker capture had no visible folder rows despite
reporting one loaded entry; its typed-path field was also narrow. This pass
qualifies typed-path Select/Cancel and the caller's return. The subsequent
[TASK-32665 review](../2026-09-16-compact-picker/README.md) repairs and qualifies
compact listing/navigation, validation and resize continuity. Continue remaining
per-type options and queue activity/recovery. Actual import execution,
installation, remote/provider actions and restart remain outside this evidence.
No push or integration into `dev`.
