# Workflows authoring-only dev port

Status: Approved in conversation on 2026-09-14; implementation of the existing editor, not a new runtime design.
Task: TASK-32601.
ADR required: yes — amendment to existing ADR-138, not a new ADR number.
ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md.
Reason: user-approved stable-file exchange contract (2026-09-15) bounds this authoring subset; shared ADR-125 implementation and ADR-150 boundaries remain unchanged.

## Outcome

A real, usable Workflows destination on current dev: workflow library on the left,
step navigator beside it, and a linear overview or selected-step continuous form
with independently collapsed sections. Create, edit, save, import and export are
working local actions. Run is visibly unavailable for this delivery.

## Reuse and exclusions

Use the reviewed editor-only checkpoint b34eda3d64 for the UI and its document/draft
tests; inspect later source commits only for relevant authoring corrections.
The complete source branch remains preserved at 8aa1987a. The incomplete integration
branch remains parked at fe42f99353, including its uncommitted regression evidence.

Do not import runtime.py, runtime_lock.py, run_service.py, launch_intent.py,
adapters.py, admission.py, local_services.py, or run_setup.py. Do not modify shared
helper protocols, helper capacity, TTS ownership, permissions or provider transport.
No schema v5, PID recovery, execution-ownership worker, new dependency, server write,
sync protocol, or automatic execution is authorized by this port.

## Persistence and application lifecycle

Use current dev's registered connect_private_sqlite factory and normal serialized
SQLite transactions. Database setup/validation must not open ordinary raw-file
handles on live DB/sidecar inodes in the application process. File exchange
excludes those inodes under the stable-file contract below, not a race-free
actual-open guarantee. The workflow owner registration is an
additive domain entry, not another SQLite implementation. Keep the four existing
migration files byte-identical for existing workflow-store compatibility. Expose
authoring transactions only; do not expose execution admission or acquire a
.runtime.lock. Existing runtime rows remain untouched by authoring.

The application owns the document/draft services across screen replacement.
Initialize lazily off the UI thread and retain accepted setup/save operations
until their completion. Flush drafts before switching definitions, navigation
and ordinary quit; failed writes keep the current draft and allow retry/staying.
Do not close a DB under a pending draft write. Reuse DraftSession's existing
generation checks and invalid-buffer recovery; do not build another draft layer.
Startup with no Workflows visit must not open a workflow DB or start helpers.
No changes to the user's installed profile during verification.

## Editing and exchange

Stable step IDs, inputs, config and unknown metadata survive save/import/export.
Unsupported branch/parallel definitions remain preservable and clearly unsupported,
never flattened into a different sequence. Structural mutation refuses ambiguous
references. Invalid JSON retains the last valid form projection and its exact raw
draft. Saving a revision never silently replaces another concurrent revision.

Import/export uses explicit local file selection through existing picker/private-file
utilities and a size bound. Export a selected saved definition, not credentials,
local run state, grants or draft/view metadata. Opaque embedded content requires a
clear review warning; preserving it is not certification that it contains no secrets.
No server fetch/publish or sync action is advertised as implemented.

The user approved a stable-file contract on 2026-09-15. Do not externally move,
replace or relink the live workflow database or sidecars while the store is open,
or the selected JSON file/containing path during exchange. Normal SQLite-managed
writes and sidecar lifecycle remain supported. Reject visible DB/sidecar aliases,
symlinks, multiple hard links and non-regular files before generic file access;
retain the existing private-file checks and explicit overwrite confirmation.
Metadata validation is not a path lease: post-validation substitution or a
detached live inode can bypass the precheck and still disrupt SQLite locking.
That limitation is accepted outside the operating contract, not technically
fixed. No shared SQLite changes or isolated file-I/O infrastructure are added.

The server reference for this delivery is origin/dev at
2e1a5e58d3 (refreshed 2026-09-14). Its Workflows API schemas, endpoint and core
Workflows directory are unchanged from the prior 6cd2745f69 baseline, verified
with a zero diff. Keep compatibility evidence scoped to definition preservation,
not execution conformance or negotiated synchronization.

## UI and navigation

Use current dev's BaseAppScreen, token-backed component states and existing global
F6 delegation. No screen binding shadows ctrl+c/v/x/s/d/z/a/r/w, ctrl+p/q or f1/f6.
Use actual available content width: three panes at >=132, navigator/editor at
96-131, and editor plus labeled selectors below 96. Verify 160x48, 110x36 and 60x20.
Hidden panes leave Tab/F6 traversal; field editing/validation does not steal focus.
Preserve current dev Console-follow behavior as secondary inspection/handoff
content; it must not trigger whole-editor recomposition or replace draft state.

The visual system is fixed by DESIGN.md and backlog/docs/design-language.md:
compact terminal-native workbench, semantic status text, visible focus, readable
disabled controls, no new aesthetic exercise. The existing editor is the layout
reference. Generated CSS is rebuilt, never edited manually.

## Verification boundary

Targeted tests only. Real SQLite persistence and foreign-writer regressions;
document/draft and mounted keyboard behavior; import/export round trip including
unknown fields; real-app composition and quit; token/bundle/navigation checks.
Use real rendered Textual frames with production CSS at the three target sizes.
No model/network invocation is needed to qualify authoring; future local model
checks use llama.cpp at localhost:9099, never an Ollama substitution.
