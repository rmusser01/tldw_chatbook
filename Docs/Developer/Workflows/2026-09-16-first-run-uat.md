# Session-bound file-to-Note qualification — 2026-09-16

The real app completed file → prompt → actual llama.cpp → edited review → Local
Note at 160×48, 110×36, and 60×20. A separate process then reopened the same
disposable on-disk profile after each walk. Definitions and Notes survived;
active runs/reviews did not, and no request or Note write replayed.

This is full `TldwCli.run_test` driver/compositor evidence with actual services,
HTTP transport and Notes persistence. It does not qualify native PTY input,
terminal fonts, or arbitrary providers/platforms. No `app_factory`, model
substitute, Ollama fallback, external font, installation, or host profile copy
was used for these six live runs. The older UI harness results remain separately
labelled regression evidence.

Implementation base: `b3e94fa245af0770cdf710a7f82033a8d1aee805`, branch
`codex/workflows-sequential-design`, worktree `.worktrees/workflows-authoring-dev`.
Task 6 changes qualification/docs, the authorized Notes module import order,
the privacy-test logger-level helper, an approved canvas-local mount-readiness
guard, and an approved editor stale-focus callback guard.
[ADR-138](../../../backlog/decisions/138-portable-workflow-definitions-and-local-execution.md)
governs the session lifetime; no new ADR or production interface was needed.

## Live results

Python 3.12.11, Textual 8.2.8, HTTPX 0.28.1; root repository virtualenv,
`PYTHONPATH` pinned to the worktree. Provider `llama_cpp`, selected endpoint
`http://localhost:9099`, numeric dispatch `http://127.0.0.1:9099/v1/chat/completions`.
Actual model:

```text
../../../Working/Language_Models/gemma-4-26B-A4B/gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf
```

The controller's [read-only model response](artifacts/2026-09-16-first-run/model-endpoint-readback.json)
at 2026-09-16 14:48:57 UTC confirms this identity, `owned_by=llamacpp`, context
64,000. That GET is identity evidence, not completion evidence. Every live POST
used the bounded client with `stream=false`, `max_tokens=512`, and no Authorization
header. The observer forwarded the unmodified real HTTP call.

| Terminal | Walk / restart PID | Walk POSTs / new Notes | Restart POSTs / new Notes | Approved Note ID |
| --- | --- | --- | --- | --- |
| 160×48 | 62205 / 64110 | 3 / 1 | 0 / 0 | `40dbcb35-d7a4-4a2d-8add-696cb5b69dfe` |
| 110×36 | 64331 / 64517 | 3 / 1 | 0 / 0 | `154e4fbe-02ce-4555-8e16-06279cf484d6` |
| 60×20 | 65664 / 65937 | 3 / 1 | 0 / 0 | `f4a603b7-b0e4-410e-8c9c-17261a235ca8` |

All three Notes contain exactly `Human-edited summary`, read back via the actual
captured Notes scope and again through Open Note's Library editor with the same
Note ID. The unchanged [example](../../../Tests/fixtures/workflows/file_to_note.json)
was loaded through real document services. Source text was synthetic: library
opening time, free returns, and Friday workshops.

Each walk exercised setup/destination review; file, model and Note Ask approvals;
editing and Accept; off-screen navigation and Stay; Open Note; cancel before
file access (no POST/Note); rejection of a second real generated review (no Note);
and quitting a third pending review with an unsaved edit. Restart asserted the
same saved revision and exact Note rows, no current session view, no restored
review text, zero observed HTTP attempts, and no new Note rows. Boot did not
recreate missing definitions on the final compact/medium restart checks.

The [39-SVG manifest](artifacts/2026-09-16-first-run/capture-manifest.json) names
each immutable capture, width, phase, source manifest and SHA256. All six
`live-<size>-<phase>.txt` files record passing pytest results. Adjacent JSONs hold
the actual paths, PIDs, request counts, Note comparisons and log classification.
The controller rendered and inspected all 39 [PNGs](artifacts/2026-09-16-first-run/output/playwright/)
using unchanged SVG viewBoxes, offline Chrome, four blocked font URLs/page, and
installed fallback fonts. No new Workflow layout blocker: edited review/Library
text was visible; Ask/Cancel/quit reachable; 60-column details scroll and the
session-only warning remains clear. The existing 160-column Library toolbar clip
and unselected-actions copy are outside this change. PNGs do not convert
compositor evidence into native-terminal qualification.

After the Library readiness fix, a new 110×36 live walk and fresh-process restart
passed on the same new disposable profile (`task-6-live-qualified`). PIDs
**83194 / 83684**, 3 / 0 POSTs, one / zero new Notes; Note
`993e7ced-1fe9-434a-bfe0-a50d8adea490` contains the same exact accepted edit.
The [post-fix manifest](artifacts/2026-09-16-first-run/postfix-qualified/capture-manifest.json)
verifies 13 additional SVGs. The controller inspected only its edited-review,
Open Note and restart PNGs, with no new issue. Total visual inspection is
**42 PNGs (39 original + 3 post-fix)**, not all 52 passing source SVGs. The later
editor callback fix changes no model/Note authority and is covered by real editor
controls; no additional live POST was sent for it.

The [derived packet summary](artifacts/2026-09-16-first-run/final-packet-summary.json)
cross-checks manifests against pytest outputs: **8 passing live app processes**
(4 walks, 4 fresh-process restarts), plus 2 control-complete walks whose final
qualification assertions failed. An earlier pre-app bootstrap failure is separate.
It also records the exact 42 inspected PNG paths/hashes.

## Isolation and privacy

Both launches used persistent profile
`.superpowers/sdd/2026-09-16-workflows-first-run/task-6-live-b/`. The existing
Tests bootstrap selected its private HOME/config/data and null keyring before
application imports. The ancestor autouse fixture then selected a distinct
per-test HOME/config/data directory. The probe precreated and parsed that exact
per-test TOML before the autouse config import, then selected the already parsed
persistent profile TOML for the actual app. It never reassigned host HOME itself.

Each manifest distinguishes bootstrap root, per-test root, effective HOME and
USERPROFILE, XDG paths, model/HF/tiktoken cache selectors, config, data, permission
store, and every `get_*_db_path`. Persistent config/data/cache/permission/DB paths
are equal across each fresh-process pair and inside the disposable profile.
Application/module origins are asserted inside this worktree. All app DB/Notes
services were real. Runtime source was explicitly asserted Local; `[tldw_api]`
was configured to numeric loopback port 1 with an empty token. No real tldw_server
was stopped, started, or needed.

Before child pytest launch, the runner removes ambient credential, provider URL,
proxy and cache/root selectors by name, reinstates only task-owned selectors,
and records removed names without values. The actual keyring backend is asserted
`keyring.backends.null.Keyring`. Model catalog refresh is disabled by the real
`auto_refresh_enabled=false` setting. HTTP observation refuses destinations other
than numeric loopback 9099. No host credential files are read/copied by the probe.

TOML parses before and after boot, with relevant settings/effective-path evidence.
Later manifests record before/after raw hashes and unchanged authority settings;
their semantic changed-section lists are empty. The early 160-column walk retained
the parsed after-config; its compact manifest records its canonical parsed-JSON
hash and explicitly lacks a before hash. Full originals are retained in task-owned
scratch pending review, rather than repeating unrelated defaults in every manifest.

The first post-fix walk's strict config audit failed despite completed controls:
shutdown persistence merges and writes defaults, adding three database keys to
the initially sparse authority table. The unchanged failed packet and
[source/semantic diagnosis](artifacts/2026-09-16-first-run/postfix-config-diagnosis.md)
are preserved. The next fresh profile predeclares those three keys, including a
private legacy base path, so whole-table authority equality remains strict (no
wildcard allowance). The passing post-fix pair re-evaluates every effective DB,
data, model-cache, config and permission path after quit and requires exact
equality to initial private paths without reopening databases. Both raw TOML
snapshots and hashes are retained. Endpoint/keyless/disabled-catalog values in
both snapshots were independently verified for the walk and asserted in-process
on restart; the extra assertion was added after that walk, not retroactively run.

App logs are retained alongside each manifest. Each passing process had zero
`unhandled_exception`, no `app_stopping` before the explicit quit, and exactly one
expected `app_stopping` after quit. `TldwCli.on_unmount` emits that event to mark
clean shutdown; matched lines are retained, not ignored. Source, full prompt,
actual generated responses, accepted edit and unsaved-review canaries were absent
from ordinary app logs. Intentional evidence artifacts include synthetic source,
review and Note content; that is not ordinary logging.

## Failed attempts and regression qualification

The [first live packet](artifacts/2026-09-16-first-run/attempt-1/) is preserved.
Its control assertions completed (3 real POSTs, one Note, 12 captures), but pytest
failed a naive combined-marker check on one normal post-quit `app_stopping` line.
Its historical `result=passed` means controls completed, not overall pytest success.
The corrected scan qualifies stop timing/counts explicitly. Task 6 therefore made
18 actual model POSTs total: 3 in that earlier packet, 9 in the original passing
walks, 3 in the failed post-fix config audit, and 3 in the passing post-fix walk;
six Notes were created across four disposable profiles. Controller preflight
POSTs are excluded. No separate fresh completion preflight was sent by Task 6.

The joined integration uses an owned loopback HTTP peer, actual app controls,
real document/permission/Notes services, and deliberately conflicting current
provider config after launch. [RED](artifacts/2026-09-16-first-run/integration-red.txt)
removes only edited-review propagation: the real saved Note contains the generated
text and fails the exact edited-content assertion. Restoring the wire gives
[GREEN](artifacts/2026-09-16-first-run/integration-green.txt): 1 passed, live test
skipped, one inherited dependency warning. Initial socket EPERM and driver setup
mistakes are preserved separately and are not called behavioral RED.

The authorized merged regression run produced **1,552 passed, 2 failed, 1 skipped**
in 406.41 seconds. [Complete output](artifacts/2026-09-16-first-run/merged-regressions.txt)
includes the bounded projection-performance results. The separate ProductionApp
lifecycle invocation passed **12 tests**. No full repository sweep was run.

Both failures passed in a fresh focused invocation (2/2). A combined new real-app
integration followed by all three existing UI flow widths passed (4/4). The
logging failure also reproduces after an existing authoring app test without the
new Task 6 test: application logging sets `httpx` to WARNING, while the privacy
test changes only the root level and expects an INFO record. The original
Library failure was `NoMatches('#library-note-edit')` during a recompose callback.
A permanent held-nested-mount test reproduces it: shallow authority/title nodes
exist while mode-control children have not mounted. The approved canvas-local
flag retains state until completed mounting, then the existing post-compose hook
automatically paints the latest state before the focus callback. No generic
framework, ownership or exception-handling change was needed. The permanent
[RED](artifacts/2026-09-16-first-run/library-held-permanent-red.txt) and
[GREEN](artifacts/2026-09-16-first-run/library-held-green.txt) retain this evidence;
both real-app/owned-HTTP/Notes cases passed, including exact Open Note content,
automatic latest-state painting, callback focus and no-op rebuild readiness.

The approved privacy-test helper now snapshots/sets/restores `httpx`'s level;
ordinary positive and private negative assertions remain. Existing authoring app
test followed by the full logging module passed **74 tests**, including level
restoration ([output](artifacts/2026-09-16-first-run/logging-order-green.txt)).
No production logging behavior changed.

Expanded Library coverage had **170 passed, 1 failed**: the extra Skills-import
test's `SimpleNamespace` lacks `_patch_library_skills_import_status_line`. Its
test and called production source are byte-identical to the execution base and
it does not construct or call the changed Notes canvas. The complete failure is
preserved in [expanded output](artifacts/2026-09-16-first-run/library-focused-green.txt)
(filename is an intended gate label, not a passing-result claim). No Skills
changes or weakened assertions were made. The Notes-specific covering
follow-up passed **165 tests** in 83.26s
([output](artifacts/2026-09-16-first-run/library-focused-notes.txt)); its exact
selection is in the command record, not hidden by a blanket deselection.

The subsequent unfiltered authorized merged manifest produced **1,555 passed,
1 failed, 1 skipped** in 412.13s ([complete output](artifacts/2026-09-16-first-run/merged-regressions-final.txt)).
Both original failures passed. The new failure was an existing authoring field
hit check at offscreen coordinate `(68,38)`. Its isolated selector passed and
combined editor/run modules passed 113, but a single held-callback experiment
reproduced the exact failure: an older section `after_layout` callback stole
the user's newer field focus and reset scroll before invalid input. The error
status then added two rows; it was not the initial displacement.

The approved editor fix snapshots queued focus and skips both stale scroll and
focus restoration when a newer attached in-editor control owns focus. Its
permanent event-barrier test went
[RED](artifacts/2026-09-16-first-run/editor-permanent-red.txt) on stolen focus,
then [GREEN](artifacts/2026-09-16-first-run/editor-permanent-green.txt) with all
three existing invalid-field cases (4 passed). The test asserts unchanged latest
focus/scroll and performs actual keyboard invalidation with no corrective scroll;
the post-invalid painted-hit assertion remains intact. Final covering results
are **159 passed, 1 live-only skip, 1 inherited warning** in 207.79s
([complete output](artifacts/2026-09-16-first-run/editor-final-covering.txt)):
entire editor/run/paging/projection-performance modules and the joined real-app
HTTP/Notes tests. The prior merged failure is not relabeled as a green run.
Per controller direction, no third merged sweep was run after this narrow fix.

[Static delta](artifacts/2026-09-16-first-run/static-delta.json) compares all 25
Task 1–5 changed Python paths against `cf61cb68505fc62991a0488c964a78cb7ab31cbb`.
1,201 retained lint findings versus 1,203 baseline; no introduced formatter edits.
Three moved-line UP035 diagnostics have explicit unchanged-symbol
[source attribution](artifacts/2026-09-16-first-run/static-attribution.md).
The new Task 6 file is fully Ruff/formatter clean. No whole-file-clean claim is
made for legacy files.

The final [28-file static delta](artifacts/2026-09-16-first-run/static-final-delta.json)
also covers the new integration and approved logging/canvas changes: **1,232
baseline / 1,230 current** lint findings and **101 / 101** formatter edits.
Only the same three unchanged-symbol Notes import diagnostics need manual
attribution; no new lint or formatter debt. Complete current diagnostics are
retained, not suppressed.

The [commit-scope delta](artifacts/2026-09-16-first-run/static-commit-delta.json)
adds the two editor files (30 paths total), with the same counts and no new debt.
New integration and both editor files are fully Ruff/formatter clean.
Staged source/documentation whitespace checks pass. The full staged check reports
only original blank padding in generated SVGs and one preserved RED-output line;
[exact attribution](artifacts/2026-09-16-first-run/staged-whitespace-attribution.json)
records these exceptions. Capture bytes/hashes were preserved, not normalized,
and no Git whitespace configuration was overridden.

The [final ordinary-log scan](artifacts/2026-09-16-first-run/final-log-privacy-scan.json)
retains hashes and marker/canary counts for every copied app log. Restart log
copies include prior process history; per-process manifests classify only their
own appended segment. Full model-response canaries were additionally checked in
each live process. Intentional synthetic capture and failing-test payloads are
not mislabeled as ordinary application logs.

## Limits of the result

This qualifies the selected macOS/keyless loopback setup and unchanged five-step
example, not the wider historical workflow roadmap, server schema parity, durable
run recovery, cross-instance deduplication, or exactly-once writes. A crash can
leave a Note without a run receipt; inspect Notes before manually rerunning an
uncertain write. Sessions are independent between app instances. Source paths,
database paths and aliases must remain stable while used; metadata checks do not
solve concurrent path substitution. Cancelling a request does not prove the
server stopped generation. See the [user guide](../../User_Guide/workflows.md)
for supported operations, limits and the session-loss disclosure.

At the Task 6 handoff, TASK-32691 remained In Progress for controller acceptance.
The subsequent whole-branch review found one Open Note/worker lock inversion;
the narrow fix, deterministic regression and single scoped re-review are recorded
in the [final review record](2026-09-16-first-run-review-record.md). Controller
verification at the final source head passed 11 targeted cases with one live-only
skip and the existing dependency warning. All nine acceptance criteria are now
supported and the task is closed through the Backlog CLI. This is not an all-green
full-suite or native-terminal claim, nor authorization to push, create a PR or merge.
