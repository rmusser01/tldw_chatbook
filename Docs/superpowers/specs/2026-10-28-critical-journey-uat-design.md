# Critical user-journey UAT: catalog and project-chat pilot

Date: 2026-10-28
Status: approved 2026-10-28; pre-implementation review incorporated; pilot pending
Task: TASK-37270
Baseline: current origin/dev after PR #3284

## Purpose and scope

Catch failures of complete user actions before release, especially the sequence
create workspace -> bind project folder -> create chat -> inspect project. PR #3284
showed why successful tool cards and passing component tests are insufficient:
a directory tool could inspect empty scratch while the user expected the bound
project, and schema loading could fail after the initial tool plan succeeded.

The requested first deliverable is this wider workflow catalog plus an executable
project-chat pilot. Other catalog rows describe follow-on automation, not coverage
we already have. Existing unit/integration tests remain useful supporting evidence.

ADR required: no.
ADR path: N/A (test-only composition of existing pytest/Textual facilities).
Reason: no product storage, permission, provider, or runtime contract changes.
Existing contracts: [project guidance ADR-241](../../../backlog/decisions/241-workspace-project-context-and-folder-bound-index.md),
[project instructions ADR-069](../../../backlog/decisions/069-console-project-instruction-local-state-and-preflight.md),
[binding exclusions ADR-174](../../../backlog/decisions/174-workspace-binding-exclusions.md),
[compact disclosure ADR-253](../../../backlog/decisions/253-compact-discovery-before-plain-chat.md).
If implementation needs a product boundary change, assess that change separately.

## Approach

Use ordinary pytest tests, Textual Pilot, temporary fixture files, real SQLite,
and a loopback HTTP server implemented with existing dependencies or the standard
library. No new workflow language, browser framework, model download, or test service.

Two layers share the acceptance outcomes:

1. **Deterministic application journeys:** the real TldwCli startup, visible UI
   actions, production provider adapter, tool catalog, permissions, execution,
   persistence, and rendering. Only the remote model is scripted at HTTP. Run the
   small critical set on PRs without credentials or inference hardware.
2. **Live qualification:** the same user outcomes against a real configured model
   on Fedora and other supported environments. Record model/server versions and
   both context settings. Run before release and for relevant provider changes.
   A scripted pass cannot stand in for this result.

A live-only suite would expose real model behavior but be slow and nondeterministic
as a merge gate. Extending controller-level tests alone would leave startup,
provider serialization, and UI routing gaps. The user selected a catalog plus
pilot and approved this design. The review corrections below refine its evidence and isolation without changing the product scope.

Textual headless runs exercise the application event loop, not a native terminal.
Keep a small native-terminal qualification row for input, paint, and terminal-specific
behavior. A Fedora container exercises Fedora userland; it does not qualify host
SELinux policy or an actual terminal. Label these separately in results.

## Rules for every scripted workflow

- One disposable private profile and a fresh interpreter per independent case.
  Select all config/data/home/temp paths before importing the app. Reuse that
  profile only when the case explicitly tests restart. Do not copy a user's config
  or credentials. Disable unrelated cloud refresh and optional background work
  through supported settings, documenting the settings used.
- Start the actual app with its normal startup lifecycle. Do not force `_ui_ready`,
  mount ChatScreen under a substitute Host, construct a controller with `__new__`,
  replace its bridge/catalog, or prepopulate the workspace being tested in SQLite.
- Use visible controls for the journey. Await an actionable, in-viewport control,
  check the Pilot interaction was delivered, then wait for the resulting state.
  A direct Button.press fallback is an event-contract test, not click coverage.
- Observe the actual request sent to the model. Planning output before runtime
  additions is insufficient. Assert a nonzero request count for sends expected
  to reach the provider; expected preflight refusals must assert zero instead.
- Assert three views where applicable: user-visible result, real durable effect,
  and the responsible run/tool trace. A success toast alone never proves the effect.
- Use bounded waits for named conditions. Report which condition timed out. Do not
  sleep for a guessed duration or wait for all background workers globally.
- A selected case that cannot run is `blocked`, not `passed`. Missing live-model
  prerequisites are explicit; they do not turn a required release run green.
- Keep the first failed attempt. A diagnostic retry may add evidence but cannot
  overwrite the failure or silently turn a flaky gate green.
- Assert fixture cleanup and retirement of the work owned by the test. Preserve
  failure receipts before cleanup; cleanup failure is a test error.

## Common synthetic project fixture

Use a neutral directory name with spaces and a Unicode character. Create:

- `README.md`: three small lines describing a fictional work/rest timer, its local
  persistence behavior, and `Verification code: <fresh random value>`.
- A small manifest and one source file consistent with those claims.
- `AGENTS.md`: harmless guidance to consult README; no verification code or copied
  project summary. This prevents automatic instructions from satisfying the read.
- An unrelated decoy directory and a second workspace, each with distinct codes
  for isolation cases. Scratch starts empty in the project-read case.
- An excluded file with a distinct forbidden code only in exclusion cases.

The expected code belongs to the assertion/fixture side. The scripted model must
not receive it in its prompt, constructor, or canned final response. It can learn
it only from the real tool result returned in a later HTTP request. Its final reply
is derived from that result. Assert the code was absent from earlier requests,
the correct README was read, and the code is present in the final persisted and
visible final assistant answer. Match the final assistant message by its owning run/message identity, and assert its rendered viewport text separately; a code present only in a tool card, hidden widget, or earlier response cannot pass. Keep the fixture code and file contents synthetic, never secrets.

## Workflow catalog

Priority P0 means core-function or authority breakage should block a release.
P1 means important follow-on coverage. All rows below are specifications; the
implementation inventory must explicitly identify which variants actually ran.

### UAT-01 — Fresh setup and first plain chat (P0)

Setup: empty private profile; loopback provider initially healthy; no project.
Actions: complete provider/model setup, start a chat, type a unique prompt, Send.
Pass: the saved endpoint/model is the destination actually contacted; exactly one
user turn is accepted; streamed text appears and finishes; the final answer and
turn are saved; input becomes usable again. No folder is required for plain chat.
Negative: invalid endpoint produces an actionable setup/send failure, preserves
recoverable draft state, and never shows a completed assistant answer.
Evidence: provider request count/destination identity, visible final state, saved
conversation and run identity. Never include credentials in receipts.

### UAT-02 — Create project, bind folder, new chat, describe it (P0; pilot)

Setup: common fixture, empty scratch, local tools enabled, appropriate real read
permission configured through supported settings; no existing workspace/chat.
Actions:
1. Start TldwCli and wait for usable Console controls.
2. Open New workspace, enter a name, add the fixture folder, and confirm.
3. Create a new chat in that workspace; verify the active workspace/chat pairing.
4. Send: "Inspect this project's README. Summarize what it does and include its
   verification code."
5. On the first-use project-instructions notice, verify the chat, source, and
   provider destination; click Proceed. No generation may precede this decision.
6. Wait for actual completion, then inspect the visible response and saved run.
Pass: a real folder binding was saved; the model receives the owning binding;
file tools access that binding; README contents reach the provider; the answer
contains the fixture code and supported project facts; no decoy code appears;
there is no final claim that the nonempty project is empty; the chat is saved in
that workspace. A preliminary recoverable tool error may occur only in the
explicit recovery variant and must be followed by successful authorized reads.
Evidence: binding/chat/run IDs, schema names in each outbound request, tool
arguments and outcomes for synthetic files, final answer, viewport, DB rows.

### UAT-03 — Follow-up reads current files (P0)

Setup: complete UAT-02; replace README's verification code on disk.
Actions: in the same chat ask to reread README and report the current code.
Pass: a new real read occurs and the answer contains the new code. Old history or
Library material cannot satisfy it. No duplicate accepted user turn.
Variants: new chat in the same workspace; saved chat after application restart.
Evidence: old/new fixture hashes, read result, owning run, current answer.

### UAT-04 — Discovery, compact context, and schema recovery (P0; pilot subset)

Setup: nonempty project, all required readers allowed; discovery mode active.
Actions: run the project question with a small/unverified context configuration;
in a scripted recovery variant request an allowed reader before loading it, then
follow the refusal by loading its schema alone and retrying with valid arguments.
Pass: discovery is present in the actual outbound request when the plan says it
is; `load_tools` makes the reader available on the following request; the read
succeeds within the configured final request budget. Native variants must carry
actual native tool definitions, not merely a `native_tools` flag.
Negative variants: unavailable tools remain unavailable; an oversized requested
schema is refused within the same budget; if even minimum discovery cannot fit,
the app reports the limitation instead of advertising phantom capabilities.
Evidence: context source/limits, schema names and bounded request measurements,
full discovery/load/refusal sequence, final README evidence.

### UAT-05 — Wrong arguments cannot produce false project evidence (P0)

Setup: nonempty bound project and empty scratch.
Actions: script the reported call `list_directory(root_alias=..., path=".")`;
then recover using the advertised tool's actual schema and bound-root semantics.
Pass: the mismatched call is rejected before an apparent successful empty listing;
recovery lists/reads the actual project. Unknown fields must not silently select
scratch. Repeat the equivalent wrong-family read/write variants at their narrow
integration seam; a write variant must leave both project and scratch unchanged.
Evidence: refusal, subsequent valid call, read code, before/after fixture hashes.

### UAT-06 — Allow, Ask, Deny, and read-only behavior (P0)

Setup: fixture note; real permission store; each policy variant in a fresh case.
Actions: ask for a small edit; approve or deny through the visible approval card.
Pass: before approval the file is unchanged; approval applies the exact requested
change once; denial leaves it unchanged and remains understandable in transcript;
read-only bindings never become writable through schema loading. Turning local
tools off prevents local file execution while plain chat still works.
Evidence: approval/run/call correlation, actual file diff/hash, final outcome.
No custom callback may grant what the production approval path would deny.

### UAT-07 — Workspace, chat, and scratch isolation (P0)

Setup: A and B have different fixture codes; start a read in A and delay its model.
Actions: switch to B while A is running; send a question in B; finish A and reopen it.
Pass: A's run retains A's accepted authority and publishes only to A; B reads only B;
Default/unbound chats cannot obtain either folder through a stale selection;
private scratch is not shared across independent chats. Binding revocation must
follow the product's current authorization checks, not be bypassed by cached state.
Evidence: accepted session/workspace/binding IDs per run, tool results and saved
messages for both chats, explicit absence of the other fixture's code.

### UAT-08 — Missing, excluded, and awkward paths (P0)

Setup: separate cases for a missing bound folder, a nonempty folder with exclusions,
and an accessible folder whose path includes spaces/Unicode.
Actions: list/read the intended project and attempt the excluded fixture.
Pass: available paths resolve correctly; excluded names/content stay out of tool
results and model requests; unavailable roots produce an explicit availability
failure, not an authoritative successful empty listing. OS permission-denied and
symlink cases are qualified only on platforms where those controls are actually set.
Evidence: path category rather than personal paths, structured outcome, synthetic
result contents, supported platform/prerequisite facts.

### UAT-09 — Stop, provider failure, and retry (P0)

Setup: controllable provider pauses before response, midstream, or after a tool result.
Actions: Stop a running reply; send another message; separately inject a transient
provider error and use the normal retry action.
Pass: Stop settles the owning run; input becomes usable; later work completes;
no zombie run or cross-chat answer appears. Retry preserves trace/history
consistency and never treats a committed write as unexecuted. Any replay behavior
must match the existing continuation/idempotency contract; do not assert invented
"exactly once" semantics for arbitrary external side effects.
Evidence: attempt/run relationships, stream termination, durable tool outcomes,
fixture hashes, post-stop usable UI and subsequent successful provider request.

### UAT-10 — Save, restart, and reopen (P0)

Setup: a completed saved project chat and a separate private scratch file.
Actions: quit normally, restart using the same disposable profile, reopen the chat,
and reread the project.
Pass: history and workspace binding persist; fresh project access works; expired
scratch is not falsely presented as a durable project file; no stale Running or
pending approval state prevents the new send. Repeated profile creation is not a
substitute for testing restart against the same profile.
Evidence: before/after persisted IDs, new run, current file evidence and UI state.

### UAT-11 — Ingest -> retrieve -> answer with source (P0)

Setup: one synthetic local document with a fresh answer code, plus unrelated decoy
material; use an explicitly qualified retrieval backend.
Actions: ingest through Library, wait for actual indexing, select intended scope,
ask a question requiring the code, and open the returned source.
Pass: ingestion/indexing produces real searchable content; retrieved source and
answer match the selected document; citation resolves; excluded scope contributes
no content. No assertion can pass merely because an ingestion toast appeared.
Evidence: ingest job, stored source, index/retrieval receipt, source IDs in actual
provider context, final citation and opened source. Backend coverage is explicit.

### UAT-12 — Watchlist fetch -> saved item -> readable content (P1)

Setup: local HTTP fixture serves a tiny feed and article with a unique code.
Actions: create a watchlist/source, Check now, open the fetched item.
Pass: the fixture server observed the fetch; the item is persisted once with actual
article content; the UI shows it. Rechecking deduplicates. A server failure is
visible and cannot be reported as a successful empty update.
Evidence: HTTP receipt, persisted item/content, visible list/detail and failure.

### UAT-13 — Persona creation -> new chat -> effective behavior (P1)

Setup: synthetic persona with a distinct instruction/identity marker.
Actions: save it through the editor or approved character tool, start a chat with
it, send, then reopen the saved persona/chat.
Pass: persistence, selected persona, and the actual model context agree; editing
one persona does not alter another. Scripted checks inspect transmitted context;
a live variant checks model behavior separately.
Evidence: persona/chat IDs, selected UI state, bounded synthetic prompt excerpts.

### UAT-14 — Backup -> disposable restore -> usable data (P1)

Setup: disposable profile containing one chat, workspace, and note.
Actions: create a backup through supported UI; restore only into a second
explicitly disposable profile; restart and open/read each fixture.
Pass: restored objects are usable and match contents/relationships, not just file
counts. No real profile is overwritten. Invalid backup reports failure without
partial silent replacement. External folder availability is reported honestly.
Evidence: backup validation, restored IDs/content hashes, post-restore reads.

## First executable pilot: bounded implementation

Implement UAT-02 as one end-to-end test module, with a small number of variants:

- normal fenced discovery -> load -> bound listing/read -> final answer;
- native calls through a provider configuration that truly supports them;
- compact/unverified context with an allowed-but-undisclosed reader, then recovery;
- the reported wrong-family listing arguments, followed by schema-correct recovery.

The last variant implements the listing portion of UAT-05. Its read/write siblings
remain narrow regression coverage and planned journeys. Do not claim the entire
catalog is automated. The HTTP script should accept only the expected protocol transitions,
validate loaded tool schemas and prior results, and fail usefully on unexpected
requests. Choose actual tool arguments from the registry schemas, not historical
examples. Do not provide the final fixture answer to the model double upfront.

Use the existing private-profile child runner where it preserves true startup.
Inspect its autouse fixtures before reusing it: the general UI app factory patches
several services and must not silently substitute for this journey's real setup.
Only external model HTTP responses are scripted on the critical path. Record any
necessary test-environment substitutions explicitly.

Deliver:

- This catalog, updated with exact implemented pytest node IDs and run commands.
- One focused pilot module and the smallest shared HTTP fixture/helper needed.
- A failure receipt in pytest's artifact directory plus JUnit integration: case ID,
  commit, platform/Python/dependency versions, phase, counts, synthetic schema/tool
  sequence, binding/chat/run correlation, visible viewport, and bounded exception.
- Required-CI enrollment in the existing suitable lane/census once the pilot passes
  under that lane's dependency set. Do not add a parallel test orchestration system.
- A targeted negative-control run proving a missing/wrong project read turns the
  new pilot red. Retain that failure separately from the unmodified passing run.

## Run matrix and release decisions

The first implementation qualifies the pilot locally on Windows and, when runner
access is available, Linux using the existing CI dependency set. These are separate
results; an Ubuntu job is not described as a Fedora run.

Proposed promotion:

| When | Required evidence |
| --- | --- |
| Every PR | Small deterministic P0 set, starting with the new pilot, on the tested candidate SHA |
| Provider/tool/workspace changes | Native/fenced, normal/compact context and relevant denial/recovery variants |
| Before release | P0 workflows against pinned live llama.cpp/model on actual Fedora, plus supported Windows coverage |
| Wider qualification | P1 capabilities and native-terminal input/paint checks on their supported environments |

Start with Python 3.12 and 3.14, matching the existing recovery lane; record precise
patch versions and Fedora release. Pin or record model artifact identity, llama.cpp
build, chat template, tool mode, server `--ctx-size`, and app context limit. Hardware
can be metadata rather than a gating performance target. Fixed settle deadlines
are failure bounds, not latency promises. Do not expand all axes into a huge
Cartesian product: cover the incident combination and pair meaningful variants.

A release summary lists `passed`, `failed`, `blocked`, and `not_run` per selected
workflow/environment. Any required failure or unqualified mandatory environment
holds release. No automatic rerun-to-green or unrecorded waiver. Expanding mandatory
coverage is a deliberate policy choice, not a claim that the pilot already covers
all major features. Live execution is opt-in and needs a supplied endpoint/model;
this task does not download a model or use someone's credentials automatically.

## Existing evidence and reuse points

- `Tests/UI/test_console_new_workspace.py`: useful create-dialog selectors/actions;
  uses a substitute Host, so its pass is not full-app startup evidence.
- `Tests/UI/test_console_character_tools_mounted_uat.py`: real mounted send and
  approval-card pattern; gateway is replaced before provider serialization.
- `Tests/QA/test_console_notes_workspace_uat.py`: real tool/file round trip; starts
  below workspace creation and uses older compatibility-root setup.
- `Tests/Agents/test_tool_disclosure_refusals.py`,
  `Tests/Agents/test_first_request_window_honest_plan.py`, and
  `Tests/Tools/test_file_tools_workspace_roots.py`: focused incident regressions.
- `Tests/private_profile.py`: isolated interpreter and child failure receipts.
- `Docs/superpowers/qa/2026-10-20-workspace-project-context/guidance/README.md`:
  documents the difference between borrowed-app Host probes and actual-app Send.
- `backlog/docs/lessons-testing-evidence.md` and `lessons-live-verification.md`:
  require non-vacuous requests, real outcomes, and accurately labeled evidence.

Existing tests are starting points, not a completed mapping of the catalog. Add
node IDs, supported variants, last verified SHA/environment, and artifact locations
to this document when the pilot is implemented; leave other rows planned.


## Pre-implementation review corrections (2026-10-28)

1. **Final-answer oracle:** search the owning final assistant row and its actual
   paint, not the whole transcript, where a tool card alone could expose the code.
   Assert read provenance and DB linkage independently of the scripted response.
2. **Fixture masking:** the general UI factory replaces startup services, the UI
   conftest forces approximate token counting, and root fixtures disable probes.
   Place the pilot where it can preserve real startup/provider preparation; use
   the existing private-profile child runner and explicit network opt-ins. Record
   any remaining substitution and the token-counting mode. A heuristic estimate
   does not prove that a real model tokenizer fits its context window.
3. **Network and environment isolation:** the suite's loopback marker permits all
   loopback ports, not only the owned fixture. Add a test-local destination guard
   allowing only the bound numeric loopback address/port (and required local IPC),
   with recorded blocked attempts that fail even if swallowed. Sanitize inherited
   provider credentials, proxy configuration, and app overrides before child startup.
   Do not grant unrestricted network access. The HTTP server has an ephemeral port,
   bounded bodies/deadlines, and explicit shutdown/join with no worker left behind.
4. **Model protocol:** readiness/metadata GETs do not advance generation state.
   Validate method/path and the actual tool definitions before each call. Derive
   root aliases from the transmitted context/schema only, never hand the fixture
   binding or README answer to the model script. A native case must obtain support
   through the shipped configuration/metadata path and must send native schemas.
   Unexpected POSTs or transport fallback must fail with the observed stage.
5. **Configuration scope:** UAT-02 starts with a fresh *preconfigured* profile:
   provider/model, tools, read permissions, and completed onboarding are setup.
   It does not qualify UAT-01's setup wizard. Disable the optional project-context
   interview through its visible checkbox so an unrelated generation cannot count
   as the requested project-summary send. Do not force UI readiness or cached facts.
6. **Incident coverage:** include wrong-family listing recovery as the fourth
   pilot variant. Keep each failure mode isolated, rather than combining a compact
   budget, unloaded schema, and malformed arguments in one difficult-to-diagnose case.
7. **Gate and evidence:** JUnit alone can be green with skipped/missing variants.
   Check the exact expected variant set and reject skip/xfail/error/missing receipt.
   Retain metadata and synthetic-only failure receipts using an always-run CI
   artifact step with unique per-case names; missing required artifacts fail too.
   Measure local/CI wall time before choosing the existing lane. Preserve current
   lane budgets and do not enroll a large file blindly in a nearly full shard.
8. **Negative controls:** verify independently that no file read, a successful
   empty/wrong-root result, and a marker only in a tool message cannot satisfy the
   oracle. Run one full pilot with deliberately missing README evidence and retain
   its expected red result separately. Normal pilot cases must contain no mutation.

These are review findings and implementation requirements, not passing-test claims.
The only live qualification promised here is its workflow specification; executing
against Fedora/llama.cpp still requires a separately available environment.

## Running the executable pilot

The catalog is broader than the implemented pilot. The executable entry point is
`Tests/QA/test_project_chat_uat.py::test_project_chat_uat` with variants `fenced`,
`native`, `compact-recovery`, and `wrong-family`. Everything else in the catalog
remains planned. This is scripted-provider functional UAT, not a live-model eval.

Install the same dependencies as the required lane in a disposable virtualenv:

```bash
python -m pip install -e . pytest pytest-asyncio pytest-timeout packaging aiohttp-jinja2
TLDW_UAT_ARTIFACTS=project-chat-uat python -m pytest \
  Tests/QA/test_project_chat_uat.py \
  Tests/QA/test_project_chat_uat_support.py \
  Tests/QA/test_project_chat_uat_reports.py \
  -q -o addopts= --timeout=240 --tb=short \
  --basetemp=project-chat-uat-pytest --junitxml=project-chat-uat.xml
python scripts/check_project_chat_uat.py \
  --junit project-chat-uat.xml --artifacts project-chat-uat
```

Use a fresh artifact/output directory for each invocation: duplicate receipts are
rejected deliberately. On PowerShell, set `$env:TLDW_UAT_ARTIFACTS` before pytest.
Windows cases allow 600 seconds for cold native storage/profile startup; Linux
cases allow 240 seconds. These are test deadlines, not performance targets.

To prove a real missing-read failure is caught, set
`TLDW_UAT_NEGATIVE_CONTROL=missing-readme` and select `[fenced]` in its own fresh
output directories. That run **must fail**, with `README read did not supply a
verification code`; clear the variable before a normal run. Pure oracle tests also
reject an empty/error/wrong-root read and an answer containing the code only in a
tool card. The report tests reject missing, duplicate, skipped, failed and partial
case evidence.

Receipts contain synthetic requests/results, run linkage, phase timings, platform,
Python and tokenizer mode. Viewport SVG/text and child app logs diagnose failures.
Do not use real project files or credentials in this fixture. Real Fedora/llama.cpp
qualification is a separate catalog workflow and has not been performed by this
scripted run.

CI enrollment: the existing `pr-fast-lane-admission-2` required job runs this in a
separate pytest process, checks the exact four cases even after failure, and uploads
evidence with `if: always()`. No runner or job-timeout increase (ADR-103). The base
job took 12m16s in run 40779693441, leaving headroom under its existing 35-minute cap.
Qualification results will be added after the actual pilot passes; writing a test
or enrolling it is not proof that the journey works.

The CI job also runs the missing-evidence negative control after the normal gate.
It requires pytest exit 1, the precise missing-code failure, and an actual `fs_read`
result from the damaged README. A startup/network failure cannot satisfy this
control. Its artifacts are kept separately in `project-chat-negative`.

Independent implementation review found and fixed three additional false-green
paths: wrong/missing outbound model identity, tool calls that compensate for absent
argument schemas, and a run linked to the wrong saved assistant message. Ten focused
negative cases cover these defects. Review fixtures and gate checks passed locally;
full Linux pilot qualification remains pending. Direct SHA/dependency-version fields
in individual receipts are deferred; CI artifacts remain associated with their run
and tested head.

Qualification exposed an omitted UI action: first-use project-instruction consent.
Run 40801165548 preserved the visible notice and a worker waiting for its decision.
The pilot now accepts it through the real Proceed button and asserts no model
request preceded consent. Active task/thread stacks are retained separately from
idle widget stacks so a blocked journey identifies its actual wait.

The saved run is checked for owner, completion, and matching successful read-call
IDs. Its metadata intentionally omits file arguments and bodies; those are checked
against the actual provider HTTP history instead. The test never weakens production
trace privacy or injects raw payloads into AgentRunsDB.
