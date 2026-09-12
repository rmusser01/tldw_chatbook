# Speculative Voice Dev Integration Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for implementation and task-scoped review. This is a port of existing approved behavior, not renewed feature design or physical qualification.

**Goal:** Open one PR against latest `dev` containing the complete audio feature and its required existing Console functionality, without reverting newer dev work or altering the original feature worktree.

**Architecture:** Preserve the committed audio-process/core boundaries and app-owned provider, promotion, cleanup and trace authority. Transplant self-contained voice files from the pinned source; adapt shared integration hunks to current dev rather than replacing shared modules with older snapshots.

**Tech Stack:** Python 3.12 test environment, Textual 8.x, SQLite, existing native WebRTC AEC3 companion.

**Spec:** Existing voice contract in `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`, as amended on source commit `258beb6120a5e8c84d4f83b12a39427733301e57`, plus the existing accepted `backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md`. The user clarified that the best route must integrate the audio work in its entirety onto latest dev, including its required existing Console functionality; the earlier narrow-file-only restriction is superseded.

ADR required: no new ADR for a behavior-preserving port
ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`
Reason: Reuse the existing ADR-094 acceptance/lifetime implementation required by ADR-098. This is integration of approved architecture, not a new voice-only acceptance subsystem. Retain current dev trace privacy, off-thread settlement and Canvas authority.

## Global Constraints

- Source commit: `258beb6120a5e8c84d4f83b12a39427733301e57`.
- Initial dev base: `c4d45c0926580a8756cfa13c5463b1d0fc808c1a`.
- Latest integrated dev: `37bf45fb6232a1d4fb50fdba8f3c19c856ae7664`, normally merged in `2e5da4ebd2110c03d84a5ccfed6f5208e0e28e8e`; refresh again before publishing if dev moves.
- Work only in `.worktrees/speculative-duplex-voice-dev`, branch `codex/speculative-duplex-voice-dev`.
- Original worktree's 19 unrelated edits remain untouched; original diff SHA-256 is `601b06a6481de59adce4fa3a2273291471828f2372d44acc847f31666bb3b867`.
- Silence default remains 700 ms; native streaming STT preferred with rolling fallback; interruptions extend the same unaccepted turn; speech after playback completes begins a new turn.
- Preserve sequential cancellable TTS, fail-closed AEC and bounded process/data/resource custody. No threshold, credit, timeout or cleanup relaxation.
- Keep all packaged platforms unqualified. Never regenerate authority from historical evidence or add a packaged selection bypass.
- Preserve dev app version `0.2.0`; align the companion's source version and exact optional dependency to it. Historical attestation/qualification fixtures remain historical, not new evidence.
- Preserve migrations through schema 68. The voice provenance extension must use the next migration slot and retain current dev guards and indexes.
- Use the main repository's existing `.venv`; do not install dependencies or change user config/provider selection.
- Targeted software tests only. No full suite, live app/audio, model/provider requests, physical qualification, route changes, matrix, repetitions or soaks.
- The user separately approved normal automatic PR CI, including the imported hosted native-wheel matrix. This does not authorize local native testing/installations, manual workflow dispatch, physical qualification or release publishing.
- Include the existing ADR-094 accepted-turn runtime prerequisite: frozen inputs, synchronous custody, view detachment, retained decisions, terminal receipts and bounded close/quit. Do not replace it with a parallel voice-only acceptance service. Preserve current dev trace-settlement authority and privacy.

## Task 1: Import the self-contained audio and speech implementation

**Files:** New voice-owned `tldw_chatbook/Audio` modules, `native/voice_aec`, their new `Tests/Audio` files/fakes/fixtures, `Tests/STT/test_resident_buffer_runtime.py`; shared `Audio/dictation_service_lazy.py`, `STT/executor_worker.py`, `Local_Ingestion/transcription_service.py`, `TTS/{TTS_Generation,pcm_stream,request_admission}.py`, `Utils/fd_protection.py` and their affected tests.

**Interfaces:** Keep source `AudioFrame`, rolling transcript, child protocol and native ABI unchanged. Add source `ResidentBufferRuntime`, explicit facade `local_buffer_owner`, `synthesize_hands_free`, normalized PCM iterator and bounded descriptor-protection timeout without changing ordinary callers.

- [x] Verify a clean isolated dev baseline with `Tests/Audio/test_dictation_speech_resumed.py`, `Tests/TTS/test_pcm_stream_plan.py`, `Tests/TTS/test_tts_request_admission.py`, `Tests/Utils/test_fd_protection.py`: 100 passed, three existing dependency warnings, 5.03 s.
- [x] Resolve missing shared authority contracts: the user's clarified scope authorizes bringing the existing ADR-094 prerequisite. Task 2 integrates it before production voice activation is exercised.
- [x] Import absent voice-only files from the source commit, byte-for-byte. Use committed Git blobs, never the dirty source worktree. Do not replace files that already exist on dev.
- [x] Apply only the source voice hunks to shared speech modules. Preserve dev's `recorder_factory` injection and other current behavior.
- [x] Import the corresponding source-added test cases, retaining current dev tests. Check the new cases fail for the absent contract before applying shared-module hunks.
- [x] Run affected software audio/STT/TTS/descriptor tests with an isolated temporary basetemp. Do not run native GIL-hold groups or soaks for unchanged native source.
- [x] Record exact imported hashes and any adapted files, review the task diff, then commit only the explicit task paths.

## Task 2: Integrate the existing Console accepted-turn prerequisite

**Files:** Existing source changes in `Chat/console_runtime.py`, `console_turn_context.py`, `console_chat_controller.py`, `chat_persistence_service.py`, `console_fleet_wake.py`, `console_launch_wake.py`, `console_agent_bridge.py`, `console_prompt_queue*.py`, `conversation_local_marks_service.py`, `message_metadata.py`, `Agents/agent_models.py`; navigation-facing hooks and tests in the source prerequisite range. Keep source-specific voice additions for Task 3.

**Source boundary:** The contiguous existing prerequisite implementation is `15903ab116..5ed008b091`, starting with `9bc983ea92` and ending at `5ed008b091`. Use that delta, not the entire interleaved trace branch. Later voice commits amend this implementation in Task 3.

**Interfaces:** Bring `ConsoleTurnCustodyRequest`, synchronous `ConsoleRuntime.accept_turn`, immutable handoff inputs, detach-only `leave_console`, exact terminal receipt/local mark publication, retained decision clocks, and close/quit custody. Keep current dev store/controller as run-state authorities and retain Canvas and trace teardown.

- [x] Port source prerequisite tests before shared implementation; establish the absent-custody/navigation behavior as the targeted RED baseline.
- [x] Merge the prerequisite delta into current dev shared modules, retaining newer provider, PII, Canvas, inspector and settings code. No whole-file checkout of shared modules.
- [x] Preserve ordinary chat, agent, wake and recovery callers of changed ownership APIs. Import the source's corresponding fixture updates only where required by that contract.
- [x] Verify explicit runtime-custody/navigation/terminal-attention/quit tests; compare old and new failures rather than rewriting expectations to accommodate regressions.
- [x] Review the scoped prerequisite diff and commit explicit files. Task 3 then adds voice promotion/process ownership to these existing APIs.

## Task 3: Integrate existing app-side voice authority with current dev

**Files:** New `Chat/console_speculative_voice*.py`, `Chat/console_voice_*.py`, `Chat/voice_phrase_sequencer.py`; voice hunks in `chat_persistence_service.py`, `console_chat_controller.py`, `console_chat_store.py`, `console_runtime.py`, `console_provider_gateway.py`, `console_prepared_request.py`, `console_turn_preparation.py`, `console_exchange_capture.py`, `console_context_compaction.py`, `console_visual_transcript.py`, `console_trace_{models,repository,service,final_values,provenance,custom_pii}.py`; next DB migration and narrow `ChaChaNotes_DB.py` integration; voice-specific Chat/DB tests.

**Interfaces:** Preserve source synchronous winning-promotion claim and exact original provider context custody. Preserve dev's ordinary response accumulator, settlement sink/handoff, off-thread persistence and Canvas cleanup. Voice terminal import must retain current trace owner/privacy/GC guards.

- [x] Use Task 2's existing accepted-turn custody/receipt/close APIs; preserve synchronous claim semantics without acknowledging merely scheduled work.
- [x] Port only voice-owned additions and required narrow adapters; do not copy whole controller/runtime/store/trace files from source.
- [x] Reuse current dev privacy projection and revision masks at the existing provisional gateway/winning import boundary. Freeze scoped policy and one-shot revision, redact before sealing, import immutable bytes/spans atomically, and consume one-shot privacy only on a successful winning claim. Document this adaptation in ADR-098 before implementing it; do not downgrade privacy-enabled voice to text-only custody.
- [x] Port voice provenance SQL as schema 68 to 69 only after confirming that slot is still free; retain every dev migration and existing trigger/index behavior.
- [x] Adapt the source voice migration regression to a real schema-68 predecessor and verify ordinary calls retain default provenance while promoted calls carry the exceptional provenance/reason.
- [x] Run explicit voice attempt/process/promotion/capture tests plus the exact dev persistence and trace tests covering changed entry points.
- [x] Review the task diff against both source behavior and dev invariants, then commit explicit owned files.

## Task 4: Integrate the visible control, settings and release packaging

**Files:** Voice hunks in `UI/Console_Modules/{hands_free,dictation,wiring}.py`, `UI/Screens/chat_screen.py`, `app.py`, relevant Console and Speech/TTS widgets, `config.py`, package initializers, `Utils/persistent_diagnostics.py`; new voice preview widget, source-only launcher, voice Packaging/scripts/workflows/docs and scoped UI/packaging tests; `pyproject.toml`, `MANIFEST.in`.

**Interfaces:** Visible Hands-free selects the isolated session only through existing qualified selection or the source-only exact-HEAD development entry. Preserve dev's navigation, meeting recorder injection, settings and startup laziness.

- [x] Import voice-only packaging/docs/test files; merge shared docs and initialization hunks without reverting current dev content.
- [x] Keep `0.2.0` app/public versions; set source companion metadata and speech extra pin to `0.2.0`. Retain old unqualified packaged identity bytes unchanged.
- [x] Port the UI factory/lifecycle and categorical diagnostics; use the combined owner APIs resolved in Tasks 2 and 3.
- [x] Run source-only launcher validation and mounted visible-control tests with fake audio/provider/model boundaries; prove the actual selected session class.
- [x] Update source/lint inventories for renamed migration/tests and actual port dependencies. Verify the launcher is excluded from wheel and sdist; no application config/env/release override.
- [x] Review and commit only the explicit task paths.

## Task 5: Verify and publish the dev-targeted PR

**Files:** This plan, task-23175 implementation notes, integration evidence, and any narrow integration corrections exposed by the named gates.

- [x] Run one joined targeted software integration gate over affected voice/core/Chat/UI/packaging files, excluding live/native-hold/soak entry points. Record commands, counts and warnings; classify failures before changing fixtures.
- [x] Run affected-file Ruff lint/format and `git diff --check`; verify current dev is an ancestor and audit the exact PR path/commit range for unrelated work. Retain the exact baseline lint/format debt described below; this is not a globally clean-lint claim.
- [x] Independently review the port, focusing on shared-module differences rather than re-qualifying the unchanged audio implementation.
- [x] Verify original feature HEAD and protected diff hash are unchanged; keep both worktrees. Root independently reverified after the latest-dev merge.
- [x] Update task implementation notes without marking broader release qualification complete.
- [x] Push only `codex/speculative-duplex-voice-dev` and create one PR explicitly targeting `dev`; do not merge it or publish the native package.

## Checkpoint history

The isolated integration branch contains planning commit `89d803a25a` and the
reviewed audio-core import `c88f2d9ba0`. Task 1 is complete: 637 targeted software
cases passed, with 28 app-composition cases explicitly deferred until Tasks 3–4.
The imported native/vendor implementation is unchanged from the pinned source;
only the approved companion source-version alignment differs.

Task 2's existing ADR-094 prerequisite is committed as `c142efd4a5`, with spec
corrections in `5f859a4f2c` and `2efa2af6bf`, retaining current dev ownership,
Canvas and trace behavior. Independent spec review approved all five findings
as resolved at `2efa2af6bf`. Independent quality review approved the full task
after runtime cleanup/publication corrections in `e226221c8c` and `ec6411f160`:
no UI handoff under a blocking lock, no closed-owner recovery retention, and no
lost queued attention update after a same-value recompute. The controller
independently reran focused correction gates (100, 36, 39 and 41 passes).
This is not a full-suite or formatter-clean claim: diagnosed baseline failures,
existing format debt and incompletely attributed aggregate descriptor warnings
remain explicit in the evidence. Tasks 3–4 are complete; Task 5 remains pending.
The task-scoped reports and
recovery ledger under `.superpowers/sdd/2026-09-06-speculative-voice-dev-integration`
record exact commands, diagnosed baseline failures and current checkpoints.

Remote dev was refreshed to `6460d1eef39556137e1fc391a4c0daa011381db2` on
2026-09-07 and merged in `2244fc47a5e5cffb61353f3d2124a36a19ca8617`; schema
remains 68. Both merge parents and a clean tracked worktree were verified. The
reconciliation preserves new note actions, switcher ownership, trace-construction
recovery and content-free send diagnostics. It resolves the two shared Console
conflicts, forwards every frozen custody input through the diagnostics wrapper,
and keeps synchronous admission in the existing app runtime. The controller's
independent 19-case targeted check passed with three existing dependency warnings.
The broader scoped merge evidence retains a one-off wizard teardown failure
(exact rerun passed) and unattributed aggregate descriptor growth; neither is
reported as clean. The ignored merge report records the exact commands and scope.
Task 3 adapted the existing voice authority to this merged base, including
current trace/privacy contracts and schema 69. Refresh dev again before publication.
No PR exists for the integration branch yet.

Task 3's main authority/privacy import is committed as `9718c69bef`, with
specification corrections in `a43f11348a` and `159fc21b34`. Specification review
approved the ordinary acceptance and mandatory artifact-sanitizer corrections.
Independent quality review approved Task 3 at `247ebf7a13`, after `5ad7a0ec54`
fixed cross-turn/retry surface composition and the final correction preserved
changed-prompt authority and sanitized generated inline metadata before sealing.
The real gateway-to-SQLite test now verifies exact stored headers; production
reconstruction tests cover consecutive imports, retries, inherited history and
ordinary continuation. Root independently verified committed correction gates
of 9 and 22 cases; the worker's overlapping component gates passed 102 and 42.
These are software integration results, not physical qualification or a full
suite. Explicit Task 4 dependencies (preview/settings/process helpers, app quit
permit, launcher and packaging) and Task 5 joined checks remain pending. Packaged
platform qualification is unchanged and all platforms remain unqualified.

Task 4 is committed in `aa012fec4a`, with a one-document shortcut correction in
`0321caca66`. Independent specification and code-quality reviews approved the
full task at `0321caca66`. The root's committed five-case mounted check verifies
the exact-HEAD launcher selects `ConsoleSpeculativeHandsFreeSession` and the
actual default factory creates an attested private fake child, with toggle,
navigation, suspension and unmount cleanup. Reviewer gates passed 30 and 47
specification cases and 28 settings/preview/quit cases. The worker's overlapping
final shared gate passed 148 cases with one exact upstream-baseline failure
excluded; these counts must not be summed.

Actual application wheel/sdist checks and independent archive inspection confirm
development-launcher exclusion, retained profile-core packaging, version 0.2.0,
and unchanged packaged authority JSON bytes. The existing CSS budget passes at
803996/804000 bytes after narrow whitespace-only compaction, with deterministic
regeneration and parser checks. This 4-byte headroom must be rechecked after the
latest-dev merge; no budget increase is authorized. Existing app E402/format
debt, dependency warnings, the upstream workspace-row click failure and the
source-equivalent MCP diagnostic AST mismatch remain explicit limitations.
Task 5 normally merged newer dev `3cccd9326c` (schema 68) in `75b72a8a17`,
retaining the integration's schema 69 extension. Final independent review and
publication of one PR against dev remain root-owned and pending.
No local live/native/physical/soak verification or release publication is added.

Task 5's joined targeted software gate passed **326 cases**, with three existing
dependency warnings, in 166.38 s. Its 93-selector manifest and exact command/
failure history are in `task-5-software-nodes.txt` and `task-5-report.md` under
the existing ignored evidence directory. This catches up all 28 deferred Task 1
parameters and composes the real producer/privacy/reconstruction boundaries,
visible exact-HEAD launcher/default factory with fake child, settings/preview/
quit, ephemerality, rotating diagnostics and latest-dev preservation cases.
Only software selectors from mixed files ran; the actual installed-native AEC
corpus node remains explicitly excluded.

The initial 27 failures were obsolete upstream fixture seams: close API,
pending-send attribute, Textual-worker completion and automatic draft restore.
Adaptations retain exact draft/session recovery and invoke the actual UI restore
action before unchanged repaint/caret assertions. Passing diagnostic/blink
fixtures also retained thread-owned SQLite handles on untouched exact dev;
observed refresh retention justified the same bounded fixture teardown. Existing
same-file quiescence now asserts zero registered handles after runtime disposal.
The final joined observer recorded 11→32 descriptors (peak 35), with no sentinel
warning. This does not reclassify earlier Task 2/3 aggregate resource warnings.

All six derived-artifact preflight checks pass after narrow diagnostic inventory
regeneration and 12 exact SQLite query-plan-backed schema-69 index pins. CSS
remains 803996/804000 bytes. Ruff checks on five owned test files pass; the full
274-file affected check retains 162 diagnostics matching exact dev by code and
message, and 84 files of format debt already present at the merge checkpoint.
No production code, cleanup policy, budget, native authority or qualification
changed after the merge. The reviewed blink exceptional-cleanup follow-up is
verified separately after the frozen joined run; final application archive
paths/hashes and its exact result are retained in the Task 5 report. Task23175
remains In Progress; broader qualification and actual PR publication are pending.

Publication refresh: merged dev `37bf45fb62` normally in `2e5da4ebd2` after
the frozen 326-case gate. Its eight upstream paths are Library-only danger
styles, spacing, docs and tests; no voice/Console/provider/schema/config boundary
changed. Both source and generated Library CSS coexist with voice compaction.
The two incoming synthetic paint nodes (wide/narrow each) and the boot CSS guard
passed **5 cases**, four warnings, in 8.49 s; the fourth warning is the unchanged
4-byte CSS headroom. Exact incoming Python files match upstream, including one
F841 and four files of existing format debt. The six-check preflight and refreshed
actual wheel/sdist result are recorded in the Task 5 report. No destructive UI
action, repeated voice gate, native operation or install was used. Earlier 326
results remain attributed to the pre-refresh input, not relabeled as a37bf run.

The initial persistence dependency scan found a load-bearing architecture mismatch
(the accepted-turn prerequisite is now integrated by Task 2):

- Source `ConsoleChatController.submit_accepted_voice_turn` constructs `ConsoleTurnCustodyRequest` and synchronously invokes `runtime.accept_turn` after freezing staged evidence. Dev has neither contract.
- Source `ProcessVoiceEffects` interprets the synchronous receipt as accepted custody and emits `terminal_claim`. Dev's asynchronous `submit_draft` captures staged inputs later. Scheduling that coroutine is not acceptance and cannot truthfully stand in for the source receipt.
- Source promotion assumes async close/quit claim fences absent from dev's current lifecycle. Whole-runtime replacement would import the separate accepted-turn/navigation program and discard current dev Canvas and off-thread trace behavior.
- Source speculative preparation hardcodes PII redaction off; dev's effective scoped/custom PII policy must remain authoritative.
- Dev's provider response accumulator and settlement sink/handoff supersede source aggregate trace settlement. A provisional adapter must preserve current ordinary trace settlement and GC/privacy rules.

The user resolved the scope question by asking for the best route to audio integration in its entirety. Proceed with the existing accepted-turn prerequisite under ADR-094, preserving its functionality rather than inventing a new adapter. Neither an early fabricated claim nor disabled tool/capture behavior is an acceptable shortcut. The original pause is resolved.

Original feature HEAD remains `258beb6120a5e8c84d4f83b12a39427733301e57`; its
protected diff checksum was reverified unchanged on 2026-09-07. No integration
application launch or hardware activity occurred during the port. The removed
temporary html5lib test environment is replaced only for test commands by an
existing read-only standalone cached package; no dependencies were installed
and no shared environment or user configuration was changed.

## Final publication outcome

Opened [PR #2504](https://github.com/rmusser01/tldw_chatbook/pull/2504) from
`codex/speculative-duplex-voice-dev` to exact current dev `37bf45fb62` after
all five task-scoped specification/quality reviews and the final independent
whole-branch review approved implementation `f6ed08120e` with no actionable
findings. Root independently ran 18 targeted boundary cases (18 passed, three
existing dependency warnings, 29.19 s), all six derived-artifact checks, and
inspected the actual final wheel/sdist contents and original authority hashes.
These checks overlap the earlier gate and are not a new aggregate test total.

The PR is open, not merged. Approved normal automatic PR CI has started;
its result is not claimed passing here. This final follow-up records publication
only and changes no reviewed runtime, tests, package metadata or authority bytes.
The protected source remains HEAD `258beb6120a5e8c84d4f83b12a39427733301e57`
with the same 19 dirty paths and diff SHA256
`601b06a6481de59adce4fa3a2273291471828f2372d44acc847f31666bb3b867`.
Both worktrees are retained. AC19 is complete; task23175 remains In Progress
because broader hardware/release qualification is not complete. No native
package publication, manual workflow, new live test or local soak was performed.
