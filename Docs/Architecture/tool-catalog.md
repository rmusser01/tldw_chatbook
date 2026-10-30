# Tool catalog and providers

This document describes how tools become visible to agents and get dispatched: the `ToolCatalogRegistry` provider seam, the provider families (builtin, local `fs_*`/git, skills, MCP, canvas/library/profile), namespacing and shadowing, per-run composition, and the runtime tools the loop owns itself.

## Authoritative files

| File | Role |
| --- | --- |
| `Agents/tool_catalog.py` | `ToolProvider` Protocol, `ToolCatalogRegistry`, `BuiltinToolProvider`, `SkillToolProvider`, `ToolExecutionPolicy`, all runtime-tool schemas |
| `Agents/local_tool_provider.py` | `LocalToolProvider`, `LocalToolSpec`, admitted workspace roots model |
| `Agents/mcp_tool_provider.py` | `MCPToolProvider`, `MCPPendingCall` |
| `Chat/console_run_start_tools.py` | Console run-start MCP composition and conditional cross-run reuse |
| `Agents/library_tool_provider.py`, `library_rag_tool_provider.py`, `profile_tool_provider.py`, `canvas_tool_provider.py`, `raw_shell_tool_provider.py`, `virtual_cli_provider.py` | The remaining provider families |
| `Tools/tool_executor.py` | `Tool` ABC plus the two always-on builtins: `DateTimeTool` (`get_current_datetime`) and `CalculatorTool` (`calculator`, AST-whitelisted eval) |
| `Tools/local_tool_impls.py`, `git_tool_impls.py`, `patch_tool_impls.py`, `web_tool_impls.py`, `virtual_cli_impls.py` | Synchronous tool implementations |

There is no `AVAILABLE_TOOLS` global anymore (the old batch executor was retired). The source of truth is `_GATEABLE_BUILTINS` plus `ALWAYS_ON_BUILTIN_NAMES` in `tool_catalog.py`, gated by `[tools]` config keys.

## The provider seam

`ToolProvider` is the minimal protocol: `list_catalog` / `load_schema` / `invoke`. Optional protocols extend it: `ToolRecordProjectionProvider` (content-boundary projection for run records) and `PathAwareToolProvider.path_targets` (validated scopes for project-instruction activation — see [console-file-authority.md](./console-file-authority.md)).

### Registry mechanics

- `register_provider` appends and invalidates a snapshot cache; **registration order defines shadowing** — the first registrant wins on id/name collisions.
- One provider sweep builds an immutable catalog snapshot (`by_id`, `by_name`, `entries`). `AgentService` resets the cache at run start (per-run freshness doctrine — registries are never reused across runs).
- `invoke_by_name` is **the** dispatch choke point, in order: canvas authority checks → JSON-string argument coercion/repair → persona call-cap check (`RunToolPolicy`, refuses before dispatch) → ephemeral-session gate (unaudited sources refused) → `provider.invoke(tool_id, args)`. It never raises — it returns a `ToolResult`.
- `timeout_for(name)` honors duck-typed per-provider overrides; `execution_policy_for(name)` fails closed to `BOUNDED_ABANDONABLE`.
- `probe_initial_catalog` discloses full schemas only if every cumulative prefix fits `max_schema_tokens`, otherwise the agent discovers tools progressively via `find_tools`/`load_tools` (ADR-104 token-budgeted disclosure).
- Authenticated registration seams (`register_builtin_library_provider`, `register_canvas_provider`) are fail-closed against lookalikes: they require exact provider types plus exact authority objects.

### Namespacing

| Family | Tool id | LLM-facing name |
| --- | --- | --- |
| Builtin | `builtin:<name>` | bare name |
| Local | `local:<name>` | bare name; permissions resolve under the synthetic hub server `local:__local__` |
| Skill | `skill:<name>` | bare name |
| MCP | — | `mcp__<server>__<tool>` (deduplicated; can never equal a bare builtin name) |
| Runtime (loop-owned) | `runtime:<name>` | bare name |

## Per-run composition

The Console bridge composes a fresh registry per `run_reply`, in a fixed order: Builtin → Canvas → Local → virtual CLI → raw shell → Library (authenticated) → profile → Skills → MCP. MCP registers last and is therefore shadowed by every earlier provider. The composition returns the registry plus the run's `allowed_tools` set. Ephemeral sessions drop skill and MCP providers entirely.

## Local tools and file authority

`LocalToolProvider` exposes the `fs_*` family (`fs_list`, `fs_read`, `fs_write`, `fs_edit`, `fs_glob`, `fs_grep`), git tools, patch tools, and web tools. Local tools deliberately reuse the MCP permission model through the `local:__local__` hub server key (ADR-032), so they appear in the same permission store and approval cards. Their root model is the admitted-roots contract described in [console-file-authority.md](./console-file-authority.md): `None` = legacy single `workspace_root`; `()` = path tools stripped entirely; ≥1 root = per-alias specs with an injected `root_alias` parameter when multiple. Read-only composition filters every mutating spec — `fs_write`/`fs_edit`/`fs_patch`/todo tools are simply not advertised.

Tool results that exceed budgets spill into the chat's private scratch space (`tool-spill`, per-result cap 32 KiB, aggregate inline budget 256 KiB, 0600 files in a 0700 dir); the preview names the read-back path so `fs_read` can page it back with no new grant. The opaque absolute scratch locator is stripped from every model/log-bound result via the redaction-root seam.

`fs_write` is CAS-capable (`expected_sha256`/`expected_absent`, portalocker inode lock, `O_NOFOLLOW` descriptor writes) and the provider injects stale-write guards from the per-run ReadLedger — a write to a file the run has not (re)read refuses with "Re-read the file and retry".

## Skills as tools

`SkillToolProvider` lists skills from a per-run snapshot (`{name, description, argument_hint}`; the schema is a single free-form `args` string). Its `invoke()` **raises `RuntimeError` by design** — skill tools never dispatch through the registry. The Console bridge intercepts skill names before registry dispatch and runs them through the run-scoped skill runner, which:

1. re-verifies trust at render time (fingerprint digest comparison; mismatch → `SkillTrustBlockedError`),
2. renders the skill (`skills_service.execute_skill`, mode `"local"`),
3. intersects the skill's declared `allowed_tools` against the run's builtin+local names — narrow-only, a skill can never grant a tool the run lacks,
4. spawns the rendered prompt as a child run.

Script skills (`run_skill_script`) add three independent gates: the runtime policy launch rule, per-run trust re-verification against the trusted manifest, and a human confirm card whose "Always allow" grant is pinned to the fingerprint digest. Script execution is POSIX-only (sandbox support gate).

## Runtime tools

The loop owns several tools itself (ids `runtime:<name>`, schemas in `tool_catalog.py`): `spawn_subagent`, the fleet tools (`wait_agents`, `check_agents`, `send_to_agent`, message/steering tools, worktree merge/discard), `find_tools`/`load_tools`, `skill_file` / `install_skill` / `run_skill_script`, `fork_chat` / `new_chat` (ADR-150), and the run-log search tools. Fleet waits dispatch in-loop rather than through `invoke_tool` so they are not killed by the per-call timeout (see [agent-runtime.md](./agent-runtime.md)).

## MCP tools

`MCPToolProvider.compose_catalog()` builds the agent-visible MCP tools: kill switch → empty catalog; otherwise local external records plus the builtin inventory, one batched `effective_tool_states` resolution, `deny` states dropped, persona maximums enforced, names deduplicated. Invocation serializes on a per-provider lock (the stdio session is not proven multiplex-safe), peeks per-run approval stamps, applies the persona floor, and routes execution onto the main loop bounded by the configured tool timeout. The full permission decision lifecycle is in [mcp-hub.md](./mcp-hub.md).

The stock local MCP capability manifest reads and parses its registration source once per manifest call, sharing that AST across the tool, resource and prompt walkers. Every Console send's configuration capture reads the manifest on the UI loop, so the parsed tree is kept while the source file's identity (path, device, inode, size, modification and change times) is the same before and after the read, and the app's idle pre-import pass parses it once ahead of the first send (TASK-33620.15.5). A changed source file is parsed again on the next call, the walkers only read the tree, and every call still builds fresh, detached schema values.

The Console run-start helper may install a kept MCP composition into a fresh provider when the service, catalog fingerprint identity, connection states and provider composition key still match. Only a plain `MCPToolProvider` is eligible for this reuse. Incomplete inventory composition is not kept; a valid empty inventory can be reused. `install_composition()` clears approval stamps. This cache belongs to the run-start helper; the full registry and persona call caps remain run scoped.

## Config keys

`[tools]`: `read_file_enabled`, `list_directory_enabled`, `write_file_enabled`, `create_note_enabled`, `update_note_enabled`, `glob_files_enabled`, `grep_files_enabled`, `expand_document_enabled` (all default **false**), `web_deep_search_enabled` (double opt-in), `ask_user_enabled` (default on). Every gate is read through `coerce_bool_setting` — a quoted `"false"` in TOML is truthy and must not fail a gate open.

## Boundaries

- `Agents/` owns the loop, the provider Protocol, per-run composition, and the approval seams.
- `Tools/` owns synchronous implementations and workspace file-root context; it has no knowledge of the registry.
- `MCP/` owns the permission store, hub catalog normalization, and the external client/server (see [mcp-hub.md](./mcp-hub.md)).

## Verified gotchas

1. Skill tools must never go through `provider.invoke` — it raises by design; the bridge dispatches them before registry dispatch.
2. Registration order is shadowing order; MCP is deliberately last.
3. `local:__local__` resolver consults only `HIGH_RISK_TAGS = {"mutates","process"}` — the local specs' `reads`/`network` tags are deliberately inert markings; the real protection is the default-ask global plus in-workspace hard refusals.
4. The default `workspace_root` is the app cwd — launching from `$HOME` makes `$HOME` the confinement root, which is why the sensitive-path denylist is a mandatory second layer.
5. Registry composition is never cached across runs; resetting the catalog cache at the wrong moment would disarm the run's persona call caps.

### Scoped code review tools (ADR-281)

`Agents/code_review_tool_provider.py` supplies closed guide/prepare/read/submit/release
schemas. The bridge accepts a host-created `code_review_provider` in its shared
live/preview first-request builder. Reserved names reject lookalikes even when the
feature is disabled; ephemeral and existing persona/goal policies still narrow the
catalog. The provider uses actual `CurrentRunActor` attribution, and emits only
schema/status/size metadata to record projections. Source references and cursors
map to original admitted binding paths through `path_targets`; the existing
instruction ledger remains the guidance-delivery owner.

Advertised read views and submission kinds follow the host actor and live child
stage grant: primary reads source/report/index/summary and submits check actions;
reviewers read source and submit findings/coverage/completion; verifiers read
source/blinded batches and submit dispositions/check proposals/completion. Child
catalogs and schemas share role-specific guidance, including initial bounded
source listing at `relative_dir="."`, `side="base"` without a known source reference.
The store's canonical submission policy supplies the advertised kinds. Public wire
validators and runtime binding/role checks remain authoritative and unchanged;
changing a model's name or guessing an unadvertised view cannot grant access.

`Agents/code_review_guide.py` reads the two explicit packaged primary recipes via
`importlib.resources`. The whole canonical envelope (team references and setup
status included) fits 12 KiB UTF-8 plus the trusted live result-character ceiling;
overflow returns a body-free diagnostic rather than partial instructions. Guide
lookup exposes role-associated stable IDs, current display names, independent
focuses and configured provider/model identities from current preset rows/routes.
Only those identity fields are projected; endpoints, credentials, preset prompts
and sampling/fallback details are excluded. Missing/disabled references stay
explicit. Lookup cannot capture source, request destination
consent or launch children. Role-specific child briefs retain submission/evidence
requirements without primary orchestration or reviewer provenance in verifier tasks.
Preparation stores the common independent-review brief used by every reviewer;
report guide fingerprints hash that stored brief, with optional focuses appended
separately at launch. Both child roles receive the captured target kind and digest
without projecting target paths or URLs into their task metadata.

The canonical `AgentsSettingsPanel` stores one ordered team in
`[agents.code_review]` through the existing atomic TOML writer. Preset IDs remain
stable through rename/disable/delete, with current labels and explicit missing,
disabled and duplicate-route diagnostics. Focus/order/verifier drafts use separate
Save team/Cancel actions; load and cancel write nothing, and neither action changes
preset provider/model fields. Config reads/writes run off the UI loop. Saved team
changes affect later preparations; admitted slots remain frozen. The optional
`Docs/Examples/skills/multi-model-review/SKILL.md` is an uninstalled user-invocable
inline alias using normal Library trust, disabled for model invocation.

`Agents/code_review_service.py` accepts only grants retained by a live process-local
`ReviewStageIssuer`. Preparation freezes stable-ID preset definitions, explicit
run overrides, resolved routes/fallbacks and the guide under scratch-store quotas.
Its required host capacity and destination callbacks supply current admission
floors. `ConsoleCodeReviewCoordinator` derives stage intent from actual accepted
USER messages and narrows it to exact preparation fingerprints. Only messages with
stage intent (or the first answer to a native offer, which consumes it) are
recorded; each session keeps at most 64 and evicts its oldest request that never
led to review work. `compose_native_review_provider` returns before building the
lazy coordinator for an ordinary turn, composes nothing with `[tools]
code_review_enabled` off (`CODE_REVIEW_GATE_KEY`, hand-listed in
`all_tool_gates()` like `web_deep_search`), and without one local root refuses
only code-review wording or a stage control for a review the session holds.
Capture verifies each commit once, lists each pinned directory once per
`ReviewGitReader`, refuses a change set beyond the file quota before reading
entries, and holds the review-store lock only for its precondition checks (the
source guard may wait on the UI thread). Changed binary/symlink/submodule/over-size
files become path-named coverage gaps and withheld sensitive changes one unnamed
counted gap. `PreparedReview.coverage_gaps` carries the first eight (clipped,
`coverage_gaps_omitted` counts the rest) and a reviewer's first listing of the
review scope repeats them as page gaps within half its budget; the persisted launch
task carries only the gap count. A configured verifier
is frozen without authorizing verification: only a separately issued matching
verification grant adds it to preparation's finite schedule accounting.

Source pages contain context-control and caller-only submission revisions before
packing. Store create/publish/release admission fits the exact complete response
before allocation/publication/tombstoning. Findings receive stable slot-qualified
IDs across retries. Source/query bodies are bounded canonical JSON, with no adapter
wrapper or generic truncation used as recovery. Primary-only summary/report/index
readers page host envelopes, intact validated originals and separately captured raw
assistant output. The index carries every accepted original ID, its record reference,
attempt, completeness and coverage reference without inserting claim prose. Oversized
indivisible records produce an explicit model-read gap while the store retains their
intact bytes; cursor replays retain their fitted page or refuse narrower capacity.
Blinded verifier batches use the private original-to-unit map and checkpointed
ledger described in [agent-runtime.md](agent-runtime.md#blinded-verifier-request-checkpoints-adr-240-task-8).

Scoped launch uses the mutually exclusive `spawn_subagent(review_id,
review_slot_id)` form. `AgentService` claims a frozen slot, rechecks current route
and destination authority, and uses the existing inline/fleet admission and child
budget owners. Actual accepted run IDs are bound before child tools execute.
Repeated requests return the existing receipt or a pending refusal; uncertain
launches cannot produce another association. The runtime refreshes the primary
adapter's remaining-child reader from accepted native spending. A proven native
thread non-start retires that exact empty attempt and permits a fresh slot claim;
an exception after actual start preserves the association and spending. Inline
terminal receipts retain the actual run identity and status, including failure,
so replay does not convert a failed child into success or spend another child.

Each review child gets a private service adapter and closed catalog with source
read/role-owned submit tools. The adapter adopts the final hook-narrowed child
result cap before any tool, keeping pages and mutation acknowledgments complete. Runtime capabilities are separately denied, including
guessed names; `allowed_tools` alone is not that boundary. Both protocols reject
mixed scoped/ordinary spawn fields, even when provider schema adaptation omits
`oneOf`. Frozen definitions, sampling, endpoints and fallback candidates survive
preference edits, with live authority rechecked at launch and fallback selection.
Verifier feedback mode follows actual dispatch, including the inline kill switch.
The existing source path-target ledger delivers admitted nested guidance before
evidence; cycle projections retain only an opaque operation digest so distinct
source queries and cursor pages do not falsely appear to repeat one call.

Task 6 captures each nonstreaming assistant response before runtime/fleet shortening;
terminal settlement closes capture with an empty marker, so final text is not charged
twice. Thinking metadata and tool/source responses are not raw assistant output.
Existing credential/private-key/reasoning filtering applies without the diagnostic
length cap, with explicit withholding; storage failures/overflow preserve earlier
chunks and remain incomplete. Raw, structured, host receipts and cached pages share
the existing reservations and participant/aggregate ceilings. Host sequences never
consume a participant submission revision.

`capture_review_output` and `settle_review_run` are host-only ports. Sealing requires
an actual successful native terminal receipt, a completed capture receipt and valid
accepted `reviewer_complete` coverage agreeing with the findings. Refused submissions
remain format gaps until that same submission identity is corrected in the live run.
New participant writes after native settlement refuse. Cancellation/abandonment does
not invent a drained capture; late model callbacks cannot append. Retries retain
previous attempt originals/receipts with distinct streams. Exact terminal-trace loss
prevents sealing, including when a durable terminal status row exists.

Host envelopes retain frozen target/guide/preset fingerprints and actual dispatch
routes/fallbacks. Review report bodies stay in session scratch: child durable rows
omit content, full run-log callbacks omit review bodies, and inline/fleet results
carry report lookup references. Generic fleet result caps remain unchanged. Native
report/export UI and continuation admission remain later owners; the host store's
`read_record` returns the full stored original under its live context lease.

Task 8 adds verifier `read(view="batch")` through the existing review tool provider.
The host service supplies its last admitted request budget/estimator; the model cannot
override those values. Whole blinded pages and source evidence obey independent input
and transport limits, and current source permissions are checked again before delivery.
Batch cursors replay the same page; the next batch waits for independent dispositions
for every delivered unit. Accepted submissions use the existing immutable chunk and
atomic manifest publication before the history checkpoint advances. An exact replay
returns its previous acknowledgment; revised verdicts require a new explicit attempt.
`expand_dispositions` is primary-only and restores every original through the private
map. No original attribution or multiplicity enters verifier pages or generic logs.
Executable check orchestration and the native full-report reader remain later slices.


### Disposable review checks and native reports (ADR-281, slice 9)

`Agents/code_review_checks.py` holds only per-review proposal/claim/receipt facts
and the retained native execution owner. `ReviewStore` owns their quota, streams,
leases and lifecycle; there is no separate scheduler, registry or job database.
Primary check actions have an independent `(primary run, review, verifier attempt)`
stream. Neither primary actions nor native receipts advance verifier revisions.
The actual fleet `wait_agents` predicate returns a bounded `check_requested` notice
without collecting the live verifier. Its proposal-specific `verification_batch`
read retains its admitted operation charge and finite verifier deadline/cancel
context. Capacity failures signal unstarted work before fleet draining. Inline
launch atomically acknowledges an unavailable receipt and never waits for a parent.
The launch branch, rather than the preparation settings, fixes this mode.

**Stock RawCliRuntime cannot execute review checks.** Its executor has no aggregate
byte/file write boundary. The host-only `review_check_boundary_factory()` discovers
an optional `reserve_review_boundary(parent, sources, max_bytes, max_files)` method
on the actual external executor. Only an executor that enforces the aggregate
allowance before copies/generated files exist can implement this contract. It
returns a private `root` and `release() -> bool` cleanup proof. No model argument,
post-execution directory scan, generic result string or per-file size limit grants
this capability. Production absence resolves `quota_enforcing_executor_unavailable`
before any copy/specification/process launch. Fault-controlled tests supply a
finite scripted external boundary through the real RawCliRuntime admission path;
they do not prove stock subprocess sandboxing or live quota enforcement.

Claims reserve source copies, generated bytes/files and bounded output against the
same review. The exact normalized shell specification still uses existing process
permissions. Native `commit_launch` acceptance binds the invocation ID; only the
raw provider's actual terminal result can register evidence. Accepted/uncertain
work cannot be retried or declined as unstarted. Quota/scratch leases remain charged
until the actual owner confirms cleanup. A closed verifier cannot append a verdict;
late native cleanup does not reopen its disposition stream. Pending receipts have
reserved publication headroom. Check evidence pages preserve their identity and
cursor, use the existing complete-response packer and the verifier's 1024-token
read ceiling, and mark a receipt read only after its final page.

`CheckProposal.verification_unit_ids` names related claims, not a proof that
execution is necessary. A valid source-only decision remains possible after an
unavailable check. `CheckEvidence` requires an actual executed receipt for the same
attempt/target, fully delivered to that live verifier. Whether a claim semantically
requires execution cannot be established by the schema; the recipe requires an
unresolved disposition and `next_check` when indispensable execution is unavailable.

`ConsoleCodeReviewModal` accepts only an owning service and opaque review ID.
Native reads use store leases off the UI loop and show literal, paged originals,
coverage/routes/raw output and host-expanded original-to-unit dispositions with
captured source and check evidence. Full originals remain accessible beyond model
transport limits. The controller enters Textual through `call_from_thread`, mounts
or reuses one reader for the exact active session/review, and acknowledges the
original only after the reader displays it. Closed/background/unavailable UI does
not acknowledge presentation. Already-authorized verification proceeds after this
handoff; there is no second consent prompt. Transcript references state session
retention and cannot restore expired bundles from saved chats. `u` reopens the
latest retained review for the active chat. Explicit Export reuses the existing
save-Markdown destination modal/writer. Native confirmed Discard refuses active
leases and displays cleanup pending; a model `code_review_release` call cannot
supply the native discard intent.

The retained selected-source guard uses the same authority predicate from both
threads: worker calls enter through Textual's owning-loop dispatcher, while native
wake validation already on that loop calls the predicate directly. Calling
`App.call_from_thread` on the app loop is rejected by Textual and would incorrectly
mark the review source unavailable. Pending accepted-process notifications are
cleared under the publication lock before another finite condition wait; acceptance
does not turn the waiter into an immediate polling loop.

The optional `RawCliRuntime.execute(admission_scope=...)` callback is host-only and
absent for ordinary calls. Its private check owner validates source authority before
acquiring the review publication lock, then rechecks live review/attempt/deadline
under that lock. The existing runtime admission lock precedes the review lock;
the runtime state lock is acquired inside only for its ordinary final authority
check and `commit_launch`. Native acceptance is recorded before the review lock is
released. Close publishes its fence and revokes the exact scratch generation without
waiting on the publication lock. Admissions entering after that fence refuse;
already-entered admission remains native-owned until commit/settlement and only
its actual callback records acceptance. Neither approval, process execution nor
runtime drain holds the review lock. A prepared invocation
keeps its lease through context exit; a never-prepared claim can be cleaned by actor
finalization. Accepted or uncertain work keeps the actual native cleanup owner.
The waiter rechecks store liveness every iteration even while its original read
lease remains held, so closing an actively waiting review terminates that wait.


Qualification is recorded in the [multi-model review QA record](../superpowers/qa/2026-10-18-multi-model-code-review/README.md).
The deterministic Console journey covers native original presentation and complete
nonzero expansion. Scripted successful gateways do not establish live review quality.


Final-review corrections: model-bound raw/structured reports, finding indexes,
summaries and cached cursor replays check current source permission outside the
publication lock, then revalidate exact context/binding/revision and every captured
path against current exclusions before returning bytes. Report bodies may quote
any captured source, so revoking one such path refuses the aggregate report.
`verifier_complete` records intent only after all its ledger units were delivered
and independently disposed with no request gaps. Verifier seals use that accounting
plus actual successful terminal/capture receipts; reviewer coverage is separate.
Reasoned unresolved dispositions can finish accounting. Host-filled unprocessed
units never create completion intent, and late terminal writes remain refused.
