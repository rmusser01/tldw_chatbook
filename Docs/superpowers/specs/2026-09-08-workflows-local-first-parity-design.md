# Workflows: local execution and portable authoring

- Status: Approved by the user on 2026-09-08; no runtime or UI implementation in this task.
- Date: 2026-09-08
- Task: [TASK-32077](../../../backlog/tasks/task-32077%20-%20Design-portable-Workflows-editor-and-local-execution-parity.md)
- ADR: [ADR-138, accepted](../../../backlog/decisions/138-portable-workflow-definitions-and-local-execution.md)
- Catalog: [Server parity matrix](2026-09-08-workflows-parity-matrix.md)
- Server baseline: `rmusser01/tldw_server` dev, `6cd2745f696af04668a61c20b84ab8a9e69ca5e4`, checked through GitHub on 2026-09-08.
- Local evidence: current Chatbook checkout, HEAD `22aa927f0287e84c76ab39836a05a028f2a73e82` plus existing working changes; source inspection, not execution evidence.
- Revision: the user approved the revised detailed contracts and numeric safety defaults on 2026-09-08; retains the 21-step v1 target and introduces a smaller first delivery milestone.
- Review: [independent design and contract findings](../../../.impeccable/critique/2026-09-08T19-54-42Z__2026-09-08-workflows-local-first-parity-design-md.md).

## 1. Agreed product direction

Workflows prepares reusable procedures, executes them locally, and exchanges their definitions with tldw_server and other people. Its primary audience is a terminal user turning repeated source-to-output work into an inspectable procedure.

The user approved:

1. A workflow library column, a step navigator beside it, and a large main canvas.
2. Overview and focused-step views of the same workflow.
3. A Cover Flow-inspired selected card with neighboring previews.
4. One continuous editable form with independently collapsible sections.
5. Sequential orchestration in v1, branching in v2, and parallelism in v3.
6. Maximum reasonable server parity and portable sharing/sync as design goals.
7. Local execution without a connected tldw_server as a v1 requirement.

The detailed Chatbook contracts and minimum adapter set in this document are approved design inputs. The cross-repository sync extension still requires a supporting counterpart contract and implementation; Chatbook approval does not establish server support. None of these decisions claims that the capabilities already exist.

## 2. Existing contracts and gaps

The current Chatbook Workflows screen is a destination shell with an active-work Console handoff. It has no general definition editor or local workflow runner.

The server API owns the execution definition: name, version, description, tags, inputs, metadata, ordered steps, and an optional completion webhook. Each step has an ID, optional name, type, config, retry, timeout, and optional success/failure/timeout routing. Its ordinary execution path advances in array order. `prompt` renders text; `llm` calls a model. Earlier step outputs enter the expression context under their stable step IDs.

The web graph editor's nodes/edges JSON is a separate representation. Its current serializer drops the optional version and replaces metadata. This is an interoperability gap to fix deliberately; treating that file as an API definition would lose meaning.

The server exposes 130 registered step names and a step-types endpoint with descriptions, examples, basic configuration schemas, and replay capabilities. Some schemas are generic objects. Registration is not proof of dependency availability or local execution parity.

Definitions are currently private. New versions are immutable database records, but integer IDs and name/version uniqueness do not provide portable identity or edit-conflict protection. Neither server Sync v2 nor Chatbook's envelope adapters currently supports a workflow domain.

Primary references at the pinned server commit:

- [Definition and run schemas](https://github.com/rmusser01/tldw_server/blob/6cd2745f696af04668a61c20b84ab8a9e69ca5e4/tldw_Server_API/app/api/v1/schemas/workflows.py)
- [Workflow API, discovery, preflight, versions, approvals](https://github.com/rmusser01/tldw_server/blob/6cd2745f696af04668a61c20b84ab8a9e69ca5e4/tldw_Server_API/app/api/v1/endpoints/workflows.py)
- [Execution order and template evaluation](https://github.com/rmusser01/tldw_server/blob/6cd2745f696af04668a61c20b84ab8a9e69ca5e4/tldw_Server_API/app/core/Workflows/engine.py)
- [Web editor serialization](https://github.com/rmusser01/tldw_server/blob/6cd2745f696af04668a61c20b84ab8a9e69ca5e4/apps/packages/ui/src/store/workflow-editor.ts#L600)
- [Current sync domains](https://github.com/rmusser01/tldw_server/blob/6cd2745f696af04668a61c20b84ab8a9e69ca5e4/tldw_Server_API/app/core/Sync/v2/models.py#L21)

The general `/workflows` API is this design's target. The separate `/chat-workflows` question/dialogue templates retain their own format; import detects them and explains the distinction rather than guessing a conversion.

## 3. Screen composition

```text
Workflows / Research brief       Draft recovered - Based on v3
Next run: [Local]  Setup ready   [Save revision] [Validate] [Run]
-------------------+----------------------+-----------------------------
WORKFLOWS          | CURRENT WORKFLOW     | [Overview] [Step]
Search...          | Overview             |
                   | Inputs               | Previous: Gather sources
> Research brief   | Requirements         | +-------------------------+
  Daily digest     | Versions             | | 02 - Summarize          |
  Review notes     |                      | |                         |
                   | 01 Gather sources    | | > Inputs                |
Source: [All]      | >02 Summarize        | | v Action                |
                   | 03 Review           | |   Model, instructions   |
                   | 04 Save note        | | > Outputs               |
                   |                      | | > Execution             |
                   |                      | | > Advanced              |
[New] [Import]     | [Add step]           | +-------------------------+
                   |                      | Next: Review
-------------------+----------------------+-----------------------------
Results: Run 42 / v2 / Local / Earlier revision / Open in Console
```

This is an ASCII design sketch, not evidence of a rendered Textual implementation. The product's semantic theme, density, and keyboard conventions remain authoritative.

### Library

Rows show name and a compact source/state summary: local, linked to a named server, imported copy, changed, or conflict. Source and synchronization state are independent. A workflow can be local and linked, or imported and subsequently linked. Filtering never moves a draft to a different authority.

New opens a named empty draft with Add first step. Import opens a review. New from template is another explicit creation action. An empty list offers these actions without requiring a server connection.

Add step opens a searchable, task-grouped chooser: Text and models, Sources and data, Tools and outputs, and Run controls. Rows include a friendly action label, canonical type, short input/output example, and availability for the next-run target. For example, Render text (`prompt`) and Call a model (`llm`) are distinct. Ready/setup-required actions appear first; Show all retains discovery of server-only and future-version types. Inserting a setup-required step is allowed into a draft with a visible issue; unavailable local execution is never implied. Choosing an item explains requirements before insertion and returns focus to its first required field.

### Navigator

Overview, Inputs, Requirements, and Versions precede the ordered step list. The selected step marker is distinct from its execution status. Names can change without changing step IDs. Validation problems appear next to the affected step and remain understandable without color.

Selecting a step reveals its card in Step view or scrolls its box into view in Overview. Activating a selected overview box opens its editor. Returning to Overview restores diagram position. Add before/after, Duplicate, Move earlier/later, and Delete are visible contextual actions. Deleting or moving a referenced step previews the affected references; no dangling dependency is silently accepted.

Group contextual actions into Insert, Move, and Delete rather than six equally prominent buttons. A new or duplicated step gets a new stable ID; renaming never rewrites references. Dependency previews identify each affected consuming field and offer Repair or Cancel. Opaque expressions that cannot be analyzed block an unsafe reorder until explicitly resolved; a text substitution is not sufficient dependency analysis. V1 Overview is an automatically arranged linear diagram, not a second drag-to-wire authoring model.

### Canvas and continuous form

The selected card owns the editable controls. Neighboring cards are non-editable summaries with title, type, and a short description of inputs/action. Clicking a neighbor selects it. At wide widths they may flank the selected card; narrower layouts replace them with labeled Previous/Next rows. Motion is optional and must not delay keyboard navigation.

Sections are independently collapsible, never an exclusive accordion:

| Section | Initial state | Content |
| --- | --- | --- |
| Inputs | Collapsed when configured; open when required data is missing | Fixed values, workflow inputs, and references to earlier outputs |
| Action | Expanded | Type-specific controls using the server's config names and semantics |
| Outputs | Collapsed | Documented result fields, reference insertion/copy, and selected-run output preview |
| Execution | Collapsed | Step retry and timeout; v1 has no alternate-route editor |
| Advanced | Collapsed | Remaining supported fields and an advanced JSON configuration editor |

Collapsed summaries reveal changed defaults and missing values. Activating a validation issue expands and focuses its section; passive validation updates markers without stealing typing focus. Next/previous issue actions navigate the same issue list. All sections may remain open. Section state and scroll position survive step navigation during the editing session. Long instructions can expand into a larger editor and return without losing the draft.

Inputs and Outputs are presentation groupings over the canonical config and result contract; they do not invent top-level per-step input/output fields. Model selection is explicit or bound to a workflow input, such as `{{ inputs.model }}`. A workflow-wide convenience control must write an explicit compatible representation rather than introduce an undocumented runtime default.

A field update preserves unrepresented config keys and metadata. Unknown or newer semantic fields that cannot be safely edited make the relevant document read-only; they are never discarded by model reserialization. Schema discovery refresh must not rewrite an open draft.

At an eligible consuming field, choose Fixed value, Workflow input, or Earlier step output. The reference picker shows the producer name, stable ID, field path, documented type, and compatible fields; v1 only offers earlier producers. Unknown result schemas remain explicitly unverified and require runtime validation. Selection inserts the canonical expression and restores the consumer's cursor. Copy result as fixed value is a separate action that previews the historical value and its provenance; it never masquerades as a live reference. Workflow Inputs is labeled Run inputs in navigation to distinguish it from a step's Inputs section.

### Draft durability and result provenance

The document service privately retains a draft buffer keyed by profile, workflow identity, and base revision, independently of immutable saved revisions. It includes raw advanced-editor text even while invalid, the last valid structured projection, and a dirty generation. Invalid JSON cannot update the structured projection, run, publish, or sync; the UI explains that forms still show the last valid structure and prevents form edits from overwriting the invalid buffer. Draft writes are debounced to 500 ms; navigation, version selection, and orderly shutdown flush and await the current generation. Only a successful durable write earns Draft recovered/saved locally feedback. Abrupt termination may lose the visibly pending last interval; no stronger guarantee is shown.

If a flush fails, preserve the in-memory buffer, show Not saved locally with Retry, and keep the editor active unless the user explicitly confirms leaving without recovery. Save revision validates the document's structural integrity and creates one immutable revision; missing destination setup can remain a clearly labeled requirement. Unsupported imported definitions may be preserved but not rewritten through an incompatible serializer. Save and run additionally requires run admission. Escape never discards work. Discard draft is explicit and confirmed, returning to the base revision. Ordinary workflow navigation restores that workflow's draft; selecting a historical revision opens inspection first, and Edit as new revision explicitly creates a draft without replacing another dirty draft.

Results are keyed by workflow identity, saved revision, run ID, target identity, step ID, and attempt. The drawer shows those labels, plus Results are from an earlier revision when the editor has diverged, and offers Inspect run definition. Switching workflows restores that workflow's selection or an empty result state; it never carries unrelated results forward. Changing Next run target affects only future admission and its capability indicators. No run means documented output shape, not fabricated example execution. Follow execution changes the highlighted running step only when enabled; it does not rebind an editable draft. Successful completion offers Open result for the actual retained note/artifact, with a clear missing/deleted-result state.

### Width and focus

Allocate the main editor first. The two navigation columns collapse independently and remain available through labeled selectors. Below the width needed for both columns and a useful editor, collapse the workflow library first; at small widths show one region at a time with `Workflow / Step` breadcrumbs. At 60x20, all core operations must remain reachable through compact controls and scrolling.

F6 retains the global pane cycle, Tab traverses active controls, and Escape unwinds an expanded editor or overlay before changing context. Proposed single-letter actions apply outside text inputs only. The implementation must check global bindings and expose only working context-specific footer actions under ADR-031.

| Transition | Visible regions and focus rule |
| --- | --- |
| Three panes fit | Library, navigator, and editor register their visible regions with the existing global F6 cycle; Tab stays in the normal visible-control order. |
| Library collapses | A labeled Workflow selector opens the library as an overlay. Its close restores the opener; choosing an item durably retains the old draft before loading the new one. |
| Single-region mode, including 60x20 | Workflow and Step selectors remain reachable above the editor. Selecting a step closes its navigator and focuses that step's prior field or first required field. Results opens as a full-region inspector with Return to editor. |
| Resize hides the focused region | Store its logical field/cursor anchor and move focus to its visible selector. Hidden widgets leave Tab/F6 traversal. Widening does not unexpectedly steal focus; explicit reopening restores the anchor. |
| Picker, expanded text editor, or results closes | Escape closes the topmost surface, keeps buffered edits, and restores its still-eligible opener; otherwise use the visible editor heading/selector. |
| User activates a validation issue | Reveal the owning region, select the step, expand the section, and focus the exact field. Passive validation never performs this transition. |

Neighbor previews support Enter/Space activation as well as pointer selection. An overview box is a selectable list-like item with the same step label and status available without diagram interpretation. Long forms and results have distinct labeled scroll regions; status and actions remain reachable without horizontal scrolling at the minimum terminal size.

## 4. Local execution and offline behavior

Local means the workflow scheduler and adapters run inside Chatbook. It does not by itself mean every selected provider is offline: a local workflow may explicitly use a cloud model, internet search, or remote MCP service. Requirements separately reports external connections. Fully offline execution is possible with local files/data, installed models, and local tools, without tldw_server or external internet access.

Run target is selected for the whole run: Local or a configured server. There is no automatic per-step offload or fallback. A local run that lacks a dependency stops at preflight with a named remediation. Remote-required steps can still be inspected and exchanged.

The proposed architecture follows the existing local Research service/engine division:

1. A workflow document service owns drafts, immutable saved revisions, bindings, imports, and conflict resolution.
2. A local run service is the single writer for run, step, attempt, approval, and artifact-reference state.
3. A UI-independent sequential engine calls adapters and requests transitions through that service. Textual owns neither execution nor persistence.
4. Adapters translate canonical server config/result shapes to existing local services and normal tool permissions. They do not import the server's FastAPI application, database layer, or scheduler.
5. A remote client implements the same UI-facing operations using the explicitly selected server. Capability differences remain visible.

Proposed homes are `Workflows/` for domain services and execution, `DB/Workflows_DB.py` for private persistence, and `UI/Workflows_Modules/` for region widgets. These are proposed files, not existing APIs. The store participates in the app's private-data and backup owner registries; retained user results are application data, not diagnostic logs.

### Run lifecycle

Run captures a saved definition revision and its execution settings. If the draft has changes, Run opens a Save and run confirmation with a visible revision summary. A failed save never starts execution. Ongoing edits remain in the draft until another explicit revision save and cannot mutate the active run.

V1 executes one authored step at a time in array order. Retry and timeout semantics must match the admitted server contract. Non-linear routing, explicit branch nodes, map/parallel nodes, and unsupported nested orchestration block local v1 execution; imported documents remain preserved and inspectable. A straight-line explicit success chain may be represented when it is equivalent to array order and is updated consistently by a user-approved reorder.

Pause means stop before the next step; current work may finish. Cancel requests adapter cancellation and prevents later steps. A non-cancellable or interrupted side effect is recorded as needing review, never shown as undone. Human-review steps park the run and can survive an app restart. Approval identities require a valid destination mapping; an imported server user ID is not implicitly the local user.

At startup, interrupted running attempts become Needs review. Completed steps remain recorded. Resume proceeds only from a proven safe checkpoint; a retry of an uncertain external write requires explicit review. V1 does not promise exactly-once effects or transparent resume inside a model/tool call.

### Admission, exact execution fields, and recovery

Admission builds a private run manifest containing the immutable definition revision, runtime target/profile identity, adapter contract revisions, non-secret input values and binding selections, limits, and a unique launch-operation ID. The target is not re-read from a newly selected app profile mid-run. Capabilities and ordinary permissions are rechecked against this captured authority before an effect; revocation or unavailable identity parks the run for review, never redirects it. Setup ready means static requirements passed, not that future generated values or remote calls are guaranteed to succeed.

| User control | Canonical field and admitted meaning |
| --- | --- |
| Additional attempts | `step.retry`, an explicit non-negative integer; zero means no automatic retry. Do not write `config.retry` as a substitute. |
| Attempt timeout | `step.timeout_seconds`, explicitly authored as a positive whole number of seconds; new steps default to 300. Imported zero, null, fractional, or omitted values require a displayed effective-value decision before local admission rather than accidental coercion. |
| Provider/adapter request timeout | A separate type-specific `step.config.timeout_seconds` only where that adapter contract supports it. Show its purpose and effective relationship to the enclosing attempt deadline; it does not extend that deadline. |
| Human response deadline | For human/approval steps, `step.config.timeout_seconds` is the response wait, distinct from the short adapter invocation. Omitted or non-positive means no response deadline in the admitted server profile. Persist the resolved absolute deadline; an app restart does not reset it. V1 expiry fails the wait and blocks downstream steps; alternate routing remains unsupported. |

These mappings follow inspected [engine attempt handling](https://github.com/rmusser01/tldw_server/blob/6cd2745f696af04668a61c20b84ab8a9e69ca5e4/tldw_Server_API/app/core/Workflows/engine.py#L735), [retry policy](https://github.com/rmusser01/tldw_server/blob/6cd2745f696af04668a61c20b84ab8a9e69ca5e4/tldw_Server_API/app/core/Workflows/engine.py#L1645), and [human deadlines](https://github.com/rmusser01/tldw_server/blob/6cd2745f696af04668a61c20b84ab8a9e69ca5e4/tldw_Server_API/app/core/Workflows/engine.py#L1730). Discovery schemas alone are not the authority for field placement. Each adapter's conformance fixtures document its admitted subset and any intentional safety/privacy difference; do not silently copy test-only shortcuts or permissive server behavior.

An automatic retry requires a retryable failure, remaining attempt/time/token budget, and replay-safe behavior or verified destination idempotency. An uncertain write, an unknown replay contract, or a still-live worker parks the run instead. Keep physical worker ownership until it actually exits: a timeout or cancelled database row does not release its execution lease. Late completion remains evidence on the original attempt and cannot advance a cancelled run. Refuse overlap even after manual Retry until ownership is resolved. Waiting for a human does release active execution capacity once the adapter invocation has ended.

Persist the launch-operation ID before dispatch. Repeated activation or transport retries reuse it and the same payload; explicit Run again creates a new operation. For remote runs use the API idempotency key and reconcile uncertain responses against that destination before offering another launch. If the server cannot establish the outcome, show Launch outcome unknown and require review rather than create another key automatically.

Human decisions apply once to an identified run, step, wait generation, and actor. Expiry, cancellation, or a superseding wait invalidates a late decision. On restart, overdue waits expire before resume; pending waits remain visible. Store only credential binding references, never their resolved secret values in the manifest. Reacquire the same bound credential from the destination's credential owner; a missing or changed account requires explicit reauthorization. Refreshing a credential for the same authorized account does not switch providers or resource bindings. An API secrets object is not proof that every adapter consumes it; secret-reference paths must be demonstrated per adapter.

### Initial local safety profile

These are approved configurable workflow-local defaults, not server or machine-wide limits. The effective target's stricter limits win. Imported definitions remain inspectable when they exceed the local profile; execution requires an explicit supported limit change, never silent truncation or a lower retry count.

| Bound | Initial default | Exhaustion behavior |
| --- | --- | --- |
| Active workflow execution | One active run per Chatbook workflow runtime, retaining its slot while its adapter has any live owned work; no automatic queue in the first milestone | Refuse another launch with a link to the active/stopping run. Paused/human-waiting runs can resume only after reacquiring capacity. |
| Definition size | 100 authored steps and 2 MiB canonical JSON | Block local admission; separately bounded preview explains the limit. |
| Run inputs | 10 MiB serialized non-secret inputs | Reject before copying into the run store or execution context; large file resources stay references rather than embedded input blobs. |
| Attempts | At most 3 additional attempts per step | Require an explicit settings change for a higher imported value. Reviewed manual retries still consume the same run budgets. |
| Active run time | 60 minutes cumulative execution time, excluding only parked waits/pauses with no active work | Stop admitting later attempts; retain results and request review. An already-live worker remains owned until it exits. |
| Model work | 100,000 input-plus-output token admission units per run | Reserve estimated input plus explicit maximum output before each call; retain conservative reservations when usage is unavailable. Refuse an unbounded call. This is not an exact dollar-cost guarantee. |
| Inline outputs | 1 MiB serialized result per step; 100 MiB aggregate per run | Use an admitted artifact-backed result shape when supported; otherwise stop with Output limit exceeded rather than change the canonical result type or silently truncate. |
| Workflow-owned artifacts | 1 GiB per run; 5 GiB total retained workflow-owned artifacts | Account incrementally during writes and stop before admitting more data. Preserve earlier results; incomplete files are marked incomplete. Cleanup is explicit and previewed. |

These counters survive pause/resume/retry and restart. Byte bounds use UTF-8/actual artifact bytes and apply before unbounded allocation, with incremental checks for streamed values. Adapter/provider/model hard limits and shared tool-capacity owners remain authoritative; the workflow layer does not claim to bound independent Console runs or external processes. A user-approved limit increase is a recorded execution-policy amendment, not a mutation of the saved definition; resetting counters requires an explicit new run. Record exact token usage when available without mistaking missing usage for zero.

Remote runs show only limits the selected server actually enforces. The client cannot promise intra-run token/time caps from polling alone or after disconnect; a requested unsupported remote safety requirement blocks admission until explicitly revised. A client observation timeout never means the remote run stopped. Remote cancellation remains a requested operation until confirmed by that run's authority.

Workflow artifacts exclude domain-owned notes, media, and prompts: deleting a run must not delete those records. There is no automatic pruning of referenced run artifacts, saved definitions, or recoverable drafts. Partial writes and cleanup obey the existing private-path and ownership rules. Payload-bearing `log` steps belong in private run results/events, not ordinary persistent application logs; this privacy difference from server logging is explicit in their adapter contract.

The selected card stays under user control. Follow execution is optional. The results drawer provides step status, output/artifact previews, errors, and retry context; Open in Console follows the exact run. Normal persistent diagnostics exclude prompts, credentials, and result bodies under ADR-029.

## 5. V1 adapter scope and capability reporting

The complete pinned catalog is classified in the companion matrix. The retained v1 minimum target contains 21 names: `prompt`, `llm`, `rag_search`, `notes`, `prompts`, `chunking`, `media_ingest`, `pdf_extract`, `mcp_tool`, `tts`, `stt_transcribe`, `wait_for_human`, `wait_for_approval`, `delay`, `log`, `template_render`, `json_validate`, `csv_to_json`, `json_to_csv`, `regex_extract`, and `text_clean`.

This is the user-approved target to retain through the revision, not a claim that these adapters are implemented or a cap on v1 parity. The first end-to-end milestone below is smaller than this full v1 target. Other sequential action types can join v1 once their adapters pass the same gates. Branching remains v2 and map/parallel orchestration remains v3 regardless of adapter count. Nested workflow calls and scheduling need their own lifecycle contracts before admission.

Support is assessed per configuration as well as by type. For example, an implemented local RAG adapter must reject an unsupported reranking option rather than ignore it. Textual-friendly forms may cover common fields first, but a richer form never increases runtime capability by itself.

Each installed adapter reports a supported contract revision, field/operation coverage, required dependencies, external connections, and replay/cancellation behavior. The user sees a concise conclusion: Ready locally, Setup required, Requires server, or Unsupported in this version. Unavailable functionality is explained in Requirements. Offline discovery uses a bundled versioned catalog; connecting to a newer server augments discovery without silently changing local support.

## 6. Portable definitions, resources, and sharing

Canonical `.workflow.json` contains the server API definition. Preserve step IDs, ordering, config, routing, version, metadata, and meaningful unknown config keys. View preferences and native graph coordinates stay in a separate local view record; foreign editor metadata is preserved if present.

Propose a shared, versioned `metadata.tldw_workflow` namespace containing a global workflow UUID, immutable revision UUID, parent revision UUIDs, and portable requirement declarations. This namespace must be adopted across clients before it is treated as established shared behavior. Server integer definition IDs remain installation-specific references. The definition's numeric version remains its display/API version, not a global identity or optimistic-concurrency token.

Legacy imports with no portable identity become new local workflows. Re-import never matches solely by title. Import offers New copy by default; Update existing is available only for an explicit verified identity/revision match and shows a diff. Sharing a reusable copy forks its identity and records provenance. Syncing one's linked workflow preserves identity.

Requirements use logical names for model selections, tools, collections, notes/prompts, files, and approval recipients. Destination bindings map those names to actual local/server objects and credentials. Resource IDs with the same integer on different installations are not assumed equivalent. Missing resources do not trigger silent substitution.

### Executable input and binding contract

The proposed namespace uses `format_version: 1`, `workflow_id`, `revision_id`, `parent_revision_ids`, optional `input_schema`, and a `requirements` object keyed by logical requirement name. Each requirement declares its kind, requiredness, and owned `input_keys`; one input key cannot belong to two requirements. The client validates this structure before resolving it. This namespace describes client behavior; the current server does not resolve it automatically.

For ordinary inputs, use the saved definition's `inputs` as reviewed literal defaults, overlaid by explicit run-form values. Overlay is shallow by key: an explicit object/list replaces the whole default value, and explicit null is not omission. Validate the result against the optional input schema without silently converting lists to strings. Missing required values block admission. This default merge is performed by the client and sent in full as `RunRequest.inputs`; do not rely on the current server to merge definition defaults.

Requirement-owned keys are filled only from the selected destination's binding resolver, not from imported defaults or free-form run overrides. Changing one requires the mapping picker, resource validation, and renewed review of affected authority. Their resolved non-secret values are added to the run input object and private manifest, not the portable definition. Bindings resolve before admission; references to earlier outputs resolve before the consuming step and undergo type, path, capability, and permission validation again. Dynamic effects whose destination is not yet known cannot receive blanket pre-approval at initial preflight.

Credentials stay in the existing credential owner and are acquired only for the captured provider/account or tool connection. Use `RunRequest.secrets` only for an adapter with a tested supported secret-reference contract. In particular, the inspected server `llm` adapter does not establish a universal mapping from that object to provider API keys: v1's baseline uses destination-managed provider credentials, and unsupported per-run credential injection is refused. Never place a secret in ordinary `inputs`, metadata, or a persisted run manifest to make an adapter work. Secret-entry widgets are excluded from draft recovery buffers.

Expressions retain the server's canonical syntax, such as `{{ inputs.summary_model }}` and `{{ prepare.text }}`. Admitted pure dotted-path expressions preserve their JSON value type; mixed text renders as text. Unresolved required expressions block the consuming operation rather than being sent literally to a model or tool. More elaborate expressions require an explicitly supported parser/evaluator contract; preserving an imported expression is not proof of executable support. These safety restrictions are recorded conformance differences, not silent modifications to the saved document.

### Worked two-destination contract fixture

This source-shaped fixture demonstrates binding and default materialization; it is not evidence of an executed local adapter. The same definition is retained for both destinations:

```json
{
  "name": "Portable brief",
  "version": 1,
  "inputs": {"style": "three bullets"},
  "steps": [
    {
      "id": "prepare", "type": "prompt", "retry": 0, "timeout_seconds": 300,
      "config": {"template": "Summarize as {{ inputs.style }}: {{ inputs.source_text }}"}
    },
    {
      "id": "summarize", "type": "llm", "retry": 0, "timeout_seconds": 300,
      "config": {
        "provider": "{{ inputs.summary_provider }}",
        "model": "{{ inputs.summary_model }}",
        "prompt": "{{ prepare.text }}",
        "max_tokens": 512
      }
    }
  ],
  "metadata": {
    "tldw_workflow": {
      "format_version": 1,
      "workflow_id": "ba9e62aa-9859-4d42-98dc-26e3b10cdba0",
      "revision_id": "b42d45fb-dc91-4645-a6d9-a72c589282bd",
      "parent_revision_ids": [],
      "input_schema": {
        "type": "object",
        "properties": {
          "source_text": {"type": "string", "minLength": 1},
          "style": {"type": "string", "minLength": 1}
        },
        "required": ["source_text"]
      },
      "requirements": {
        "summary_model": {
          "kind": "model", "required": true,
          "input_keys": ["summary_provider", "summary_model"]
        }
      }
    }
  }
}
```

The run form supplies `source_text: "An example source."` and leaves style at its reviewed default. The local mapping selects an installed Ollama model; the server mapping selects that server's configured Ollama model. The following model names are illustrative selections, not installation or availability claims:

| Run input | Local destination | Server destination |
| --- | --- | --- |
| `style` | `three bullets` | `three bullets` |
| `source_text` | `An example source.` | `An example source.` |
| `summary_provider` | `ollama` | `ollama` |
| `summary_model` | `llama3.2:3b` | `qwen3:8b` |

The client passes the complete selected column as the local run service's inputs or the saved server definition's run request `inputs`; no key is obtained from the server's interpretation of `metadata.tldw_workflow`. Each new logical launch adds its own idempotency key. `prepare.text` becomes the next step's prompt; `summarize.text` is the documented result. The actual model is disclosed for each run, and different models do not promise identical wording. Endpoints, keys, run inputs, and generated results stay outside the shared definition. File sharing forks the fixture's identity through the normal copy flow; linking the same workflow for exchange does not. The wider first milestone adds file ingestion, human review, and note persistence with their own operation-specific conformance fixtures.

### Whole-definition safety and export review

V1 sharing is file-based: Review export, Export definition, and Import. Review covers the complete document, including inputs/defaults, prompt constants, metadata, and workflow-level hooks. It lists included content and portable requirements. Known credentials, connection tokens, active grants, run history, generated results, and device-specific paths must not enter the ordinary portable export. Discovered sensitive literals require explicit removal or conversion to a binding; removal is a reviewed draft/copy transformation, not mutation of the preserved import. Do not claim that a scan certifies arbitrary opaque strings secret-free. Unknown blobs remain privately preserved; shared export blocks until they are inspected or explicitly omitted from a new copy. No unreviewed automatic export or upload follows import.

Admission inventories workflow-level effects as well as step types. The current server's `on_completion_webhook` can include outputs by default, including when expressed as a URL string. Requirements shows the actual destination, triggering outcomes, and output inclusion before remote-run approval; existing server egress checks still apply. Local v1 does not implement completion callbacks: any non-empty hook blocks local execution until the user selects a supported remote target or explicitly removes the hook in a new revision. Keeping it in JSON never means it was executed locally. No provider response or imported metadata can grant network, tool, path, or publishing authority.

Execution and sharing approvals bind to the relevant document revision, destination and effect set. A change invalidates the affected approval. For adapters that discover effect targets at runtime, request review at that boundary with the resolved target and scope. Unknown executable fields are never silently stripped by a publish serializer or ignored during local execution. Where a destination cannot round-trip a field, explain the incompatibility and keep the original document intact.

Existing definitions can contain direct resource IDs or paths. Such imports remain readable, but are labeled Needs mapping; conversion to logical inputs/references is an explicit draft change. File sharing does not grant tool permission or publish a public link. Public galleries and collaborative editing are separate server capabilities beyond this proposal's initial file-sharing flow.

## 7. Server exchange and synchronization

Keep distinct actions and claims:

- Import/Export transfers a file.
- Fetch from server / Publish version uses existing definition APIs and records an explicit server binding.
- Sync uses a negotiated workflow-domain protocol and detects concurrent edits.

Fetching or publishing does not justify a Synced badge. The current server can support explicit exchange, but automatic workflow sync requires coordinated server and client changes. It is a separate v1 workstream, not deferred to v2 branching. Completion of the sync goal requires that counterpart work; Chatbook alone cannot deliver it.

Propose a new advertised `workflow.definition` domain using the current Sync v2 protocol vocabulary and its supported encryption policy. Do not send workflow payloads through a notes or chat domain. The exact new domain and metadata namespace remain proposed until the paired server contract is reviewed.

The synchronized object is the saved definition head and its immutable revision payload. Draft buffers, active executions, credentials, local bindings, output files, and UI preferences do not sync in v1. The client retains the last acknowledged object revision/hash/cursor; push includes its base revision. The server must reject a stale base and persist/recover definition-version materialization before acknowledging the change. Existing workflow name/version insertion is insufficient for this operation.

Conflict behavior is intentionally explicit:

1. Remote changed; local has no unpublished change: pull the remote saved revision.
2. Local changed; remote base is unchanged: publish the local revision.
3. Both changed: retain both and present Compare, Use local, Use remote, or Keep both.
4. A resolution becomes a new revision with the relevant parent references. Use remote preserves the displaced local work as a recoverable draft/copy; Use local still requires a current server revision check.
5. A local deletion moves the workflow to local Trash. Propagating deletion is separately previewed; a remote deletion concurrent with local edits is a conflict, never an automatic discard.

V1 does not auto-merge step order, prompts, or config maps. Revision history is never confused with an active run's snapshot. A failed network request preserves the draft, revision, pending operation, and last acknowledged base for retry.

Existing servers that do not advertise the new domain remain usable for explicit exchange and remote execution. The UI states Workflow sync unavailable on this server and retains the user's work. A web editor that cannot preserve the portable metadata is not yet a safe participant in bidirectional sync.

## 8. Verification and delivery boundaries

Design verification in this task is source-backed catalog coverage, path/link checks, whitespace checks, and consistency review. No source inspection here counts as a passing execution test.

Implementation will need targeted evidence for:

- Canonical definition round trips, including unknown config, stable IDs, metadata, typed expressions, and unsupported v2/v3 imports.
- Default/run-input precedence, binding-owned key collision rejection, typed references, unchanged two-destination definition JSON, and target/binding isolation across app profile changes.
- Whole-document effect inventory, completion-hook local refusal and remote disclosure, opaque export review, secret-widget exclusion, and unsupported publish-field preservation.
- Adapter conformance against pinned server fixtures for accepted config, deterministic outputs, errors, and side-effect behavior. LLM parity compares invocation/result contracts, not nondeterministic wording.
- Real local execution with tldw_server unavailable: local file -> prompt -> local model -> human review -> saved note, using installed dependencies and no external network. Additional audio/RAG fixtures cover optional adapters.
- Pause, cancel, timeout, retry, restart, pending approval, uncertain writes, and draft isolation using real SQLite and controlled adapters.
- Duplicate launch delivery, still-running timed-out workers, late completions/decisions, missing credentials on resume, response deadlines across restart, and persistent budget/accounting boundaries.
- Invalid-JSON draft recovery, failed flush navigation, discard confirmation, revision switching, and old-run/new-draft labels without changing the active run's identity.
- Both directions of server exchange, and later real negotiated sync: concurrent edits, duplicate names, deletion/edit conflicts, offline changes, repeated delivery, and partial projection recovery.
- Actual Textual screenshots and keyboard walkthroughs at wide, ordinary, and 60x20 terminals, including long forms, collapsed sections, validation jumps, and narrow results navigation.
- Focus restoration through resize, reference pickers, historical run inspection, and drawer closure; passive validation must preserve typing focus and cursor.
- Sharing to a clean installation with new resource IDs, no inherited credentials, and successful explicit requirements mapping.

Full test sweeps require user opt-in. The next implementation plan is scoped to the first end-to-end milestone, not all 130 types or the entire cross-repository program. Foundations and their targeted tests are included in that slice rather than declared complete as disconnected infrastructure.

### Staged v1 delivery

| Milestone | User-visible exit evidence | Boundary |
| --- | --- | --- |
| First local end-to-end slice | Author/save/recover a workflow in the three-pane UI; local file -> ingestion -> prompt -> local model -> human review -> saved note succeeds with tldw_server and external internet unavailable. Confirm permissions, result provenance, cancellation, and restart recovery. | Implement only the operation subsets needed for this example, plus the shared document/run/safety foundations. Not the full v1 release or complete adapter parity. |
| Complete the 21-step baseline | Each retained minimum type has declared field/operation coverage and conformance evidence, with installed-dependency checks and usable forms/reference selection. Optional packages need not be installed on every machine. | Preserve all 21 names as the v1 target. Other sequential candidates may join after the same admission tests; do not infer full operation coverage from a name. |
| File and existing-server exchange | Reviewed export/import and fetch/publish round-trip the same canonical definition; a clean destination can map requirements and run it without inherited secrets/grants. | No Synced badge based only on exchange. Unsupported metadata-losing editors remain outside safe bidirectional participation. |
| Paired-server workflow synchronization | Negotiated workflow-domain sync passes concurrent-edit, deletion, duplicate-delivery, offline, and partial-projection recovery tests against a real supporting server. | Explicit v1 workstream and completion gate for the sync goal; neither reassigned to v2 nor claimed complete by Chatbook-only changes. |

These milestones are delivery boundaries, not newly created Backlog tasks or future task-ID dependencies. Once the document contract is stable, counterpart sync work may proceed independently of extending local action adapters. Branching remains v2; map/parallel workflow orchestration remains v3. The first slice does not reduce the overall v1 target.

ADR required: yes. ADR path: `backlog/decisions/138-portable-workflow-definitions-and-local-execution.md`. Reason: new storage, runtime ownership, portable identity, cross-module interfaces, and server sync contract.

## 9. Review checkpoints

The user-approved layout, form style, local execution requirement, v1/v2/v3 roadmap, retained 21-step target, and review-remediation direction are fixed inputs. The user approved this revision's binding/input contract, whole-definition safety, attempt ownership, durable drafts/results, keyboard transitions, staged delivery, and concrete local safety defaults on 2026-09-08. Approval permits implementation planning; it is not verified runtime support.

The [first local milestone implementation plan](../plans/2026-09-08-workflows-local-file-to-note.md) covers the file-to-note slice only. Later v1 adapter, exchange, and paired-server sync milestones remain explicit program boundaries.

No application code, server code, or existing runtime behavior is changed by this draft.
