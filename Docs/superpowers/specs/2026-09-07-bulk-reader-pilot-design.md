# Bulk-reader named-agent pilot

User approval: the architecture comparison recommended a read-only bulk-reader
preset tested on repository and long-transcript questions; the user said to do it.
Backlog: TASK-32026.

## Scope

Add a single editable preset to the existing Settings > Agents form. Choosing
Bulk reader creates an unsaved new form; Save uses the existing database CRUD.
It never installs itself on startup or overwrites a selected definition. Its
model field starts empty and explicitly explains that savings require choosing
a cheaper model available at the parent's existing provider endpoint. The
configured model must reach the actual Console provider boundary, not merely
the AgentService callback.

The preset permits only fs_list, fs_read, fs_glob and fs_grep. Existing runtime
allowlist intersection and workspace permission checks remain authoritative.
The instructions ask for the question and explicit relative paths, compact
findings with exact quotations and file/line references, contradictions and
missing evidence. Source text is data, never new instructions. Findings are
leads for verification; the preset does not validate quotations or enforce the
requested file subset independently of the existing workspace boundary.

Use an opt-in comparison script with nonsensitive, pinned synthetic repository
and transcript cases. Each arm uses AgentService and the real local read tools.
The direct arm cannot spawn; the delegated arm requests the named worker, and
the report checks that delegation actually occurred. Both receive the same
question and paths, never preloaded source bodies. The main agent may reread
exact sections after delegation. A fresh scratch database and confined corpus
workspace isolate each case/arm from personal conversations and files.

Record every provider call, model, usage availability, disjoint cache buckets,
latency, run status, tool reads, answers, worker output, corpus hashes and run
settings. Price calls separately by model with the existing pricing catalog;
unknown or incomplete usage/pricing makes total cost unknown. Never compute a
bill from RunOutcome.total_tokens. Retain failed-call metadata and do not call
an unsuccessful or non-delegating run a successful comparison. Quality remains
pending manual review against expected facts and a fixed rubric; offline tests
prove plumbing and accounting only. Live calls require explicit CLI opt-in and
the user's chosen same-provider model pair. No model pair is currently chosen.

The first live runner supports Moonshot and ZAI only. Their existing gateway
contract forwards the request timeout and retry controls needed for bounded,
per-request accounting. Other current adapters do not share that control seam;
the llama.cpp streaming path can also issue an unrecorded fallback request.
Refuse unsupported providers before application imports or network calls.
This evaluation limit does not restrict the preset's existing same-provider
model support. Adding other live evaluation providers requires verifying their
transport and request-accounting path first.

No automatic routing, new runtime framework, cross-provider routing, generated
file writer, Library integration, storage migration or dependency is introduced.
The broader Proposed ADR-133 Library reader is a separate experiment.

ADR required: no
ADR path: N/A
Reason: this reuses the existing named-agent, permission and provider contracts;
the isolated evaluation is developer tooling and changes no runtime boundary.

## Acceptance and validation

- Selecting the preset performs no DB write; saving creates the configured
  definition and duplicate names never overwrite another definition.
- A real AgentService spawn applies the chosen model and restricts tools even
  when the parent can write; a malicious worker write cannot change a fixture.
- Both baseline and delegated evaluation arms execute, reports include worker
  spend and failures, and missing accounting remains unknown.
- Targeted pytest runs, changed-file lint/format checks, and a production-CSS
  Settings render verify the implementation. Full-suite runs are not authorized.
- Run and inspect the live comparison once the user supplies a model pair;
  report its limits and pending quality review honestly.
