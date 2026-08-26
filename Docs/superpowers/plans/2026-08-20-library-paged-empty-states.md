# Library paged empty states implementation plan

> TASK-19023 implements Wave 1's paged browse empty-state grammar from the
> approved Library lifecycle and progressive-disclosure design.

## Goal

Replace fresh zero-result pager mechanics with a concise, truthful recovery
path in Media, Conversations, and Prompts, while preserving filter scope and
all loading, error, stale, mutation, and retained-page authority.

## UX contract

```text
fresh + total 0 + no filter
    -> source-empty copy + one creation/recovery action

fresh + total 0 + active filter
    -> no-match copy + preserve visible filter + one reset action

uninitialized / loading / initial error
    -> existing unknown-total status + Retry authority

retained stale / refresh failure
    -> retained read-only rows + existing Retry/pager authority

fresh + total > 0
    -> existing toolbar + rows + exact pager
```

Source actions:

- Media: `Import media` opens the existing Library Import destination;
  filtered type zero offers `Show all types`.
- Conversations: `Start in Console` uses the existing Console live-work route;
  query zero offers `Clear filter`.
- Prompts: `New prompt` opens the existing Prompt create editor and the existing
  `Import…` action remains its approved secondary path; query or collection
  zero offers `Clear filter` or `All prompts`, respectively.

## Ownership and architecture

ADR required: no

ADR path: N/A. Existing ADR-067 owns exact totals/pagers and source-specific
controllers. Existing ADR-076 owns the lifecycle composition grammar. This task
does not change persistence, storage, service envelopes, or source authority.

The canvases derive presentation only from their already-authoritative state:

- Media: `pager.title_count`, `canvas.active_type`, and current rows;
- Conversations: `canvas.pager.title_count`, `canvas.query`, and current rows;
- Prompts: `pager.title_count`, `browse_result.status/scope`, and current rows.

Each canvas keeps its own IDs and copy. LibraryScreen binds the new buttons to
existing route/request seams. No shared empty widget, new controller, new
router, or generic action model is introduced.

## TDD sequence

### 1. Canvas presentation RED

Add focused mounted canvas tests proving that a fresh exact unfiltered zero:

- omits Previous/Next, zero range/page copy, and disabled Select explanations;
- renders the approved enabled source recovery action set (one action for Media
  and Conversations; New prompt plus Import for Prompts);
- keeps the title's authoritative `(0)` count;
- does not affect unknown/loading/error or stale retained rendering.

Add filtered-zero tests proving the active filter remains visible and only the
matching reset action is offered. Include Prompt query and collection variants.

### 2. Minimal canvas GREEN

Add small source-local predicates inside each list composer. Return from the
fresh-empty branch before normal toolbars/list/pager composition. Keep initial
error and stale branches on their current code paths.

### 3. Screen action RED/GREEN

Add mounted LibraryScreen tests that press the actual recovery controls:

- Media enters the production Import canvas and focuses its path input;
- Conversations calls the production Console live-work seam once;
- Prompts enters the production blank create editor;
- type/query/collection reset requests page 1 with the rest of the applied
  source scope preserved.

Use the existing admission/generation/focus guards. Do not route through label
parsing or duplicate destination state.

### 4. Geometry and keyboard proof

At 100x30 and 170x48 with exact `TldwCli.CSS_PATH`, settle each source to a
fresh exact empty response and assert:

- copy and recovery action are painted inside the active canvas;
- no zero pager controls or inactive selection copy are mounted;
- the recovery action is enabled and reachable in the semantic Tab order;
- filtered reset remains reachable and does not silently mutate before press.

No CSS change is expected. If mounted proof exposes clipping, change only the
smallest source CSS block and regenerate/check the bundle.

## Focused verification

Run only touched/direct owners:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_library_prompts_canvas.py \
  Tests/UI/test_library_multiselect_media.py \
  Tests/UI/test_library_multiselect_conversations.py \
  Tests/UI/test_library_shell.py \
  -k 'library and (media or conversation or prompt) and (empty or zero or no_match or recovery or filter or pager or focus or geometry)'
```

Then run Ruff on the final changed Python inventory and `git diff --check`.
Run CSS build/parity only if CSS changes. Repository-wide pytest is explicitly
outside this task's evidence boundary.

## Required inverse checks

Apply one mutation at a time and immediately restore it:

1. Re-enable pager composition for a fresh exact zero; the no-zero-pager test
   must fail.
2. Treat filtered zero as source empty; the preserved-filter/reset test must
   fail.
3. Hide Retry for an initial or retained error; the recovery-authority test
   must fail.
4. Bypass the existing source route/request seam; the actual-action test must
   fail.

## Closeout

Update the Library user guide with ASCII-only examples of source-empty versus
filtered-empty recovery. Check all TASK-19023 ACs, add concise Implementation
Notes with exact bounded evidence/inverses, perform self/spec/quality review,
and mark Done through Backlog CLI only after every gate is green.
