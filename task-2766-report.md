# task-2766 — Decomposing `_open_console_prompts_modal`

Branch `refactor/prompts-modal-2766`, based on `origin/dev` (`a453e7ded`).

## The decomposition, and why

The method was not one thing that had grown too long. It was **three** things
sharing a scope, and the split follows what the closures actually captured —
not what they were named.

### 1. `_ConsolePromptSource` — the data adapter (5 closures)

`capabilities`, `list_page`, `search`, `detail`, `save` captured **exactly one
thing**: `app_instance.prompt_scope_service`. Each did the same three steps —
look a method up, refuse in user-visible copy if the source cannot serve it,
forward the call with this cluster's own contract. That is the textbook shape
of an adapter, so it is one: a class with one field, whose bound methods the
modal receives.

It earns its keep beyond line count. The cluster's source contract — browse
pages of 10, searches capped at 25, `save`'s `source` routed into the
service's `mode` — is now stated in one place instead of being spread across
five sibling closures where a drift between them would be invisible. The
five duplicated `getattr`/`callable`/`raise` triples collapse into one
`_require` helper that carries the per-callable refusal copy.

`record_prompt_usage` moved here too, even though its closure lived in the
improvement group: it reads the same service, through the same duck-typed
lookup, and the improvement flow now reaches it the same way it reaches
`detail` for its freshness probes.

### 2. `_ConsolePromptImprovementFlow` — the stateful sub-flow (11 closures)

These eleven were a single conversation with **shared mutable state**: pin and
disclose a provider target, build a request snapshot against it, validate what
comes back against the artifact it came from, apply it, retry a failed
System-prompt persist. The `nonlocal pinned_improvement_resolution` is the
tell — one closure wrote it, three read it, and nothing else in the 397-line
scope touched it. That is an object; the `nonlocal` becomes
`self._pinned_resolution` and the eight values the closures each re-captured
become named constructor arguments.

Its app-level needs stay **callables** (`active_session_settings`,
`build_provider_selection`, `sync_system_prompt_surfaces`), and `app_instance`
is held rather than snapshotted into a bound `notify` — the same binding rule
the controller follows, for the same reason: the suite replaces those targets
on the screen instance after the modal is already open.

One dedupe inside it: the session-changed / System-prompt-changed guard pair
appeared **three times verbatim**. It is now `_stale_reason()`, returning the
copy or `""`; the two raising call sites raise it, the applying one returns a
stale outcome with it. Short-circuit order is preserved, so the live
fingerprint is still only computed when the session check passes.

### 3. Promoted out of closure form

- `restore_focus` → `ConsolePromptsController._restore_console_composer_focus`.
  It captured only `self`; there was never a reason for it to be a closure.
- `_resolution_identity` → module-level `_provider_resolution_identity`.
  A pure function of its argument, capturing nothing at all.

### What I deliberately did **not** do

- **No new module.** The two collaborators are private to the prompt cluster
  and are only meaningful next to the controller that builds them. A
  `prompts_modal.py` would have split one story across two files and put the
  module's "Zero DOM" AST test on the wrong side of a boundary.
- **No extraction of the opening preamble.** The 30 lines before the
  collaborators are read-once fact-gathering with two early returns; hoisting
  them into a `_gather_open_state()` would have produced a dataclass whose
  only consumer is twelve lines below its construction.
- **No bundling of the five provider dependencies.** `model_display`,
  `build_selection`, `gateway`, `blocker_copy` and `provider_recovery` look
  like a group by name, but they are five distinct services with five distinct
  lifetimes (one is a bare-attribute read handed on uncalled). Collapsing them
  would have bought a smaller number at the price of a value type the screen
  must now construct — exactly the manufactured win the brief warned against.

## Per-closure verdict

| Closure | Lines | Verdict |
|---|---|---|
| `apply_improvement_result` | 74 | → `_ConsolePromptImprovementFlow.apply_improvement_result` |
| `activate_improvement_context` | 58 | → flow method (shares the pin) |
| `build_improvement_snapshot` | 56 | → flow method (shares the pin) |
| `validate_saved_recipe` | 27 | → `flow._validate_saved_recipe` (internal; only `apply` calls it) |
| `retry_improvement_persistence` | 21 | → flow method |
| `validate_saved_prompt` | 19 | → `flow._validate_saved_prompt` (internal) |
| `record_applied_usage` | 18 | → `flow._record_applied_usage`; its service lookup → `_ConsolePromptSource.record_usage` |
| `capture_manual_resolution` | 9 | → flow method (writes the pin) |
| `_resolution_identity` | 8 | → module-level `_provider_resolution_identity` (pure) |
| `save` | 6 | → `_ConsolePromptSource.save` |
| `capabilities` | 5 | → `_ConsolePromptSource.capabilities` |
| `list_page` | 5 | → `_ConsolePromptSource.list_page` |
| `search` | 5 | → `_ConsolePromptSource.search` |
| `detail` | 5 | → `_ConsolePromptSource.detail` (also the guards' freshness probe) |
| `_active_system_fingerprint` | 3 | → `flow._active_system_fingerprint`, folded into `_stale_reason()` |
| `validate_improvement` | 3 | → flow method. **Moved on membership, not size**: it reads the opening snapshot, which is flow state. Left behind it would have needed that snapshot passed in — more parameters than the body has lines. |
| `restore_focus` | 2 | → `ConsolePromptsController._restore_console_composer_focus` |

Nothing was left inline. The two shortest closures moved for opposite reasons
and both are stated above.

## Numbers

| | before | after |
|---|---|---|
| `_open_console_prompts_modal` | 397 lines | **82** |
| nested closures in it | 17 | **0** |
| `ConsolePromptsController` class | 1093 lines | 788 |
| `prompts.py` module | 1251 lines | 1474 |
| `prompts.py`, code-only (no docstrings/comments/blanks) | 804 | 839 |
| constructor named dependencies | 18 | **16** |

The module grew 223 lines; **188 of those are docstrings** for the two new
classes and their 20 methods, per this repo's Google-style requirement. The
real code cost is +35 lines — two `class` statements, thirteen field
assignments and twenty signatures, against 17 closure headers removed. I
consider that the honest price of the trade and flag it rather than hide it:
the win is the 397→82 method and the 17→0 closure count, not a smaller file.

## Constructor dependencies: 18 → 16

The three post-apply re-sync bridges (`_sync_console_chat_core_state`,
`_sync_console_rail_system_line`, `_sync_console_settings_summary`) were
called **only** as an ordered trio, at exactly the two moments the store
accepted a new System prompt, and never individually anywhere in the cluster.
They are now one dependency, `sync_console_system_prompt_surfaces`.

This is a cohesion win rather than an arithmetic one: the flow's actual need
is "the System prompt landed — refresh the surfaces that display it", and that
is now what it asks for. `wiring.py` supplies it as a nested function that
resolves each screen method **by name at call time**, so the late binding that
eleven of the sixteen dependencies rely on is untouched (verified: the wiring
suite's late-binding tests pass, and `test_ui_responsiveness.py`, which
replaces `_sync_console_chat_core_state` on the screen instance, passes).

No other reduction was available without manufacturing one — see "What I
deliberately did not do" above. The remaining 16 are what the cluster
genuinely reaches for.

## Behaviour evidence

**AST equivalence against `HEAD~1`.** I extracted every moved body from both
revisions, applied the rename map, and diffed. Byte-identical:
`capture_manual_resolution`, `validate_improvement`, `validate_saved_recipe`,
`validate_saved_prompt`, `restore_focus`. Every other difference is one of the
four documented transformations and nothing else: `_stale_reason()` (3 sites),
`_require()` (5 sites), the sync-trio collapse (2 sites), and `self.` field
prefixes.

Call order is preserved end to end, including the two separate
`_console_provider_blocker_copy()` reads (one into the disclosure context, one
into the modal kwargs) and the modal's kwarg evaluation order. The controller
still contains zero `query_one`/`query` (the module's own AST test asserts it).

**One deliberate non-identity, disclosed.** `record_prompt_usage`'s
availability lookup now happens inside `_ConsolePromptSource.record_usage`,
which the caller invokes inside its `try`. Previously the lookup sat outside
it. This differs only if attribute *access* on the scope service raises — no
such service exists in the codebase or the suite — and the new behaviour is
strictly safer if one ever did: the user gets the "usage could not be
recorded" warning instead of an exception escaping into an apply that has
already landed (the old `apply_improvement_result` did not wrap that call).

## Test evidence

Characterisation was written, mutation-checked and pushed **green before**
any production change (`d635a862a`): 7 new tests in
`Tests/UI/test_console_prompts_controller.py` covering the exact prompt-source
contract each data callable forwards, the per-callable refusal copy, the
dismissal focus restore, the pinned-target lifecycle shared between
`activate_improvement_context` and `capture_manual_resolution`, the
moved-System-prompt activation guard, the unpinned-target snapshot refusal,
and `validate_improvement`'s captured-vs-opening snapshot preference.

Two mutations confirm they bite: `per_page` 10→11 and dropping the
capture-once guard each failed exactly one of the new tests, and only that one.

| Suite | Baseline | After |
|---|---|---|
| `test_console_prompts_controller.py` + `test_console_controller_wiring.py` | 21 + 16 = 37 (+7 new) | **44 passed** |
| `test_console_native_chat_flow.py` (13 real modal-open drives, the whole improvement sub-flow) | 304 passed | **304 passed** |
| prompt-cluster batch (workbench contract, composer menu, prompts modal, system prompt, command composer, composer history, prompt picker, system prompt chip) | — | **268 passed** |
| `test_ui_responsiveness.py`, `test_console_agent_swap.py`, `test_console_internals_decomposition.py` | — | **195 passed** |
| `Tests/Architecture/` + `test_application_state_ownership.py` + `test_console_model_section.py` | — | **89 passed, 1 failed** |
| `Tests/UI` collect-only sweep | — | **9073 collected, no errors** |
| `ruff check` on all three changed files | — | clean |

The single failure is `Tests/Architecture/test_persistent_diagnostic_inventory.py`
— the pre-existing task-2768 failure. I ran the underlying checker directly:
its report names neither `prompts.py` nor `wiring.py`.

## Concerns

- **Module length.** `prompts.py` is now 1474 lines. That is a docstring-heavy
  1474, and the largest method in it is 82 lines, but if the programme wants
  the file itself smaller, `_ConsolePromptSource` and
  `_ConsolePromptImprovementFlow` are the natural seam for a later move — they
  have no dependency on the controller at all, only on values it passes them.
- **`_ConsolePromptImprovementFlow` is 402 lines** (about 210 code-only). It is
  a genuine unit — every method touches the pin or the opening disclosure — but
  it is the next-largest thing in the cluster, and a future reviewer should
  measure it before assuming it is fine.
- The improvement flow's constructor takes 13 arguments. That is the per-open
  state the closures shared, made visible rather than added; I did not try to
  reduce it, because bundling it would hide exactly what this task set out to
  expose.
