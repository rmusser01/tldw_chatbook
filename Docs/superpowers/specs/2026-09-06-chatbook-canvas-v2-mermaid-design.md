# Canvas V2: offline Mermaid subset and immutable runtime profiles

Date: 2026-09-06
Status: Written design awaiting user review; implementation has not started.
Owner: [TASK-31933](../../../backlog/tasks/task-31933%20-%20Design-Canvas-V2-Mermaid-subset-and-pinned-runtime-profiles.md)

ADR required: yes

ADR path: [ADR-124](../../../backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md)

Reason: Extend [ADR-121](../../../backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md) with a library execution boundary, exact-profile compatibility and security refusal policy.

## Goal and evidence

Let users open small Mermaid diagrams in Canvas alongside ordinary HTML and
interactive JavaScript, without changing Canvas's strict zero-egress runtime.
This is a **Mermaid syntax subset with a Canvas-specific renderer**, not the
Mermaid browser API or a promise of identical upstream layout.

The [disposable compatibility spike](../../Canvas/V2_MERMAID_SPIKE.md) demonstrated
only a two-node linear flowchart and a two-participant message sequence. Full
and Tiny Mermaid bundles exceeded the current script ceiling and needed native
browser facilities. A grammar-only prototype fit the existing virtual runtime.
Branching, rejoining, notes, long labels and Unicode remain qualification work,
not capabilities established by that spike. No probe code is approved for reuse
as production code merely because it worked.

## User-facing contract

- Native terminal users open the existing system-browser Canvas shell. Served
  users keep the same-origin Canvas surface; no new port or login flow appears.
- Multiple named Canvases, stable IDs, revisioned titles, transcript cards,
  active-branch resolution and explicit historical selection remain unchanged.
- Creation auto-opens according to the existing preference. Accepted updates
  reload the current preview and preserve history with Updated / Undo / View
  previous affordances. A failed preview is clearly marked, not silently replaced
  with the old image under the new revision's identity.
- Temporary Canvases retain their badge and session-incarnation ownership. Their
  complete staged history joins the chat promotion transaction atomically; an
  unsaved session's end destroys it.
- Preview-first continues: no built-in editor. Existing source inspection, copy,
  source download and confirmed unsent repair drafts remain available.
- Model mutations remain reversible, complete-document operations without tool
  approval. Existing list/read tools and the signatures below do not change:

  ```text
  canvas_create(title, html)
  canvas_update(canvas_id, expected_parent_revision_id, html)
  ```

Add **Open in Canvas** to `mermaid` code fences through the existing code-block
import ownership and replay checks. It wraps the exact text, escaped for an HTML
text node, in a complete document. It does not interpolate source into JavaScript.

### Assistant authoring guidance

Keep tool parameters unchanged, but supply a bounded, Chatbook-owned authoring
guide with the Canvas tool descriptions/context. It names the selected profile
and execution availability, the declarative wrapper, accepted syntax, major
exclusions, shared budgets and the difference between source acceptance and
preview success. Include one complete small flow example and one sequence example
with explicit participants. Do not advertise upstream Mermaid's browser API.

Use the same profile resolver as execution to select guidance: creation describes
the installed admitted entry; reading a selected revision describes that exact
profile, including source-only refusal. Guidance for updates must not replace an
older profile's contract with the current creation default. Limit each guide to
8 KiB UTF-8, supplied once per effective profile in the active model context,
with no source/labels copied into general diagnostics or transcript cards.

Unsupported-feature and limit errors have allowlisted, source-free repair hints
(for example, declare participants explicitly or split a large diagram). Browser
errors still reach the assistant only through the confirmed unsent repair draft;
guidance does not add an automatic feedback loop, retry or submission. Test the
provider-visible guidance as well as its bounded/source-free UI/log projections.

## Declarative authoring and startup lifecycle

```html
<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Decision flow</title></head>
<body>
  <h1>Decision flow</h1>
  <pre data-canvas-diagram="mermaid">flowchart TD
    A[Start] --> B{Ready?}
    B -->|Yes| C[Continue]
    B -->|No| D[Revise]
  </pre>
</body>
</html>
```

Only text-only `pre[data-canvas-diagram="mermaid"]` declarations are admitted.
Nested element content, empty declarations and unknown diagram kinds fail
structural validation for a new V2 document. Decode HTML entities exactly once
using the compiler's normal text-node rules. Preserve the original complete HTML
as revision truth; retain the decoded diagram text separately in the derived plan.

The compiler selects declaration targets in document order. Startup parses every
declaration, explicitly validates supported semantics, and builds all diagram
output detached before applying it through the existing typed-patch boundary.
Only after successful diagram preparation does authored JavaScript run. Both
phases share the existing startup operation and its limits. Failure anywhere
discards that startup transaction; no partly drawn diagrams are published.

Render declarations **once per load, before authored JavaScript**. There is no
mutation observer, diagram-update API or automatic rerender when a script edits
text or adds a declaration. A new revision/reload reruns startup. Ordinary V1
scripts may subsequently change the rendered DOM through the existing facade;
this grants no additional capabilities or diagram parsing entry point.

The source remains inspectable even after its preview is replaced with SVG. A
scripts-disabled or profile-unavailable opening is inert source, never a native
Mermaid fallback. Source HTML downloads containing declarations require a
compatible Chatbook profile to render them; they are not standalone Mermaid apps.
Existing warnings about running authored HTML outside Chatbook still apply.
Standalone diagram/SVG export is not part of this release.

## Supported syntax and layout

The first catalog entry pins the Mermaid 11.17.2 grammar input investigated by
the spike, subject to dependency/security qualification before shipment. A
different upstream input requires a reviewed profile, not an invisible upgrade.

| Family | Initial accepted forms | Explicit exclusions |
| --- | --- | --- |
| Flowchart | `flowchart TD`, `TB` (TD alias), or `LR`; named nodes; rectangles `A[Text]`, rounded nodes `A(Text)`, diamonds `A{Text}`; directed `-->` arrows with optional `-->\|Text\|` labels; acyclic branches and rejoining | Cycles/self-loops, subgraphs, other directions/shapes/edge styles, compound node/edge shorthand, edge IDs, styling, click/link actions |
| Sequence | `sequenceDiagram`; explicit `participant A` or `participant A as Label`; solid `A->>B: Text` and dashed `A-->>B: Text` messages; `Note left of A: Text`, `Note right of A: Text`, `Note over A: Text`, or `Note over A,B: Text` | Implicit participants, self-messages, actors, activation, loops, alternatives, parallel/critical regions, numbering, create/destroy, links and callbacks |
| Labels | Plain text, including Unicode and quoted flowchart labels; renderer-owned wrapping | HTML/Markdown labels, embedded formatting, HTML line-break tags, URLs with action semantics |

Blank lines and ordinary `%%` comments are allowed. Front matter, initialization
directives such as `%%{init: ...}%%`, themes, custom CSS/class definitions, parser
configuration and all syntax not listed above are refused. Ordinary document CSS
remains under V1 rules; it cannot configure the parser or authorize a resource.
Identifiers use `[A-Za-z_][A-Za-z0-9_]*`; Unicode is supported in labels, not IDs.
Flow node declarations may be referenced repeatedly but cannot change an already
declared shape/label. Duplicate identical declarations are harmless; conflicting
ones, unknown sequence participants and duplicate participant IDs are errors.

An upstream parse success is not admission. A closed semantic adapter must
account for every recognized operation and reject unsupported syntax before
layout. Do not use missing callbacks, accidental exceptions or ignored events as
validation. Refusals identify a diagram ordinal, stable error code and bounded
line/column when available; never silently remove content to make a graph fit.

Flow layout uses deterministic DAG ranks, source-order ties, bounded ordering
passes and orthogonal edge routing. Sequence layout uses declared participant
order and message/note order. Notes reserve space rather than overlay messages.
Logical text metrics and wrapping are owned by the profile and do not use native
`getBBox`/`getBoundingClientRect`, network fonts or browser DOM access. Preserve
label content; if it cannot fit safely within geometry/work limits, report a
limit failure instead of truncating it. Unicode wrapping must not split surrogate
pairs or combining/emoji sequences. Include RTL and unsupported-glyph behavior
in qualification; font availability can affect appearance.

Promise deterministic semantic output and SVG geometry for exact source/profile
under its declared text metrics, **not pixel-identical glyphs across platforms**
or upstream Mermaid aesthetics. Generated SVG uses only the existing V1 allowlist;
labels are text nodes, never markup sinks. Accessible diagram descriptions/source
use supported ordinary HTML rather than expanding active SVG or link behavior.

Default diagram typography is explicit on generated text: `monospace`, 16 CSS px,
normal weight/style and 24 px line spacing. Profile-owned logical width tables
and grapheme segmentation determine wrapping, not ambient browser/OS `Intl` or
font measurements. Pin the exact Unicode-data version, segmentation/width rules
and data hashes in the profile before qualification. Preserve original code points
without normalizing source; unavailable glyphs use local font fallback without
changing computed geometry or fetching fonts. Exact glyph appearance is outside
the guarantee; source inspection retains text that a local font cannot display.

Render at intrinsic logical size inside a scrollable ordinary-HTML wrapper;
narrow viewports scroll rather than automatically shrinking labels to illegibility.
Do not clip or ellipsize labels to fit. Initial diagram geometry is capped at
2,048 px wide, 4,096 px high and 4,194,304 square px per diagram, with 8,388,608
square px aggregate. Bounds include labels, notes and arrowheads. Coordinates,
extents and path-control values must be finite and within the admitted viewBox;
nonpositive sizes, overflow and nonfinite intermediate layout results refuse
before emitting patches. Exhaustion returns a bounded geometry-limit error with
a suggestion to shorten labels or split the diagram.

These bounds and typography describe renderer-produced output, not a new CSS
sandbox. Use existing allowed inline styles for defaults and avoid inherited page
typography where explicit defaults suffice. Authored CSS (including overriding
cascade rules) or later JavaScript may restyle/resize diagrams through V1's
existing surface. The default-layout readability guarantee does not cover those
overrides; they do not recompute logical geometry or trigger rerendering. Test
ordinary inherited CSS and explicit overrides separately. Do not introduce native
measurement, shadow DOM or new style privileges to hide this distinction.

## Runtime and dependency boundary

One small packaged catalog initially contains only this Mermaid subset. React,
D3, general plugin registration and the full Mermaid API are deferred.

- Pin upstream tarball integrity, exact grammar files/build inputs, adapter and
  layout versions, full output hashes, dependency closure and license notices.
  Reproduce the bundle deterministically with an explicit member allowlist. Do
  not use the spike's comment-marker extraction as a production build contract.
- Follow ADR-121's vendoring discipline: verification before extraction, no
  package lifecycle scripts, no undeclared dependencies or opportunistic fetches.
  Installed Python packages include all required verified assets; end users need
  no Node/npm or internet access.
- The trusted host selects a fixed integrity-checked asset closure before the
  generated-execution acknowledgement. Include those assets in the browser
  harness's exact allowlist. No lazy fetching, dynamic imports, module loader,
  CDN resolution or generated URL may enter the runtime.
- Library parsing, semantic validation and layout execute only inside the
  existing QuickJS-WASM worker realm. Authored script cannot access the private
  parser, adapter, lifecycle handles or transport. The native wrapper only
  transports bounded source and validates typed output; it never evaluates
  generated/library JavaScript in the native realm.
- Preserve opaque-origin iframe isolation, CSP, current private bridge,
  authentication, browser-session scope, selection/load freshness checks,
  off-event-loop compilation admission and late-result fencing from V1.
- No network, filesystem, cookies, persistent page storage, parent DOM or Chatbook
  APIs. `canvas.submit` still requires confirmation and inserts only an unsent
  draft. Downloads retain their passive-format confirmation contract.

V2 has its own closed derived-plan schema for inert diagram records and target
node IDs. V1's existing six-field wire contract remains unchanged. Compiler,
prepared-plan admission, gateway, renderer and worker must agree on exact profile
and plan schema; no optional-field compatibility bypass. Derived plans are not
exported as artifact truth.

## Immutable profiles and one selection authority

Reuse the existing revision `runtime_profile` field and archive format 3.0.
No new dependency database or archive schema is required. A short immutable ID
such as `canvas-v2-mermaid-1` references a packaged immutable manifest containing
full digests and the grammar/adapter/layout/semantic-limit contract. It also pins
the exact QuickJS-WASM engine build, the virtual DOM/facade and render-plan
compatibility versions, and the Unicode segmentation/width rules and data.
The ID satisfies the existing 64-byte safe-identifier constraint. Never reassign
it to changed manifest bytes. A semantic change allocates a new profile ID.

Changing the pinned engine, facade contract, grammar, layout or Unicode inputs
requires a new profile even when motivated by a security fix. Do not silently
substitute a patched engine under an old immutable identity. A vulnerable profile
can be revoked and remain source-only; recovery uses the explicit new-Canvas path
below. Security-only changes to nonsemantic host admission, transport validation
or compiler rejection rules may preserve a profile if compatibility fixtures
confirm unchanged accepted semantics. Record their separate build/security-policy
identity for admission and derived-cache invalidation. No permissive engine
version range or untested compatibility substitution is introduced.

One Canvas-domain profile resolver owns selection and admission; native and
served transports do not infer it independently. Compiler output, prepared plan,
revision metadata and publication must all use the same resolved profile.

| Operation | Required profile behavior |
| --- | --- |
| New Canvas without declarations | Select `canvas-v1` |
| New Canvas with a valid Mermaid declaration | Select the initial installed, allowed V2 profile |
| Update V1 parent, first Mermaid declaration | Create a V2 child; leave the V1 parent immutable |
| Update V2 parent, including removing every declaration | Retain that exact V2 profile; no automatic downgrade or latest-version selection |
| Rename | Same source/profile as the selected parent; no implicit upgrade |
| Historical update or branch change | Derive only from the exact selected reachable parent; stale-parent rules remain |
| Temporary stage, turn commit and durable promotion | Preserve the already resolved profile for every revision; do not reselect at commit |
| Reopen, replay, imported revision or cache lookup | Use stored profile exactly; do not infer from current source or current defaults |

Unknown, missing, integrity-failed, retired or **security-revoked** profiles are
source-only. Security refusal takes precedence over execution reproducibility;
preserve source/identity/history and explain the reason without substituting a
newer or weaker engine. Known profile availability is not enough: current Canvas
execution policy and compiler security checks must also permit it. Apply the
effective policy on every execution admission.

V2 uses a verified, process-lifetime catalog/policy snapshot. Application/runtime
or packaged revocation-policy updates are **restart-required**: stop the native
Chatbook process, or the served parent and all its child processes, before
installing the update, then start them on the new snapshot. Restart invalidates
old browser loads, capabilities and pending bridge reservations; reconnecting
browsers receive fresh admission under the new policy. A browser refresh alone
does not update a running host's snapshot. Do not claim that overwriting package
files revokes an already-running engine.

Native and served execution must use a consistent build/catalog/policy identity.
Served parent/child identity mismatch fails Canvas admission closed, rather than
mixing old authority with new runtime bytes. Serve the verified asset bytes bound
to the snapshot; no mutable-file rereads can replace them mid-load. There is no
package watcher, hot policy reload or online revocation service in this release.
For immediate containment before an upgrade, the existing explicit Canvas-disable
latch still stops active execution and revokes its capabilities/bridge state;
that action does not require waiting for an application update. Source/history
remain stored under the existing lifecycle.

Read/export and metadata-only rename can preserve an inert revision; HTML updates
from an unavailable profile refuse rather than guess. Recovery is explicit: copy
the source into a **new Canvas** admitted under the current supported profile,
leaving the original history untouched. An in-place profile-upgrade tool is out of
scope. Old V1 revisions retain V1 semantics even if their text happens to contain
the newly meaningful attribute.

Compiler security validation may become stricter without rewriting the semantic
profile. Any derived cache must bind exact source digest, profile/manifest
identity and compiler/security-policy identity; a restarted revoked policy cannot
reuse executable cache entries from the prior snapshot. Archives contain source
and profile IDs, not executable catalog bytes or an installer. Import never
fetches a missing profile and must not treat
an archive-supplied manifest as execution authority.

## Budgets: shared, bounded and observable

Retain V1's 512 KiB source, 256 KiB evaluated-script bytes, 32 MiB guest heap,
512 KiB stack, 250 ms startup interrupt, 50 ms event interrupt, 100 pending-job
drain cap, 1,800 DOM nodes, 900 CSS rules and 500 patches per operation. Existing
timer/rate/bridge ceilings and the terminable-worker backstop remain unchanged.
Loading/evaluating the library and rendering diagrams count toward these limits,
including the script-byte allowance alongside authored scripts. Do not move work
outside the measured operation or reset clocks/counters per diagram. An engine
interrupt target is not a guaranteed wall-clock deadline; test worker termination
separately, including slow parsing.

Initial additional admission ceilings proposed for the first immutable profile:

| Resource | Per diagram | Whole document |
| --- | --- | --- |
| Declaration count | — | 4 |
| Decoded Mermaid UTF-8 input | 8 KiB | 16 KiB |
| Flow nodes / edges | 16 / 24 | 24 / 32 |
| Sequence participants / messages / notes | 6 / 16 / 8 | 8 / 24 / 12 |
| Individual label / all label UTF-8 bytes | 512 bytes / 4 KiB | 8 KiB all labels |
| Generated SVG elements / serialized UTF-8 output | 250 / 48 KiB | 400 / 64 KiB |
| Layout work units | 10,000 | 20,000 |
| Geometry width / height | 2,048 / 4,096 CSS px | Per-diagram bounds apply to each |
| Logical viewBox area | 4,194,304 square CSS px | 8,388,608 square CSS px |

A work unit is each examined node, edge, label segment or candidate routing cell;
each ordering sweep, collision scan and wrapping iteration charges its inspected
items. Cap each loop's input before allocation. Input bytes, semantic sizes and
bounded engine execution additionally constrain parser work. Serialized-output
limits must be enforced while constructing output, not after a large allocation.

These are refusal ceilings, **not a guarantee that every graph below each ceiling
fits the shared startup/patch budget**. Mixed HTML, scripts and diagrams share
the same remaining capacity; all ceilings apply conjunctively. The implementation
qualification must demonstrate the useful fixtures below without raising V1
limits. If it cannot, pause for a design revision rather than silently dropping
features, resetting quotas or enabling a larger runtime. Freeze profile ID and
manifest only after that qualification; this document does not publish a bundle.

## Save state is not preview state

Host structural/source/profile validation precedes staging; invalid host input
creates no revision. Full Mermaid parsing/layout occurs in browser QuickJS and
can fail **after** source has been staged or committed. Do not add an implicit
host-side JavaScript evaluator to promise precommit rendering validation.

Keep two independent outcomes:

- **Artifact:** staged for its originating turn, committed, or discarded by the
  existing cancellation/failure transaction. A tool success means source was
  accepted into that lifecycle, not that a browser rendered it.
- **Preview:** pending, ready, failed, or source-only/unavailable for an exact
  revision and load generation. It is ephemeral browser evidence, not durable
  revision truth and not transferable from another browser/load.

On parse, unsupported syntax, layout, quota or integrity failure, identify the
newest selected revision, display a bounded trusted error and expose its source
and View previous action. Do not retain the previous image as if the update
succeeded. Discard failed startup output and stop scripts. Selecting the previous
revision is explicit and preserves existing pin/branch behavior. A browser render
failure does not itself roll back a committed revision or fail its assistant turn.
Actual turn cancellation still discards stages and restores committed selection.

Stale worker successes/errors cannot change a newer load's status. Error text is
plain bounded data, never HTML; diagnostic logs/tool cards omit source/labels and
arbitrary parser exceptions. Returning an error to the assistant uses the existing
user-confirmed unsent draft path, not automatic submission or tool retry.

## Qualification and release gates

Implementation planning must assign these gates to atomic, independently testable
Backlog slices. This design task is not a blanket implementation or release claim.

1. **Semantic fixtures:** exact models and stable geometry for TD/LR flows,
   rectangle/rounded/diamond nodes, a six-node branch/rejoin with labeled arrows,
   three-participant solid/dashed sequences and every admitted note placement.
   Test long labels, escaping, ampersands, non-Latin text, emoji/combining marks,
   RTL, empty/malformed input and conflicting declarations. Reject every excluded
   operation explicitly, including valid upstream syntax outside the subset.
   Verify pinned Unicode segmentation/width behavior without native `Intl`,
   default typography, narrow-viewport scrolling, CSS inheritance versus explicit
   overrides, geometry bounds and missing-glyph source recovery.
2. **Composed budget tests:** four small diagrams together and a mixed interactive
   HTML + flow + sequence document must render under unchanged V1 ceilings.
   Exercise each individual/aggregate boundary and combined exhaustion, including
   many small labels, dense DAGs, routing/wrapping stress and malicious parser
   inputs. Show work/memory/patch counters; qualification failures cannot count as
   successful limit refusals without the expected bounded reason.
3. **Real browser boundary:** run the existing mandatory adversarial zero-egress
   suite with the exact added trusted asset closure and diagram-specific payloads.
   Prove no generated/native fallback, no post-start resource activity, no private
   handle recovery, and termination for initialization/parser/layout exhaustion.
   Inspect successful flow/sequence screenshots and error recovery; report actual
   browser/platform coverage and skips without claiming untested parity.
4. **One profile authority:** test every row in the selection table through native
   and served paths, including rename, staged history, promotion rollback/retry,
   cancellation, replay, historical branches, concurrent stale updates and
   prepared-plan mismatches. Removing a diagram must not downgrade its profile.
5. **Honest status/recovery:** source acceptance without a browser, parse failure
   after commit, failure after a previously good preview, stale ready/error events,
   View previous, explicit disable during a live load, restart into a revoked
   profile, served parent/child snapshot mismatch, scripts-disabled open, source
   copy/download and confirmed unsent repair drafts. Prove that a browser refresh
   is not mistaken for host policy reload and prior-load reservations cannot
   survive restart or disable.
6. **Real portability:** conversation and Chatbook export/import, import-as-new and
   same-identity restore preserve exact source/profile and graph. Unknown/revoked
   profiles stay inert; archive-supplied library bytes cannot execute. JSON source
   serialization alone is insufficient evidence. No Canvas sync payload changes.
7. **Supply chain and usability:** reproducible verified build, notices and package
   data inclusion; missing/tampered asset refusal; source fidelity and accessible
   descriptions; actual native system-browser and same-origin served workflows.
   Publish measured cold-start and near-limit timings, not just the spike samples.
8. **Authoring and compatibility:** profile-specific bounded model guidance with
   both examples, correct historical-profile selection and unavailable-profile
   messaging; source-free repair hints/projections; no automatic browser-error
   submission. Engine/facade/Unicode changes must allocate a new profile; approved
   nonsemantic security fixes must invalidate derived caches while preserving
   accepted semantic fixtures. All guidance examples are executable test fixtures.

## Scope boundaries and review corrections

No React/D3 catalog, full Mermaid, editor, live diagram API, hosted rendering,
multi-file/VFS, filesystem/network/cookies, synchronization, collaboration or
standalone SVG export. Canvas sync remains explicitly deferred under
[TASK-31003](../../../backlog/tasks/task-31003%20-%20Define-server-synchronization-contract-for-Canvas-artifacts.md).

The seven approved review corrections are represented by: security-first profile
refusal; separate save/preview state; whole-document quotas; explicit semantic
subset validation; one profile authority; once-before-script declarations with
safe escaping; and honest layout/fidelity qualification. The only new numeric
values are proposed admission ceilings and typography defaults above, to be
reviewed with this written spec. The follow-up four corrections define
restart-required policy deployment, complete profile compatibility, profile-aware
assistant guidance and bounded/default-styled geometry. User approval of this
document precedes an implementation plan.
