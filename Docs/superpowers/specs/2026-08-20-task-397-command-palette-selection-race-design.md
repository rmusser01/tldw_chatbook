# TASK-397 Command Palette Selection Race Design

## Summary

Fast keyboard selection in Textual's command palette can race the palette's
asynchronous result refresh. When a query has several matches and providers are
still producing results, a user can move the highlight and press Enter while
Textual clears and rebuilds the option list. The result can be a reset selection,
an ignored selection gesture, or a palette dismissal that does not run the command
the user acted on.

TASK-397 will first reproduce or rule out the reported failure under a deterministic
pilot harness. Regardless of whether the exact live symptom is repeatable, the app
will add a narrow compatibility shim that stops result gathering when keyboard
navigation or Enter selection acts on an actionable visible list. The
displayed options then remain stable for the remainder of that selection gesture.
Navigation before any result appears, or while only Textual's stale disabled
“No matches” placeholder is visible, continues waiting normally. The underlying
framework behavior will also be reported upstream with a minimal reproduction.

## Context and Evidence

- The project pins Textual `8.2.8` exactly.
- `TldwCli` registers several independent command providers. Their results are
  consumed concurrently by Textual's built-in `CommandPalette`.
- Textual batches provider results. Each batch and the final refresh calls
  `clear_options().add_options(...)` and then assigns `highlighted = 0`.
- A user's Down or Enter can therefore act while more provider results are pending.
- The same refresh and highlight-reset behavior is present in current upstream
  Textual `main` as of 2026-08-20.
- The app's providers use the ordinary Textual `Provider` contract and do not
  independently mutate the palette or its highlight.

The pilot test remains the authority for the exact failure mode. Source inspection
establishes the race window but does not substitute for mounted behavior evidence.

## User-visible Contract

When command results are visible:

1. Down, Up, Page Up, Page Down, Ctrl/Shift+Home, Ctrl/Shift+End, or Enter acts on
   one stable snapshot of the displayed command list. Plain Home and End retain
   their existing command-input editing behavior.
2. Once the first list-navigation or list-selection gesture occurs, late provider
   results do not replace the options beneath that gesture.
3. Enter runs the command represented by the visible highlight exactly once.
4. The palette closes using Textual's existing selected-command behavior.
5. Escape remains cancellation and runs no command.
6. Opening the palette, typing a query, settled-result selection, provider matching,
   command ordering before interaction, and mouse selection retain their existing
   behavior.
7. Navigation keys pressed before an actionable result list is visible do not cancel
   provider gathering, including when a replacement query has started while the old
   disabled “No matches” placeholder remains visible.

The mitigation intentionally prefers predictable selection over including results
that arrive after the user begins navigating. The user can change the query or reopen
the palette to obtain a new result set.

## Design

### Compatibility subclass

Add a small app-owned subclass of `textual.command.CommandPalette`, named
`StableCommandPalette`.

Override the palette's `_action_command_list(action)` compatibility seam. Query the
real `CommandList`; when `_list_visible` is true, at least one option exists, and the
first option is not Textual's `_NO_MATCHES` placeholder, cancel the palette's current
gather worker via the existing `_cancel_gather_commands()` method. Then delegate every
action to the base implementation unchanged. If no actionable visible snapshot
exists, delegate without cancellation so early navigation or a stale disabled
placeholder cannot strand the palette without replacement results.

This one interception point covers:

- cursor movement delegated by Down and Up;
- page, first, and last navigation;
- selection delegated by Enter.

No provider is changed. No Textual source is copied. No timing constants, sleeps, or
debounces are added to production behavior.

The subclass is an explicit compatibility boundary around protected Textual methods.
That dependency is acceptable because Textual is exactly pinned. Any upgrade must
explicitly review both protected seams even if behavioral tests remain green because
upstream may have made the compatibility override redundant.

### Application integration

Override `TldwCli.action_command_palette()` with the same two guards used by Textual's
`App.action_command_palette()`:

- `self.use_command_palette` is true; and
- no command palette is already open.

Push `StableCommandPalette(id="--command-palette")`. Preserve Textual's canonical ID,
open-state class, calling-screen behavior, provider discovery, messages, dismissal,
and delayed callback execution.

### Why not preserve the highlighted identity across refreshes?

Restoring a matching highlight after every batch still clears and replaces the option
objects while a selection event may be queued. It also has to define identity across
providers with equal labels or callbacks. Freezing on interaction is smaller and
matches the user's intent: the visible list they acted on is authoritative.

### Why not change providers?

The providers follow the documented async-yield contract. Coordinating or buffering
them in application code would duplicate framework behavior and would not protect
against future slow providers.

### Why not vendor or upgrade Textual?

Vendoring the palette would copy hundreds of framework-owned lines. A dependency
upgrade is a separate, higher-risk task and current upstream `main` still contains the
same refresh behavior. The compatibility subclass is the smallest safe boundary.

## Deterministic Verification

### Pilot harness

Create a minimal Textual app with controlled providers, a patched
`textual.command.monotonic`, and callback counters:

- provider gates control exactly when each hit becomes available;
- the fake monotonic clock advances before a gated hit is released, crossing
  Textual's batch interval and materializing a partial multi-hit visible list;
- the harness proves that partial batch is visible while at least one provider is
  still pending;
- after navigation, the clock advances again and another gated hit is released to
  force the stock refresh before selection; and
- the harness records both refreshes, their option identities, and callback counts,
  avoiding wall-clock sleeps and scheduler luck.

Use the real mounted `CommandPalette`, `CommandInput`, `CommandList`, key bindings,
workers, option messages, dismissal, and deferred callback path.

### RED evidence

Run the harness against stock Textual behavior first. Record whether it reproduces the
original no-command symptom or a narrower selection-reset/ignored-gesture form. A
test is non-vacuous only when it proves that a late result refresh actually occurred
after the user began the selection sequence. This stock failure is a temporary
diagnostic and the standalone upstream reproducer; it is not committed as a failing
repository test.

Then point the application integration test at `StableCommandPalette`; before the
production override exists, the test must fail because `TldwCli` still opens the stock
palette or because the visible selection is not stable.

### GREEN evidence

Focused tests will prove:

- the app opens `StableCommandPalette` with the canonical ID;
- a multi-hit Down+Enter sequence while a provider is pending runs the visible
  highlighted callback exactly once;
- the gather worker is cancelled before a late result can rebuild the list;
- a passing stock-palette characterization proves the forced late refresh resets the
  acted-on highlight, while the stable palette prevents that refresh;
- navigation before the first visible result leaves gathering active and eventually
  displays the gated results;
- a new query followed immediately by navigation while the old disabled “No matches”
  placeholder is visible leaves replacement gathering active;
- settled multi-hit selection still works;
- Escape still closes without running a callback; and
- the existing command-provider and basic palette tests remain green.

The gate/clock tests must assert that the late provider reached its gate before
navigation, that the stock palette performed the post-navigation refresh/reset, and
that the stable palette's provider or gather worker observed cancellation. Tests must
assert callback identity and count, not merely that the palette closed.

## Upstream Report

After the mounted stock-palette reproduction is settled, file a Textual GitHub issue
containing:

- Textual and Python versions;
- a standalone minimal app/provider reproduction;
- the deterministic event ordering;
- expected and actual selected callback identities/counts;
- the relevant batch-refresh behavior; and
- the app-side freeze-on-interaction mitigation, clearly labeled as a workaround.

Link the issue in TASK-397 Implementation Notes. Do not claim the upstream issue is
the exact original live symptom unless the pilot reproduces that symptom; describe
the narrower confirmed race honestly if that is all the harness proves.

## Error and Compatibility Handling

- A Textual upgrade requires explicit review of `_action_command_list()` and
  `_cancel_gather_commands()`. Behavioral tests remain authoritative because an
  upstream fix may legitimately make the compatibility override redundant.
- Provider exceptions continue to be handled by Textual.
- Cancellation uses Textual's existing worker-group mechanism and does not cancel
  command callbacks.
- A selection already committed by Textual remains exactly-once; the subclass does
  not invoke callbacks itself.

## Scope

In scope:

- the compatibility subclass;
- `TldwCli` palette construction;
- deterministic mounted regression tests;
- the upstream issue; and
- task/spec/plan notes required by closeout.

Out of scope:

- provider ranking or search semantics;
- palette styling;
- replacing Textual's command palette;
- changing Textual's dependency pin;
- mouse behavior beyond regression coverage; and
- unrelated keybinding changes.

## ADR Check

ADR required: no.

ADR path: N/A.

Reason: this is a localized compatibility shim around an exactly pinned framework
component. It preserves provider, application, and UI ownership boundaries and does
not introduce a durable architectural decision.
