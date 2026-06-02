# Console Background Effects Design

Date: 2026-06-02
Status: Approved by user and local spec review
Primary Repo: `tldw_chatbook`
Scope: Console background effects, transcript/event stream presentation, config defaults, and in-app Console settings

## Summary

Console should support optional ambient background effects such as snow, rain, and matrix-style glyph rain while a user has the Console open. The default experience remains unchanged: background effects are off unless the user enables them.

The first version targets the main Console transcript/event stream area by default so the effect does not interfere with rails, recovery controls, provider settings, composer controls, or inspectors. Users may opt into a broader workbench scope, but that mode is treated as advanced because it can reduce contrast around controls.

Ambient effects are presentation only. They do not become transcript content, do not affect Console state, do not participate in message reconciliation, and do not change exports, copy actions, keyboard selection, streaming, or provider behavior.

## Spec Review

Local spec review passed on 2026-06-02 using the required completeness, consistency, clarity, scope, and YAGNI checklist. No implementation-planning blockers were found. External subagent review was not run because the available subagent tool is restricted to cases where the user explicitly asks for delegation.

## Current Context

Console is a Textual-native Chat screen with a dedicated workbench layout:

- `ChatScreen` composes `#console-shell`, `#console-workspace-grid`, `#console-main-column`, `#console-transcript-region`, and the native Console composer.
- `ConsoleSessionSurface` owns the session title, tab strip, task cards, and current `ConsoleTranscript`.
- `ConsoleTranscript` is a focusable `VerticalScroll` widget with message reconciliation, keyboard selection, selected-message actions, empty-state copy, and plain-text export behavior.
- Console settings already exist through `ConsoleSettingsModal`, while app-wide Console behavior is also exposed through the Settings screen and `[console]` config.
- Existing splash screen effects include matrix and environmental animations, but those are startup-oriented and should not be reused directly if doing so couples long-running Console behavior to splash internals.

The design should preserve the existing Console transcript ownership boundary. The effect layer can wrap or sit beside transcript rendering, but it must not be mixed into transcript rows.

## Goals

- Let users enable an ambient Console background effect from both config and in-app settings.
- Keep effects off by default for readability, performance, and reduced-motion safety.
- Make transcript-only scope the default and recommended scope.
- Offer an opt-in workbench scope for users who want the effect behind more of the Console workbench.
- Support `none`, `snow`, `rain`, and `matrix` effect choices.
- Bound animation cost through preset intensity and frame-rate limits.
- Keep message rows, selected-message actions, transcript focus, scrolling, and exports unchanged.
- Pause or avoid running timers when the effect is disabled or the target widget is not mounted.
- Use theme-aware, subdued colors and labels rather than hardcoded decorative color as the primary design language.

## Non-Goals

- Do not change the host terminal emulator background.
- Do not turn the Console into a decorative terminal skin.
- Do not animate rails, composer controls, provider recovery strips, inspectors, or settings by default.
- Do not add custom effect scripting or user-provided animation code.
- Do not make the effect part of transcript content, logs, copies, exports, or persisted messages.
- Do not redesign the Console layout, transcript message rows, session tabs, or composer.
- Do not introduce network access, external assets, or optional dependencies.
- Do not add per-session effect settings in the first version.

## Approved Direction

Use a dedicated background presentation layer, defaulting to transcript-only scope.

Rejected alternatives:

- CSS-only styling: simple and low risk, but it cannot produce convincing animated effects in Textual.
- Drawing effects inside `ConsoleTranscript`: compact, but it couples decoration to message reconciliation, selection, scrolling, and export behavior.
- Full-workbench effect by default: visually stronger, but too likely to reduce contrast around operational controls.

The selected approach keeps effects behind the live transcript surface while preserving the Console's existing state and interaction model.

## User Controls

Config defaults:

```toml
[console.background_effects]
enabled = false
effect = "none"       # none, snow, rain, matrix
scope = "transcript"  # transcript, workbench
intensity = "low"     # low, medium, high
fps = 6
```

In-app controls should expose the same behavior:

- Enabled: on/off.
- Effect: none, snow, rain, matrix.
- Scope: transcript recommended, workbench advanced.
- Intensity: low, medium, high.
- Frame rate: bounded numeric value or a small set of safe preset values.

These controls are app-level Console behavior settings, not current chat session settings. If they are surfaced in `ConsoleSettingsModal`, the implementation must make the persistence boundary clear: changes write to `[console.background_effects]`, not to `ConsoleSessionSettings`.

If the Settings screen already has a Console behavior category, it should show these options there as the durable configuration home. The Console settings modal may include a compact appearance section for convenience, but it must persist through the same app-level settings path.

## UX Rules

Transcript scope:

- The effect appears behind the main transcript/event stream region only.
- Rails, composer, provider recovery strip, settings summary, inspector, and workbench controls remain unaffected.
- Message rows remain readable. It is acceptable for message rows to keep normal backgrounds while empty transcript space shows the effect more clearly.
- The transcript remains the focusable element for keyboard navigation.

Workbench scope:

- The effect may sit behind the broader Console workbench region.
- This scope is opt-in and should be labelled as visually busy or advanced.
- Controls must remain readable and focus-visible at supported terminal sizes.
- If workbench scope cannot preserve contrast reliably in implementation, it should fall back to transcript scope rather than shipping an unsafe visual mode.

Copy:

- Labels should be short: `Background effect`, `Scope`, `Intensity`, `Frame rate`.
- Scope labels should make the default clear, for example `Transcript (recommended)` and `Workbench (advanced)`.
- Avoid explanatory prose inside the main Console. Longer explanation belongs in tooltips or Settings help copy.

## Architecture

Introduce a small settings/display-state model:

```python
@dataclass(frozen=True)
class ConsoleBackgroundEffectSettings:
    enabled: bool = False
    effect: str = "none"
    scope: str = "transcript"
    intensity: str = "low"
    fps: int = 6
```

Exact names can be adjusted during implementation, but the object should satisfy these boundaries:

- It is app-level Console behavior state.
- It can be built from `app_config["console"]["background_effects"]`.
- It normalizes invalid values to safe defaults.
- It exposes bounded numeric values for animation code.
- It is independent of `ConsoleSessionSettings`.

Suggested widget structure:

- `ConsoleBackgroundEffect`: non-focusable animated renderer.
- `ConsoleTranscriptSurface`: host widget for transcript-only scope, containing the effect renderer and the existing `ConsoleTranscript`.
- Optional `ConsoleWorkbenchBackgroundSurface`: host or framing path for workbench scope if implementation can preserve controls and contrast.

`ConsoleSessionSurface` should continue to own the transcript area, but it may yield `ConsoleTranscriptSurface` instead of yielding `ConsoleTranscript` directly. Query contracts should remain stable where practical. Existing code that queries `#console-native-transcript` should still find the `ConsoleTranscript`.

## Rendering Model

Effects are cell-based and terminal-native:

- `snow`: sparse falling dots and stars.
- `rain`: vertical falling streaks using subdued punctuation or line glyphs that are safe in common terminals.
- `matrix`: subdued falling alphanumeric glyphs, using theme-aware dim green/accent styling where supported.

Rendering rules:

- The effect renderer owns its own timer.
- The timer starts only when enabled, mounted, and visible enough to draw.
- The timer stops on unmount and when settings change to disabled or `effect = "none"`.
- Frame rate is bounded.
- Particle counts derive from widget size and intensity presets.
- Resize handling regenerates or clamps effect state without raising errors.
- Rendering failure should disable the effect and leave the transcript usable.

The implementation may use `Static` text frames, `RichLog`-style renderables, or a custom widget render method. It should choose the simplest Textual-native approach that preserves layout stability and does not remount transcript rows.

## Config Behavior

Config loading should add safe defaults under `[console.background_effects]`.

Normalization rules:

- Missing section: use all defaults.
- Invalid `enabled`: coerce with existing boolean setting helpers where appropriate.
- Invalid `effect`: use `none`.
- Invalid `scope`: use `transcript`.
- Invalid `intensity`: use `low`.
- Invalid `fps`: clamp to a safe range, such as 1 through 12, with default 6.

The default generated config should document the available values. Existing config files without this section should continue to load without warnings or behavior changes.

## Accessibility And Safety

- Effects are off by default.
- The user can disable effects from both config and in-app settings.
- Important states remain text-labelled and never rely on the background effect.
- Focus outlines and selected-message styling must remain visible over the effect.
- If a reduced-motion preference is available through Textual or app settings, it should force effects off or require explicit override.
- Low intensity should be quiet enough for long sessions.
- Avoid bright full-saturation backgrounds. Accent color is earned by state and focus, not by decorative motion.

## Performance Rules

- No unbounded per-cell updates.
- No animation work when disabled.
- No transcript row remounts caused by effect frames.
- No frame rate above the configured safe cap.
- No background workers are needed for the first version.
- The renderer should degrade gracefully on very small terminal sizes by drawing fewer particles or nothing.

## Testing

Focused automated coverage should include:

- Config defaults include `console.background_effects` with effects disabled.
- Config normalization handles missing and invalid values.
- Settings UI shows effect, scope, intensity, and frame-rate controls.
- Saving settings persists through the same app-level Console config path used by other Console behavior settings.
- Disabled mode does not start an animation timer.
- Enabling transcript scope mounts the effect host while preserving `#console-native-transcript` query behavior.
- Transcript keyboard navigation and selected-message actions still work with the effect enabled.
- Transcript scope does not affect rails, composer, provider recovery strip, or inspector selectors.
- Workbench scope is opt-in and preserves required Console selectors.

Manual or visual QA should check:

- Empty transcript space shows the effect.
- Messages remain readable over each effect.
- Focus outlines remain visible.
- Small terminal sizes do not overlap text or controls.
- Effects stop or disappear when disabled without requiring app restart.

## Implementation Boundaries

The implementation plan should keep the first slice narrow:

1. Add settings model and config defaults.
2. Add transcript-scope rendering behind the existing `ConsoleTranscript`.
3. Add in-app settings persistence.
4. Add focused tests for config, settings, and transcript behavior.
5. Add workbench scope only if it can be done without compromising controls and selector stability.

If workbench scope proves risky during implementation, ship transcript scope first and keep workbench scope disabled or behind a documented future task.
