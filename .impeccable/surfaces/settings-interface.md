# Settings — Appearance, Theme and Splash

Preserve the existing Textual Settings workbench, ADR-150 tokens and ADR-161
component patterns. Its detail pane owns scrolling; nested editors grow with
their forms. The three categories have distinct commit models under ADR-033.

Appearance stages launch defaults. Preview changes runtime presentation, Save
persists the draft, and Revert restores its saved baseline. Theme browsing and
editing remain separate from Apply (runtime), Save (a custom theme file), and
Set as launch default (the startup preference). An instant launch-theme save
updates Appearance's baseline without erasing explicitly staged edits.

Splash preferences save immediately, with numeric fields submitting on Enter.
Show pending, file-write failure and post-write refresh failure truthfully.
Preserve keyboard focus and newer typed text while a save finishes. Gallery
selection and Play selected only preview cards; Default card owns startup
selection.

At compact widths, stack Appearance fields and Splash's gallery. Theme palette
rows reserve room for the full hex input and swatch; the tree fits within the
detail viewport. One-row checkbox fields use token-backed compact controls.
Verify actual painted values and full action labels through the keyboard ring,
including after resizing, rather than relying on widget presence.

TASK-32757 evidence: `Docs/superpowers/qa/2026-09-17-settings-interface/README.md`.
The bounded review covers local settings, theme files and representative static
splash previews. It does not qualify every animated card or external services.
