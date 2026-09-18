# Settings — Speech & TTS

The existing Textual Settings workbench and ADR-150/161 govern this surface.
Keep the detail pane as its single scroll owner and preserve the distinction
between global configuration and Speech Lab operations from ADR-039.

Global defaults, selected-provider setup, conversation/realtime options and
collapsed configuration details precede Save, Revert, Restore Non-secret
Defaults and Open Speech Lab. Ordinary Save validates and writes locally;
connection checks, discovery, model loading and synthesis belong to Speech Lab.
Credential mutations remain explicit and separate from ordinary Save.

The Voice value control is a select with a neighboring Browse action, not a
single-line button strip. Let this row grow to the control height, reserve the
button's width in the horizontal layout, and stack both controls when the
Speech form is narrow. Preserve focused control identity through reflow and
verify complete compositor paint, including the Browse label. Terminal width
alone does not identify the layout: the Settings rail and inspector also consume
width. TASK-32756 reproduced the horizontal failure at 190 columns after the
170-column form had already stacked.

Horizontal field containers must take the width remaining after their labels.
A 100%-width container plus a 24-column sibling label exceeds the form even
when its child control correctly uses fill width.

Custom IDs remain staged until Save. Revert restores the last loaded/saved
snapshot. Leaving an edited Speech category requires Save, Discard or Cancel;
Cancel retains the draft. This differs from categories that preserve drafts
silently across navigation.

TASK-32756 evidence lives in
`Docs/superpowers/qa/2026-09-17-settings-speech/README.md`. Its bounded scope is
configuration and keyboard navigation; it does not qualify provider availability,
audio generation/playback, model installation, realtime sessions or managed
audio.cpp lifecycle operations.
