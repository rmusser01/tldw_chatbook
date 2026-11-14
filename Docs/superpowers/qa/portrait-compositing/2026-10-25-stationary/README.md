# Stationary codec portrait correction — TASK-36699

The user rejected the original seven-character recording because whole portraits
scrolled up and down. The default one-pixel shift was conspicuous on the native
24px fallback and the exporter magnified it further. Set the existing idle-offset
token to zero and rebuild its generated CSS copies. Portrait registration remains
fixed; audio-reactive mouth selection and independent blinking continue.

The new packaged-clock regression failed against the original value and passes
with zero while exercising all six roles. **67 targeted checks passed**, covering
portrait state, avatar rendering, Console performances/radio, design tokens and
CSS bundle synchronization. Ruff, formatting, diff checks and independent review
passed. No new ADR is required: ADR-242/243 apply; their source-linked motion
specification and design language now describe stationary portraits.

Fresh live Pocket TTS / alba calls captured all seven selected characters through
the native WAV/CoreAudio path. All **2,993** native observations had zero offset;
each captured image byte-equalled its stationary prepared role. Only four unique
native images were needed per character (the three mouths plus the blink actually
encountered), rather than the hundreds of shifted images in the rejected run.
The enlarged comparison adds no translation. See [live-summary.json](live-summary.json).

The local corrected MP4 and full observations are retained in the task's output
folder `codec-stationary-recording-2026-10-25`. This is a headless mounted-app
capture and explicitly labelled enlarged artwork replay, not graphics-terminal
hardware qualification. Audio is reconstructed from submitted buffers/DAC estimates,
not acoustic loopback. Existing timing hitches, brief pose reversals and low fallback
resolution remain separate concerns. The configured live image-provider gap
TASK-36555 AC5 remains open. No broader speech-quality or full-suite signoff is
claimed. Original artwork, user profiles and configured voices were unchanged.
