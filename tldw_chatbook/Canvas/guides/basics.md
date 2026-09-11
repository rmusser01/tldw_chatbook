# Canvas basics

After the user accepts a Canvas offer, load only the topics needed for that
artifact. An explicit creation request or requested edit already authorizes the
work. Reuse guidance in context. Do not generate source while waiting for consent.

Supply one complete, self-contained HTML document with a doctype, html, head,
UTF-8 meta, title, and body. Use inline CSS, ordinary HTML labels, and passive
structural SVG. Start with readable text, strong contrast, and a layout that fits
narrow widths. Label every control, keep natural keyboard tab order and visible
focus, and convey chart values in text as well as shape or color.

## Complete passive comparison example

Required profile: `canvas-v1`. This static comparison needs no script. Its SVG
scales to the available width; the caption preserves exact values in plain text.

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Weekly time comparison</title>
<style>
body { margin: 0; padding: 16px; font-family: sans-serif; color: #182536; background-color: #ffffff; }
main { max-width: 560px; margin: 0 auto; }
h1 { font-size: 24px; line-height: 1.2; }
p { line-height: 1.5; }
figure { margin: 24px 0; }
svg { display: block; width: 100%; height: auto; }
text { font-family: sans-serif; font-size: 18px; fill: #182536; }
figcaption { margin-top: 12px; line-height: 1.5; }
</style>
</head>
<body>
<main>
<h1>Time spent each week</h1>
<p>Compare two sample workflows. Shorter bars mean fewer hours.</p>
<figure>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 150" role="img" aria-label="Manual: 8 hours. Assisted: 5 hours.">
<text x="0" y="24">Manual: 8 hours</text>
<rect x="0" y="36" width="288" height="24" fill="#245d91"></rect>
<text x="0" y="98">Assisted: 5 hours</text>
<rect x="0" y="110" width="180" height="24" fill="#31745a"></rect>
</svg>
<figcaption>Manual: 8 hours. Assisted: 5 hours. Difference: 3 hours per week.</figcaption>
</figure>
</main>
</body>
</html>
```

## Create, revise, and report

Use `canvas_create` for a requested new artifact. For edits, identify the intended
Canvas with `canvas_list` if needed, then `canvas_read` its current selected source
and revision. Send a complete replacement through `canvas_update` with that
`expected_parent_revision_id`; do not overwrite unseen work. See `repair` for
conflicts and diagnostics. Staged source still depends on turn settlement; saved
source does not prove a browser preview is ready.

Guides do not admit profiles. Current tool/profile guidance and runtime checks
win over these examples. Preserve a historical revision's exact profile. An
unavailable profile stays source-only; adapting it requires an explicit new Canvas.

Prefer `background-color` and explicit `border-width`, `border-style`, and
`border-color` declarations. Browser CSS parsing expands `background` and `border`
shorthands into properties outside the runtime allowlist, even when compilation
accepts the source.

Canvas is a constrained page runtime. Do not assume React, D3, Chart.js, CDNs,
modules, native `window`, HTML canvas drawing APIs, CSS custom properties or
`var()`, networking, storage, filesystem, parent DOM, or Chatbook API access.
Use the `controls` topic for supported classic-script interaction and `mermaid`
for the exact packaged diagram subset; do not fetch external libraries.
`canvas.submit` and `canvas.download` emit requests for existing confirmed host
actions, not permission to submit or download automatically.
