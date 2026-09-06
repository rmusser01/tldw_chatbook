# Canvas V1 runtime compatibility and security boundary

Canvas V1 accepts one complete, self-contained HTML document. Chatbook compiles
that source into a typed render plan; the browser never parses the source as
markup and never evaluates its scripts in a native JavaScript realm. A trusted
renderer constructs the native DOM from allowlisted operations, while a fresh
QuickJS-WASM context in a dedicated worker runs generated classic scripts
against a virtual DOM. There is no native-JavaScript fallback.

## Author-facing JavaScript subset

V1 provides ordinary ECMAScript language features plus this deliberately small
page facade:

- `document.documentElement`, `document.body`, `getElementById()`, and simple
  `querySelector()` / `querySelectorAll()` selectors consisting of one ID,
  class, or tag selector;
- `createElement()`, `createElementNS()` for the allowed SVG namespace, and
  `createTextNode()`;
- node `textContent`, `id`, `className`, form `value`/`checked`/`disabled`/
  `selected`, allowlisted attributes, an allowlisted `style` facade, and
  `appendChild()`, `insertBefore()`, and `removeChild()`;
- bubbling `click`, `input`, `change`, `focus`, `keydown`, `keyup`, `submit`,
  and `reset` events with `preventDefault()` and `stopPropagation()`;
- bounded `setTimeout()`, `setInterval()`, and their cancellation functions;
- `JSON`, a bounded non-native-output `console`, and passive structural SVG;
- `canvas.submit(jsonValue)` and `canvas.download(jsonValue)`, which only emit
  typed requests. This runtime performs no submission, composer insertion, or
  download; the later trusted product gateway owns confirmation and effects.

Scripts are classic scripts. Imports, exports, module loading, external script
sources, inline event attributes, active SVG, custom elements, markup sinks,
external URLs, and multi-route application behavior are unsupported. Clicking
a local submit control produces a virtual `submit` event while native form
submission remains suppressed. Reset listeners run, but V1 does not emulate the
browser's implicit form-reset algorithm.

Textarea values start from their text content. Select values start from the
selected option, or the first eligible option for an ordinary single select;
options without a value attribute use their text. The virtual values used by
`canvas.submit(...)` track supported user edits and option selection changes.
Detaching and reinserting controls preserves edited values, including a
textarea whose default text changes while detached.

## Absent capabilities

Generated code has a QuickJS `globalThis`, not the worker or browser global.
The facade does not expose native `window`, `self`, `parent`, `top`, DOM,
`location`, `navigator`, fetch/XHR, WebSocket/EventSource/beacon, workers or
`importScripts`, a module loader, storage, cookies, caches, filesystem/file
pickers, clipboard, dialogs, popups, native object URLs, `SharedArrayBuffer`,
`Atomics`, or WebAssembly compilation. Generated prototype or global mutations
therefore remain in the disposable QuickJS realm. Trusted install, operation,
dispatch, drain, and timer controls are private host-held QuickJS handles, not
properties of generated `globalThis`. Transaction serialization uses bootstrap-
captured intrinsics and null-prototype transport records. Timer enumeration
uses captured `Map.forEach` rather than a mutable iterator path, and the native
worker independently revalidates timer count, identity, uniqueness, and delay.

CSS is parsed twice and inserted by the trusted renderer through CSSOM. Only
flat style rules and media rules containing supported descendants are accepted;
nested descendants inside a style rule fail closed. Resource and
computed-value functions such as `url()`, `image-set()`, `var()`, `env()`, and
`attr()` fail closed. Passive images must be compiler-issued PNG, JPEG, GIF, or
WebP handles. Before creating its own object URL, the renderer rechecks the
bytes and format structure, rejects animation, bounds declared dimensions and
pixels, and requires a successful time-bounded native decode with matching
dimensions. SVG animation and link elements are not supported.

## Fixed budgets

| Boundary | V1 ceiling |
| --- | ---: |
| UTF-8 source document | 512 KiB |
| Passive image count | 64 |
| One encoded passive image / all encoded images | 1 MiB / 4 MiB |
| One image width or height / decoded pixels | 4,096 px / 4,194,304 px |
| Aggregate decoded image pixels | 16,777,216 px |
| Image frames | 1; GIF/APNG/WebP animation is rejected |
| Native image decode | 1 s each / 3 s total preparation |
| DOM nodes / CSS rules / generated script text | 1,800 / 900 / 256 KiB |
| Node/asset/patch/bridge identifier | 256 UTF-8 bytes |
| Event value / event key | 16 KiB / 64 UTF-8 bytes |
| QuickJS heap / stack | 32 MiB / 512 KiB |
| Generated startup / one event or timer callback | 250 ms / 50 ms |
| Trusted worker startup / startup-response backstop / event-response backstop | 10 s / 750 ms / 250 ms |
| Pending QuickJS jobs | 100 per operation |
| Live timers / timer firings | 64 / 100 per second |
| One timer delay | 2,147,483,647 ms |
| Event listeners / queued native events | 500 / 100 |
| Typed patches / accepted mutations | 500 per operation / 2,000 per second |
| Typed bridge requests | 16 per operation / 32 per second |
| Submit request / download request | 16 KiB / 10 MiB JSON |
| JSON structural depth | 16 |
| QuickJS console | 100 entries and 16 KiB total text |
| One worker transaction / one renderer plan or message | 12 MiB |

The DOM, CSS-rule, and per-operation patch ceilings were reduced after the
single-host Task 7.2 qualification. The remaining ceilings were retained; no
security ceiling was raised. The reproducible probe uses only explicitly
identified, agent-authored synthetic documents and does not call or sample an
LLM provider.

## What users see when a quota is exceeded

| Exceeded boundary | Result visible to the user |
| --- | --- |
| Source bytes, DOM nodes, CSS rules, script bytes, asset count/bytes/pixels/frames, or unsupported image decode | The document is refused before it can render or stage. The Canvas error identifies the bounded category and offers the normal repair path; no partial preview is committed. |
| Conversation Canvas count, revision count, committed bytes, or staged bytes | The create/update is refused. Existing committed Canvases and history remain unchanged; Chatbook never deletes history to make room. |
| QuickJS heap or stack | The worker is discarded, the last inert committed preview remains, and Canvas reports that scripts are disabled for the load. No native-JavaScript fallback runs. |
| Generated startup, event, timer, or pending-job budget | Canvas stops the worker and reports a bounded runtime-timeout/job failure. The shell stays responsive and the inert document remains available. Trusted WASM loading and native image preparation use their separate, longer clocks shown above. |
| Timer, listener, event-queue, patch, mutation-rate, or bridge-rate budget | The operation fails closed. A rejected patch transaction is all-or-nothing, so none of that operation's DOM mutations appear; scripts are disabled for the load. |
| Submit bytes/JSON depth or download bytes/type | No confirmation or host effect occurs. The invalid bridge request is refused inside the disposable worker and the user sees the bounded Canvas failure state. |
| Browser delivery cache or frame-capability capacity | The affected load is unavailable or must be retried; another conversation or Canvas is never substituted. |

These messages deliberately omit generated source and arbitrary runtime output.
Safe category, current size/count where available, and maximum are sufficient
for an assistant-authored repair without putting document contents into logs.

Plans and worker transactions use two phases. A complete plan is decoded and
validated as detached assets, CSS, and DOM before one live commit. A complete
transaction is applied to a detached shadow tree, every bridge request is
validated, and an immutable mutation journal is produced before live commit.
Only then is that journal replayed synchronously against the stable live node
maps. An invalid plan therefore leaves no DOM, stylesheet, or asset attachment;
an invalid transaction leaves the previously committed inert document
unchanged. A valid transaction preserves native control identity, dirty value,
focus, selection, and queued-event ordering unless its validated patches
explicitly change or remove that control. Either failure terminates and
discards the worker, reports a bounded failure, and marks scripts disabled.
Committed passive-asset URLs remain owned by and usable in that inert renderer
document, then are revoked when it exits.

## Browser containment

The renderer is a real HTTP document in an iframe with
`sandbox="allow-scripts"` and no same-origin, forms, popups, downloads,
top-navigation, modals, or storage privileges. Its response uses
`default-src 'none'`, `connect-src 'none'`, no fonts/media/frames/objects/forms,
blob-only passive images, and only the packaged renderer plus the minimum
WASM/worker allowances. `wasm-unsafe-eval` permits the packaged QuickJS engine
to compile its embedded WASM; `unsafe-eval` and inline native scripts are not
permitted.

An opaque-origin iframe cannot directly construct a same-origin module worker
in Chromium. The trusted renderer therefore creates a fixed `data:` module
bootstrap that imports the exact packaged worker URL; generated content never
contributes bytes or a URL to that bootstrap. The worker then imports the
integrity-checked, single-file QuickJS asset before the generated-execution
boundary. The mandatory Chromium harness withholds that boundary acknowledgment
unless it has observed exactly the five successful, completed trusted HTTP
GETs, exactly the fixed `data:` bootstrap worker, the plan-derived number of
successful completed local `blob:` image loads, no foreign startup observation,
and no independently owned egress-server receipt. Runtime asset loading after
that boundary is forbidden.

Firefox and WebKit behavior is tested when their Playwright engines are already
installed. Chromium is the mandatory release gate. The shipped Canvas V1
product tools and UI use this reviewed boundary; there is no less-restricted
fallback when the required runtime or containment checks are unavailable.
