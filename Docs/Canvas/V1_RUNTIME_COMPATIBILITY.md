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
| DOM nodes / CSS rules / generated script text | 5,000 / 2,000 / 256 KiB |
| Node/asset/patch/bridge identifier | 256 UTF-8 bytes |
| Event value / event key | 16 KiB / 64 UTF-8 bytes |
| QuickJS heap / stack | 32 MiB / 512 KiB |
| Generated startup / one event or timer callback | 250 ms / 50 ms |
| Trusted worker startup / startup-response backstop / event-response backstop | 10 s / 750 ms / 250 ms |
| Pending QuickJS jobs | 100 per operation |
| Live timers / timer firings | 64 / 100 per second |
| One timer delay | 2,147,483,647 ms |
| Event listeners / queued native events | 500 / 100 |
| Typed patches / accepted mutations | 1,000 per operation / 2,000 per second |
| Typed bridge requests | 16 per operation / 32 per second |
| Submit request / download request | 16 KiB / 10 MiB JSON |
| JSON structural depth | 16 |
| QuickJS console | 100 entries and 16 KiB total text |
| One worker transaction / one renderer plan or message | 12 MiB |

Plans and worker transactions use two phases. A complete plan is decoded and
validated as detached assets, CSS, and DOM before one live commit. A complete
transaction is applied to a detached shadow tree and every bridge request is
validated before the live tree is swapped or any request is posted. An invalid
plan therefore leaves no DOM, stylesheet, or asset attachment; an invalid
transaction leaves the previously committed inert document unchanged. Either
failure terminates and discards the worker, reports a bounded failure, and marks
scripts disabled. Committed passive-asset URLs remain owned by and usable in
that inert renderer document, then are revoked when it exits.

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
installed. Chromium is the mandatory release gate. Canvas remains unavailable
to product tools and UI until the independent security review of this boundary
is accepted.
