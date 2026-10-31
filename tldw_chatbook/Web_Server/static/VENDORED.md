# Vendored static assets

## qrcodegen.js

- **Upstream**: Project Nayuki, "QR Code generator library" —
  <https://www.nayuki.io/page/qr-code-generator-library>
- **Repository**: <https://github.com/nayuki/QR-Code-generator>
- **Pinned version**: tag `v1.5.0`
- **Source URL** (exact file at that tag):
  <https://raw.githubusercontent.com/nayuki/QR-Code-generator/v1.5.0/javascript/qrcodegen.js>
- **License**: MIT (the file carries the full license header; no
  modifications have been made)
- **SHA-256**:
  `99a61998d4cedf7b1d927ac8ae2f41e0c64a5c89d7de0a76bffd3cb4dda40e89`
- **Vendored**: 2026-10-15 (ADR-233, task 5 — QR device pairing)

### Why v1.5.0 is the pinned tag

`v1.5.0` is the **latest release tag that ships the plain-JavaScript
`javascript/qrcodegen.js` file**. From `v1.6.0` onward the upstream project
consolidated the JavaScript flavor into TypeScript sources
(`typescript-javascript/qrcodegen.ts`) with a build script, and no release
tag after v1.5.0 contains a compiled `qrcodegen.js`. Vendoring the official
released file verbatim (rather than transpiling the TypeScript ourselves)
keeps the artifact byte-identical to upstream and verifiable by hash.

### Upgrade policy

To upgrade, re-check the tags from
<https://github.com/nayuki/QR-Code-generator/releases>; if a newer tag ships
a plain `qrcodegen.js`, vendor it verbatim and update the pin and hash above.
If upstream never resumes shipping the plain-JS file, keep this pin.
