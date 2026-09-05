# Canvas — create, revise, inspect, and recover interactive artifacts

Canvas is the Console companion for a substantial visual or interactive result:
a chart, calculator, form, diagram, small simulation, or polished single-page
document. Ordinary prose and short code snippets are usually clearer in chat.
Canvas V1 accepts one complete, self-contained HTML document per revision.

## Create or open a Canvas

Ask the Console assistant to create a Canvas and describe the interaction you
want. A conversation may own multiple named Canvases; the Canvas selector shows
only those reachable from the conversation's active message branch. One Canvas
and revision is selected in each live Chatbook session.

Assistant `html` code blocks also offer **Open in Canvas**. Repeating that action
reopens the same imported Canvas; **Open as new Canvas** deliberately creates a
separate identity. Successful assistant creates open automatically when
**Settings > Privacy & Security > Open Canvas automatically after a successful
create** is enabled. Updates hot-reload an already-open preview but do not force
a closed preview to open.

The revision selector and transcript Canvas cards open exact immutable
revisions. **Pin revision** stops following newer changes; **Follow latest**
returns to the branch-resolved head. Choosing an older revision changes the base
for the next edit: that edit creates a new branch instead of rewriting history.

## Revise, rename, and undo

Tell the assistant to update the Canvas. It reads the selected complete source
and revision ID, then supplies one complete replacement document against that
exact parent. If another edit changed the parent, the update reports a conflict;
ask the assistant to reread and retry rather than overwriting the newer work.

Changing the title in the Canvas toolbar also creates a revision. After an
update, **View previous** opens its parent revision. This is Canvas undo: it
selects the earlier immutable version, and a later edit branches from it. It
does not rewind chat history or delete the newer revision.

Canvas revisions created by an assistant commit atomically with that assistant
turn. Cancelling or failing the turn discards its staged changes and leaves the
last committed preview available.

## Temporary chats and portable history

A Canvas in a temporary chat carries a **Temporary** badge and remains in
memory. Use the chat's **Save** action to promote the complete message tree and
Canvas graph together. Closing an unsaved temporary chat destroys that staged
Canvas state; closing only the browser preview does not.

Durable Canvas history is included in a Canvas-bearing **Chatbook archive**.
Those archives use format 3.0 and preserve the message tree, Canvas identities,
revision parents, origins, titles, runtime profiles, and inert source files.
Transcript-only exports are not a full Canvas-history backup. Canvas is local to
the Chatbook host and is not included in server synchronization.

## Source and confirmed actions

The trusted Canvas toolbar offers:

- **Inspect source** for the exact read-only revision source;
- **Copy source** with a warning that running it elsewhere leaves Canvas
  protections;
- **Download** for an inert `canvas-source.canvas.html.txt` source file;
- **Download as runnable HTML**, an explicit confirmed action that produces an
  `.html` file outside Canvas protections.

Generated code cannot perform a host action directly. `canvas.submit(...)`
opens a confirmation showing the complete bounded value; confirmation inserts
an unsent Console draft for review. `canvas.download(...)` shows filename, MIME
type, and size before confirmation starts the trusted browser download. Cancel,
expiry, a changed draft/selection, or a stale capability causes no insertion or
download.

## Compatibility and recovery

Canvas V1 supports inline HTML/CSS and classic scripts through its documented,
bounded DOM, forms, events, styles, passive SVG, timers, JSON, console, submit,
and download facades. It does not support modules, external scripts or URLs,
browser networking, storage, cookies, filesystem access, Chatbook APIs, or the
parent-page DOM. See [Canvas V1 runtime compatibility](../../Canvas/V1_RUNTIME_COMPATIBILITY.md)
for the exact API subset and fixed quotas.

Unsupported content is refused or appears as a bounded **Compatibility note**.
If a generated script fails or exceeds a runtime boundary, Canvas discards that
script worker and keeps the inert document available. Choose **Reopen with
scripts disabled** to inspect the non-scripted document and source. There is no
native-JavaScript fallback. Ask the assistant to adapt the source to Canvas V1,
then create a new complete revision.

## Security boundary and availability

Generated Canvas code has strict zero egress: no network, host filesystem,
cookies or storage, Chatbook API, or parent DOM. The trusted shell, compiler,
renderer, gateway, and user-confirmed submit/download actions are host product
code outside that generated runtime. Canvas is therefore not a general sandbox
for arbitrary code copied from elsewhere, and saving runnable HTML explicitly
leaves its protections.

**Settings > Privacy & Security > Enable Canvas tools, actions, and browser
delivery** is the global kill switch. Turning it off immediately revokes Canvas
execution and browser delivery while preserving stored artifacts. Re-enabling
the saved setting requires restarting Chatbook. For remote browser admission,
TLS, and incident response, see the [Web Server operations guide](../../../tldw_chatbook/Web_Server/README.md).

V2 bundled libraries, V3 multi-file virtual filesystems, and Canvas server
synchronization are deferred. V1 does not emulate them.
