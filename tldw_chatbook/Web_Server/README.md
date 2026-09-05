# Web Server Module for tldw_chatbook

This module provides web server functionality for running the tldw_chatbook Textual TUI application in a web browser using `textual-serve`.

## Installation

To use the web server functionality, install the optional dependencies:

```bash
pip install tldw_chatbook[web]
```

Or if installing from source:

```bash
pip install -e ".[web]"
```

## Usage

### Method 1: Using the --serve flag

Run the main CLI with the `--serve` flag:

```bash
tldw-cli --serve
```

Additional options:
```bash
tldw-cli --serve --host 127.0.0.1 --port 9000 --web-title "My TUI App"
```

### Method 2: Using the dedicated tldw-serve command

```bash
tldw-serve
```

With options:
```bash
tldw-serve --host 127.0.0.1 --port 9000 --title "My TUI App" --debug
```

## Configuration

You can configure default settings in your `~/.config/tldw_cli/config.toml`:

```toml
[web_server]
enabled = true
host = "localhost"
port = 8000
title = "tldw chatbook"
font_size = 12
debug = false
# Required for non-loopback access. Prefer the dedicated environment variable
# or OS keyring to storing this credential in plaintext here.
access_token = ""
# Exact browser-facing origin. Required for wildcard binds.
public_url = ""
# Set both for direct TLS, or neither when a trusted proxy terminates TLS.
tls_certificate = ""
tls_private_key = ""
# Literal immediate-proxy IP addresses only; forwarded headers from all other
# peers are ignored.
trusted_proxy_addresses = []
# Emergency plaintext development override; keep false for normal use.
allow_insecure_remote_http = false
```

`font_size` controls the browser terminal cell density. The default `12` keeps
the web UI close to native terminal screenshots; use `?fontsize=16` in the URL
or set `font_size = 16` if you prefer larger text.

## Remote access and authentication

The default `localhost` bind is local-only and permits automatic login from a
loopback browser. A non-loopback bind fails closed unless all of these are true:

1. a dedicated Chatbook web access token is available;
2. `public_url` names the exact browser-facing origin for a wildcard bind; and
3. the connection uses direct TLS or an HTTPS-terminating trusted proxy.

The admission token is resolved in this order:

1. `TLDW_CHATBOOK_WEB_ACCESS_TOKEN`;
2. `[web_server].access_token`;
3. OS keyring service `tldw_chatbook_web`, account `access_token`.

Use a password manager, service secret facility, or the dedicated keyring entry
to provision the value. Do not reuse an LLM-provider, MCP, or tldw API key, and
do not put the credential in a URL or command-line argument. Remote browsers log
in at `/auth/login`. Chatbook may also issue a one-use `/auth/bootstrap` link;
the nonce expires after 60 seconds, is consumed on exchange, and is removed from
the visible URL.

For direct TLS, configure both `tls_certificate` and `tls_private_key`. Behind a
TLS-terminating reverse proxy, set HTTPS `public_url` and list only the proxy's
literal immediate-peer IP address in `trusted_proxy_addresses`. The proxy must
overwrite forwarded client, host, and scheme headers. Authentication does not
encrypt traffic. `allow_insecure_remote_http = true` is an emergency development
override that exposes terminal, chat, Canvas, and credentials to network
observers and emits a warning; a firewall does not make plaintext equivalent to
TLS.

Successful login creates an opaque in-memory browser session. Sessions expire
after 30 minutes idle or eight hours absolute by default, and process shutdown
revokes all sessions and live channels. There is currently no logout or
individual-session revoke UI. Closing a Canvas preview revokes only that Canvas
view, not the browser's full Chatbook session. The dedicated token admits the
full Chatbook app on this host; it is not a multi-user account, filesystem, or
database isolation system. Canvas capability URLs are narrower: a copied or
guessed URL cannot reuse another browser session's Canvas authority.

### Incident response

If a token, browser session, proxy, or served host may be compromised:

1. stop the served Chatbook process to revoke every in-memory session and
   one-use bootstrap;
2. rotate the dedicated token in its actual source and remove the old value;
3. verify the certificate/private key, exact `public_url`, and literal trusted
   proxy list, then restart;
4. inspect host and reverse-proxy logs using your normal operational controls
   before admitting remote browsers again.

Turning off **Settings > Privacy & Security > Enable Canvas tools, actions, and
browser delivery** immediately revokes Canvas delivery and execution and keeps
stored artifacts. It does not revoke full Chatbook browser access. Re-enabling
Canvas requires saving the setting and restarting Chatbook.

## Privilege boundary

Served mode runs the full trusted Chatbook application with the permissions of
its host process; it is not an OS sandbox. User-armed raw CLI and Terminal
features retain their documented full host authority when available. The strict
zero-egress promise applies only to generated code inside the Canvas V1 runtime:
that code has no network, host filesystem, cookies/storage, Chatbook API, or
parent-DOM access. Trusted Chatbook code and user-confirmed Canvas submit or
download actions sit outside that generated runtime. See the
[Canvas user guide](../../Docs/User_Guide/console/canvas.md) and
[V1 compatibility boundary](../../Docs/Canvas/V1_RUNTIME_COMPATIBILITY.md).

## Binary Distribution

When packaged as a binary (future feature), users will be able to:

1. Download a single binary file
2. Run it with `--serve` to launch in web mode
3. Access the TUI through their browser without installing Python or dependencies

## Troubleshooting

If you get an import error, ensure textual-serve is installed:

```bash
pip install textual-serve
```

To check if web server dependencies are available:

```python
from tldw_chatbook.Web_Server import WEB_SERVER_AVAILABLE
print(f"Web server available: {WEB_SERVER_AVAILABLE}")
```
