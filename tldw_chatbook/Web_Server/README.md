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
tldw-cli --serve --host 0.0.0.0 --port 9000 --web-title "My TUI App"
```

### Method 2: Using the dedicated tldw-serve command

```bash
tldw-serve
```

With options:
```bash
tldw-serve --host 0.0.0.0 --port 9000 --title "My TUI App" --debug
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
```

`font_size` controls the browser terminal cell density. The default `12` keeps
the web UI close to native terminal screenshots; use `?fontsize=16` in the URL
or set `font_size = 16` if you prefer larger text.

## Security Considerations

- By default, the server binds to `localhost` which only allows local connections
- To allow external connections, use `--host 0.0.0.0` but ensure proper firewall rules
- The web server runs the Textual app in a subprocess with restricted permissions
- No shell access is exposed through the web interface

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
