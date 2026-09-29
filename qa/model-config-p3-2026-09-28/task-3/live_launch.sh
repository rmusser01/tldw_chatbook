#!/bin/bash
# Launch the worktree app in tmux on a fully scratch profile. Usage: launch.sh <cols> <rows>
set -e
W=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/model-config-p3
L=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/73fb7a69-fb3c-49ea-81ff-f711a75d6336/scratchpad/t3/live
cols=$1; rows=$2
P=$L/prof
rm -rf $P; mkdir -p $P/home $P/xdg-data $P/xdg-config $P/xdg-cache
cat > $P/config.toml <<TOML
[general]
users_name = "t33003p3_verify"
default_tab = "chat"

[first_run]
setup_started = true
setup_completed = true

[model_catalog]
auto_refresh_enabled = false

[chat_defaults]
provider = "llama_cpp"
model = "model-a"

[api_settings.llama_cpp]
api_url = "http://127.0.0.1:9099"
model = "model-a"
TOML
tmux -L t33003p3 kill-server 2>/dev/null || true
tmux -L t33003p3 new-session -d -x $cols -y $rows -c $W \
  "env -i PATH=/usr/bin:/bin TERM=xterm-256color COLORTERM=truecolor LANG=en_US.UTF-8 \
   HOME=$P/home XDG_DATA_HOME=$P/xdg-data XDG_CONFIG_HOME=$P/xdg-config XDG_CACHE_HOME=$P/xdg-cache \
   TLDW_CONFIG_PATH=$P/config.toml PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
   PYTHONPATH=$W /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m tldw_chatbook.app; sleep 600"
