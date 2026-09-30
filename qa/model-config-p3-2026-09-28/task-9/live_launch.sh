#!/bin/bash
# TASK-33003.9 live capture. T9_SCRATCH = a scratch dir (profiles land in
# $T9_SCRATCH/prof-<label>); PYTHON = the venv interpreter. Launches a tree's
# app in tmux on a fully scratch profile (HOME/XDG/TLDW_CONFIG_PATH, null
# keyring). Usage: live_launch.sh <tree> <label> <cols> <rows>
set -e
S=${T9_SCRATCH:?export T9_SCRATCH}
W=$1; label=$2; cols=$3; rows=$4
P=$S/prof-$label
rm -rf $P; mkdir -p $P/home $P/xdg-data $P/xdg-config $P/xdg-cache
cat > $P/config.toml <<TOML
[general]
users_name = "t33003p9_${label}"
default_tab = "chat"

[splash_screen]
enabled = false

[first_run]
setup_started = true
setup_completed = true

[model_catalog]
auto_refresh_enabled = false

[chat_defaults]
provider = "llama_cpp"
model = "model-a"
temperature = 0.6

[api_settings.llama_cpp]
api_url = "http://127.0.0.1:9099"
model = "model-a"
TOML
tmux -L t33003p9 kill-server 2>/dev/null || true
tmux -L t33003p9 new-session -d -s s -x $cols -y $rows -c $W \
  "env -i PATH=/usr/bin:/bin TERM=xterm-256color COLORTERM=truecolor LANG=en_US.UTF-8 \
   HOME=$P/home XDG_DATA_HOME=$P/xdg-data XDG_CONFIG_HOME=$P/xdg-config XDG_CACHE_HOME=$P/xdg-cache \
   TLDW_CONFIG_PATH=$P/config.toml PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
   PYTHONPATH=$W ${PYTHON:?} -m tldw_chatbook.app; sleep 600"
