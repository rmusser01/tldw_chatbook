#!/bin/bash
# Review round 1: paths come from the environment. T6_SCRATCH = a scratch dir
# holding a `git archive <rev> | tar -x` tree per non-head label (base, pre);
# PYTHON = the venv interpreter; HEAD_TREE = the checkout run as "head". Copy
# live_*.sh to T6_SCRATCH/live without the prefix, and ansi_cells.py as cells.py.
# Launch a tree's app in tmux on a fully scratch profile. Usage: launch.sh <tree> <theme> <cols> <rows>
set -e
S=${T6_SCRATCH:?export T6_SCRATCH}
HEAD_TREE=$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || echo "${HEAD_TREE:?}")
tree=$1; theme=$2; cols=$3; rows=$4
case $tree in head) W=$HEAD_TREE;; *) W=$S/$tree;; esac
P=$S/live/prof-$tree-$theme
rm -rf $P; mkdir -p $P/home $P/xdg-data $P/xdg-config $P/xdg-cache
cat > $P/config.toml <<TOML
[general]
users_name = "t33003p6_${tree}_${theme//-/_}"
default_theme = "$theme"
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

[api_settings.llama_cpp]
api_url = "http://127.0.0.1:9099"
model = "model-a"
TOML
tmux -L t33003p6 kill-server 2>/dev/null || true
tmux -L t33003p6 new-session -d -x $cols -y $rows -c $W \
  "env -i PATH=/usr/bin:/bin TERM=xterm-256color COLORTERM=truecolor LANG=en_US.UTF-8 \
   HOME=$P/home XDG_DATA_HOME=$P/xdg-data XDG_CONFIG_HOME=$P/xdg-config XDG_CACHE_HOME=$P/xdg-cache \
   TLDW_CONFIG_PATH=$P/config.toml PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
   PYTHONPATH=$W ${PYTHON:?} -m tldw_chatbook.app; sleep 600"
