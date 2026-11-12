#!/bin/zsh
unset TLDW_TEST_MODE
export PYTHON_KEYRING_BACKEND='keyring.backends.null.Keyring'
export HOME='/private/tmp/archive-followup-live-0tuqm_8b/home'
export XDG_CONFIG_HOME='/private/tmp/archive-followup-live-0tuqm_8b/xdg-config'
export XDG_DATA_HOME='/private/tmp/archive-followup-live-0tuqm_8b/xdg-data'
export XDG_CACHE_HOME='/private/tmp/archive-followup-live-0tuqm_8b/xdg-cache'
export TLDW_CONFIG_PATH='/private/tmp/archive-followup-live-0tuqm_8b/config.toml'
export PYTHONPATH='/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook'
export HF_HUB_OFFLINE='1'
export TRANSFORMERS_OFFLINE='1'
cd '/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook'
exec /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python '/private/tmp/archive-followup-live-0tuqm_8b/launch.py'
