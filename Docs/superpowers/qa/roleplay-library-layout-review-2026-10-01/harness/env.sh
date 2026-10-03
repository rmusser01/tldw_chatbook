# shellcheck shell=bash
# Common environment for the rp-review harness. SOURCE it (". env.sh"); never execute it.
# Works with macOS /bin/bash 3.2.
#
# Inputs (all optional):
#   HARNESS_STATE  where profiles, runs, captures, mock logs and the pycache live.
#                  Default: <main checkout>/.worktrees/.rp-harness-state  (git-ignored, outlives
#                  slice worktrees). <main checkout> = parent of `git rev-parse --git-common-dir`.
#                  Refused when it overlaps the real profile or sits in a tracked part of a checkout.
#   APP_WT         checkout whose tldw_chatbook code runs (PYTHONPATH). Default: the toplevel of
#                  the checkout that holds this harness.
#   PY             interpreter. Default: <main checkout>/.venv/bin/python, else $APP_WT/.venv/bin/python.
#   EXTRA_PYTHONPATH  appended after APP_WT (e.g. a dir holding an instrumentation sitecustomize.py).
# Outputs: HARNESS_DIR, RP_MAIN_CHECKOUT, HARNESS_STATE, APP_WT, PY (exported), and rp_app_env.

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
rp_die() { echo "rp-harness: $*" >&2; exit 2; }

_rp_common="$(git -C "$HARNESS_DIR" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" \
  || rp_die "the harness is not inside a git checkout (git >= 2.31 needed for --path-format)"
RP_MAIN_CHECKOUT="$(cd "$_rp_common/.." && pwd -P)"
unset _rp_common

if [ -z "${APP_WT:-}" ]; then APP_WT="$(git -C "$HARNESS_DIR" rev-parse --show-toplevel)"; fi
[ -d "$APP_WT" ] || rp_die "APP_WT=$APP_WT does not exist"
APP_WT="$(cd "$APP_WT" && pwd -P)"
[ -f "$APP_WT/tldw_chatbook/app.py" ] || rp_die "APP_WT=$APP_WT has no tldw_chatbook/app.py"

if [ -z "${PY:-}" ]; then
  for _rp_py in "$RP_MAIN_CHECKOUT/.venv/bin/python" "$APP_WT/.venv/bin/python"; do
    if [ -x "$_rp_py" ]; then PY="$_rp_py"; break; fi
  done
  unset _rp_py
fi
[ -n "${PY:-}" ] && [ -x "$PY" ] || rp_die "no python interpreter found; set PY=/path/to/venv/bin/python"

# Validate (and default) HARNESS_STATE in one place: harness_guard.py.
HARNESS_STATE="$(HARNESS_STATE="${HARNESS_STATE:-}" "$PY" -B "$HARNESS_DIR/harness_guard.py" state)" \
  || rp_die "HARNESS_STATE rejected (see REFUSING above)"
mkdir -p "$HARNESS_STATE"
chmod 700 "$HARNESS_STATE" 2>/dev/null || true

export HARNESS_DIR RP_MAIN_CHECKOUT HARNESS_STATE APP_WT PY

# rp_app_env <profile-dir>: sets the array RP_ENV to an `env -i ...` prefix that runs a
# command with the profile's disposable HOME / XDG_* / TLDW_CONFIG_PATH, the app code on
# PYTHONPATH, bytecode redirected into HARNESS_STATE (nothing is written into APP_WT), and no
# inherited provider keys (the environment is cleared; only an allowlist is passed through).
rp_app_env() {
  local p="$1" pp="$APP_WT"
  if [ -n "${EXTRA_PYTHONPATH:-}" ]; then pp="$pp:$EXTRA_PYTHONPATH"; fi
  RP_ENV=(env -i
    "PATH=$PATH"
    "USER=${USER:-$(id -un)}" "LOGNAME=${LOGNAME:-$(id -un)}"
    "LANG=${LANG:-en_US.UTF-8}"
    "TMPDIR=${TMPDIR:-/tmp}"
    "TERM=xterm-256color" "COLORTERM=truecolor"
    "HOME=$p/home"
    "XDG_CONFIG_HOME=$p/xdg_config" "XDG_DATA_HOME=$p/xdg_data"
    "XDG_CACHE_HOME=$p/xdg_cache" "XDG_STATE_HOME=$p/xdg_state"
    "TLDW_CONFIG_PATH=$p/config.toml"
    "PYTHONPATH=$pp"
    "PYTHONPYCACHEPREFIX=$HARNESS_STATE/pycache"
    "HARNESS_STATE=$HARNESS_STATE" "APP_WT=$APP_WT")
  # The OS keychain is per-user state outside HOME: without this the app's personal-context
  # bootstrap reads/creates items in the REAL login keychain (and, headless, can park for minutes,
  # stalling the first Console send ~10 s and Ctrl+Q for the full 120 s shutdown grace).
  # Same approach as Backup_Recovery/isolated_restore.py. RP_KEYRING_BACKEND="" = OS keychain.
  if [ -n "${RP_KEYRING_BACKEND-keyring.backends.null.Keyring}" ]; then
    RP_ENV+=("PYTHON_KEYRING_BACKEND=${RP_KEYRING_BACKEND-keyring.backends.null.Keyring}")
  fi
  if [ -n "${LC_ALL:-}" ]; then RP_ENV+=("LC_ALL=$LC_ALL"); fi
  if [ -n "${LC_CTYPE:-}" ]; then RP_ENV+=("LC_CTYPE=$LC_CTYPE"); fi
}

# rp_under_state <path>: print the absolute path when it is STRICTLY inside HARNESS_STATE
# (checked lexically before anything is created, then again after resolving symlinks);
# otherwise die. Creates the directory.
rp_under_state() {
  local p="$1" r
  case "$p" in /*) ;; *) p="$PWD/$p" ;; esac
  case "/$p/" in */../*|*/./*) rp_die "REFUSING: $1 contains . or .. components" ;; esac
  case "$p" in "$HARNESS_STATE"/?*) ;; *) rp_die "REFUSING: $1 is not inside HARNESS_STATE=$HARNESS_STATE" ;; esac
  mkdir -p "$p"
  r="$(cd "$p" && pwd -P)"
  case "$r" in "$HARNESS_STATE"/?*) ;; *) rp_die "REFUSING: $1 resolves to $r, outside HARNESS_STATE" ;; esac
  printf '%s\n' "$r"
}
