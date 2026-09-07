#!/usr/bin/env bash
# Run every derived-artifact guard the CI job runs, locally, in one command.
#
# TASK-19572. These checks are the only ones that have reliably caught real
# drift, and until now the burn-down workflow was "remember several separate
# commands" -- which is why the same diagnostic-inventory drift was
# rediscovered by hand in four separate tasks. This is the same list, in the
# same order, as .github/workflows/derived-artifacts.yml.
#
# TASK-20971 added the fifth: VALID_TABLES['chachanotes'] went stale, was
# repaired, and went stale again fourteen and a half hours later when the next
# migration added two tables. Its runtime pin (Tests/DB/test_sql_validation.py)
# was correct both times and reported both times -- after the merge. This is
# the same class of problem TASK-19572 built this file for: a guard that only
# lives in a suite nobody runs locally is not an authoring-time guard.
#
# Deliberately NOT a git hook: at ~33 s a pre-commit hook is punitive at this
# repo's commit rate, is bypassable with `git commit -n`, and does not survive
# a fresh clone. Install it as one if you personally want to (`ln -s
# ../../scripts/preflight.sh .git/hooks/pre-commit`), but the gate that counts
# is the CI job.
#
# Usage:  ./scripts/preflight.sh          # picks an interpreter at the repo floor
#         PYTHON=.venv/bin/python ./scripts/preflight.sh
#
# Exits 0 when every check passes, 1 otherwise. Every check runs even after one
# fails, so a single pass reports all of the drift.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# "stdlib-only" means the PROJECT's stdlib floor (pyproject: requires-python
# >=3.11), not whatever `python3` happens to be. On macOS `python3` is the
# system 3.9, and under it three of the five checks below die on unrelated
# tracebacks: `list[str] | None` evaluated at runtime (check_bundle_sync.py),
# `from enum import StrEnum` (check_profile_owned_path_inventory.py), and
# `ast.parse` of a source using `except*` (check_persistent_diagnostic_
# inventory.py). Defaulting to a below-floor interpreter therefore made the
# obvious invocation report three FAILED checks that have nothing to do with
# the author's change -- and a guard that cries wolf is a guard that gets
# muted, which is the exact failure this file exists to prevent. So pick an
# interpreter that meets the floor, and say so plainly when none is reachable.
PYTHON_FLOOR_MAJOR=3
PYTHON_FLOOR_MINOR=11

meets_python_floor() {
  command -v "$1" >/dev/null 2>&1 || [ -x "$1" ] || return 1
  "$1" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (${PYTHON_FLOOR_MAJOR}, ${PYTHON_FLOOR_MINOR}) else 1)" \
    >/dev/null 2>&1
}

if [ -n "${PYTHON:-}" ]; then
  # An explicit choice is honoured, but verified: failing here with one line
  # beats failing later with three tracebacks.
  if ! meets_python_floor "$PYTHON"; then
    echo "preflight: PYTHON=$PYTHON is missing or below this repo's Python" \
      "${PYTHON_FLOOR_MAJOR}.${PYTHON_FLOOR_MINOR} floor; three of the five checks cannot run under it." >&2
    exit 1
  fi
else
  # Order matters: the repo's own venv, then whatever the developer has
  # activated, and only then a versioned interpreter off PATH. Preferring the
  # newest available would silently run the checks under an interpreter the
  # project does not otherwise use.
  for candidate in \
    "$REPO_ROOT/.venv/bin/python" \
    python3 python3.14 python3.13 python3.12 python3.11 python
  do
    if meets_python_floor "$candidate"; then
      PYTHON="$candidate"
      break
    fi
  done
  if [ -z "${PYTHON:-}" ]; then
    echo "preflight: no Python >= ${PYTHON_FLOOR_MAJOR}.${PYTHON_FLOOR_MINOR} found" \
      "(tried $REPO_ROOT/.venv/bin/python, python3.14 .. python3.11, python3, python)." >&2
    echo "preflight: re-run as  PYTHON=/path/to/python3.11-or-newer ./scripts/preflight.sh" >&2
    exit 1
  fi
fi

echo "preflight: using $PYTHON ($("$PYTHON" -c 'import sys; print(sys.version.split()[0])'))"

failed=()

run_check() {
  local label="$1"
  shift
  echo
  echo "=== ${label} ==="
  if "$@"; then
    return 0
  fi
  failed+=("$label")
  return 1
}

run_check "generated stylesheets" \
  "$PYTHON" tldw_chatbook/css/check_bundle_sync.py
run_check "profile-owned path census" \
  "$PYTHON" scripts/check_profile_owned_path_inventory.py
run_check "production diagnostic inventory" \
  "$PYTHON" scripts/check_persistent_diagnostic_inventory.py
run_check "backlog task ids" \
  "$PYTHON" scripts/check_backlog_task_ids.py
run_check "chachanotes table allowlist" \
  "$PYTHON" scripts/check_schema_table_allowlist.py
run_check "index plan pins" \
  "$PYTHON" scripts/check_index_plan_pins.py

echo
if [ ${#failed[@]} -eq 0 ]; then
  echo "preflight: all derived-artifact checks passed."
  exit 0
fi
echo "preflight: ${#failed[@]} check(s) FAILED:"
for label in "${failed[@]}"; do
  echo "  - ${label}"
done
echo "Read the report above before regenerating anything."
exit 1
