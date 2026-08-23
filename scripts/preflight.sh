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
# Usage:  ./scripts/preflight.sh          # uses `python3` (stdlib only)
#         PYTHON=.venv/bin/python ./scripts/preflight.sh
#
# Exits 0 when every check passes, 1 otherwise. Every check runs even after one
# fails, so a single pass reports all of the drift.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PYTHON="${PYTHON:-python3}"

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
