# shellcheck shell=bash
# Shared helpers for the rp-review drivers. Source me: . drive_lib.sh <socket> <size-suffix>
# Captures go to ${CAPTURES:-$HARNESS_STATE/captures}/<name>-<size>.{txt,ansi[,png]}.
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
R="$HARNESS_DIR"
S="$1"; SZ="$2"; C="${CAPTURES:-$HARNESS_STATE/captures}"
mkdir -p "$C"
shot(){ PNG=${PNG:-0} "$R/shot.sh" "$S" "$C/$1-$SZ" >/dev/null; echo "  shot $1-$SZ"; }
clk(){ "$PY" -B "$R/where.py" "$S" "$1" --click "${2:-1}" >/dev/null || echo "  !! no match: $1"; }
has(){ tmux -L "$S" capture-pane -p | grep -qF -- "$1"; }
wheel(){ # wheel <down|up> <col> <row> <n>
  local b=65; [ "$1" = up ] && b=64; for _ in $(seq 1 "$4"); do tmux -L "$S" send-keys -l $'\x1b[<'"$b;$2;$3"'M'; done; }
typ(){ tmux -L "$S" send-keys -l "$1"; }
key(){ tmux -L "$S" send-keys "$@"; }
pal(){ # pal "<query>"  -> run the single remaining palette match
  key C-p; sleep 1; typ "$1"; sleep 1.5; key Down; sleep 0.3; key Enter; }
rowof(){ "$PY" -B "$R/where.py" "$S" "$1" | head -1 | cut -d' ' -f1; }
colafter(){ # colafter <row> <anchor-text> <char>  -> 1-based col of first <char> after anchor on row
  tmux -L "$S" capture-pane -p | sed -n "${1}p" | "$PY" -B -c "import sys; l=sys.stdin.read(); i=l.find(sys.argv[1]); print(l.find(sys.argv[2], i+len(sys.argv[1]))+1 if i>=0 else 0)" "$2" "$3"; }
