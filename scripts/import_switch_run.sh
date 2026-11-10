#!/usr/bin/env bash
# Brings back what a Switch build made with --record found while playing:
# vector unit microprograms the PC exploring runs never saw (which the console
# otherwise interprets, slowly) and EE code addresses that were not translated.
# Adds them to the game, translates and compiles again, and rebuilds the .nro.
#
#   scripts/import_switch_run.sh <name> <disc image> <folder> [build_switch options]
#
# <folder> is switch/ps2recomp copied off the SD card (or the card itself,
# e.g. /mnt/e/switch/ps2recomp on WSL). Run it after each session that went
# somewhere new; the recordings add up.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ $# -ge 3 ] || { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
name="$1"; iso="$2"; from="$3"; shift 3
game="$ROOT/games/$name"
[ -d "$game" ] || { echo "games/$name not found: run scripts/port.sh $name first" >&2; exit 1; }
[ -d "$from" ] || { echo "$from not found" >&2; exit 1; }

programs=0
if compgen -G "$from/vu1/*.bin" > /dev/null; then
  mkdir -p "$game/vu1"
  for f in "$from"/vu1/*.bin; do
    [ -f "$game/vu1/$(basename "$f")" ] || { cp "$f" "$game/vu1/"; programs=$((programs + 1)); }
  done
fi

entries=0
if [ -s "$from/misses.txt" ]; then
  touch "$game/entries.txt"
  # Same normalisation as port.sh: [module:]address in lower-case hex.
  entries=$(python3 -I - "$game/entries.txt" "$from/misses.txt" <<'EOF'
import sys
entries, misses = sys.argv[1], sys.argv[2]
def token(t):
    name, _, at = t.rpartition(':')
    return (name + ':' if name else '') + '%x' % int(at, 16)
known = {token(t) for t in open(entries).read().split()}
found = [token(t) for t in open(misses).read().split()]
new = sorted({t for t in found if t not in known})
with open(entries, 'a') as out:
    for t in new:
        out.write(t + '\n')
print(len(new))
EOF
)
fi

echo "imported $programs new vector unit programs and $entries new code addresses"
if [ "$programs" = 0 ] && [ "$entries" = 0 ]; then
  echo "nothing new, the current build already has everything this session ran"
  exit 0
fi
"$ROOT/scripts/port.sh" "$name" "$iso" --no-explore --native-only
"$ROOT/scripts/build_switch.sh" "$name" "$@"
echo "copy ps2recomp.nro (and args.txt) to the card again; delete vu1/ and misses.txt there to start the next recording fresh"
