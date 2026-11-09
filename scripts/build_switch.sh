#!/usr/bin/env bash
# Builds a game that scripts/port.sh has already translated into a Nintendo
# Switch homebrew (.nro) and lays out the folder to copy to the SD card.
#
#   scripts/port.sh gt4 "Gran Turismo 4.iso" --native-only   # on the PC, once
#   scripts/build_switch.sh gt4 [--with-disc]
#
# Needs devkitPro with switch-dev and switch-sdl2 (dkp-pacman -S switch-dev
# switch-sdl2) and a devkitA64 whose GCC is version 15 or newer. Everything it
# produces contains the game's translated code: keep it on your own devices.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
  cat <<'EOF'
usage: scripts/build_switch.sh <name> [--with-disc] [--boost] [--log] [--title TEXT]

  <name>        the name given to scripts/port.sh; generated/<name> must exist
  --with-disc   also copy games/<name>/disc.iso into the SD folder (large; an
                SD card formatted FAT32 cannot hold files over 4 GiB, use exFAT)
  --boost       run the CPU at 1785 MHz while playing
  --log         write the runner's output to log.txt on the SD card
  --title TEXT  name shown in the homebrew menu
EOF
}

[ $# -ge 1 ] || { usage; exit 2; }
name="$1"; shift
with_disc=0; boost=0; log=0; title=""
while [ $# -gt 0 ]; do
  case "$1" in
    --with-disc) with_disc=1 ;;
    --boost) boost=1 ;;
    --log) log=1 ;;
    --title) title="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

game="$ROOT/games/$name"
generated="$ROOT/generated/$name"
out="$ROOT/build/switch-$name"
[ -d "$generated" ] && compgen -G "$generated/*.cpp" > /dev/null || {
  echo "generated/$name has no translated code; run scripts/port.sh $name <disc image> --native-only first" >&2; exit 1; }

: "${DEVKITPRO:=/opt/devkitpro}"
toolchain="$DEVKITPRO/cmake/Switch.cmake"
[ -f "$toolchain" ] || { echo "devkitPro not found at $DEVKITPRO (set DEVKITPRO, install switch-dev and switch-sdl2)" >&2; exit 1; }
export PATH="$DEVKITPRO/devkitA64/bin:$DEVKITPRO/tools/bin:$DEVKITPRO/portlibs/switch/bin:$PATH"
gcc_major=$(aarch64-none-elf-gcc -dumpversion | cut -d. -f1)
[ "$gcc_major" -ge 15 ] || { echo "devkitA64 GCC $gcc_major is too old, 15+ is needed for guaranteed tail calls (dkp-pacman -Syu)" >&2; exit 1; }

# The executable port.sh translated: a patched copy, GT4's unpacked core, or
# the boot program named in SYSTEM.CNF, in that order of preference.
elf=""
if compgen -G "$game/patched/*" > /dev/null; then
  elf=$(ls "$game"/patched/* | head -1)
elif [ -f "$game/core.elf" ]; then
  elf="$game/core.elf"
else
  boot=$(sed -n 's/^BOOT2 *= *//p' "$game/disc/SYSTEM.CNF" | tr -d '\r')
  program=${boot#*\\}; program=${program%;*}
  elf="$game/disc/$program"
fi
[ -f "$elf" ] || { echo "cannot find the translated executable for $name" >&2; exit 1; }

echo "== configure"
cmake_args=(-DCMAKE_BUILD_TYPE=Release -DPS2_GENERATED="$generated")
[ -n "$title" ] && cmake_args+=(-DPS2_NX_TITLE="$title")
cmake -S "$ROOT" -B "$out" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$toolchain" "${cmake_args[@]}" > "$out.configure.log" 2>&1 \
  || { tail -30 "$out.configure.log"; exit 1; }

echo "== build (the translated code takes a while)"
ninja -C "$out" > "$out/ninja.log" 2>&1 || { tail -40 "$out/ninja.log"; exit 1; }
# Older devkitPro CMake files keep the .nro step out of the default target.
compgen -G "$out/*.nro" > /dev/null || ninja -C "$out" ps2nx_nro >> "$out/ninja.log" 2>&1 || true
nro=$(ls "$out"/*.nro 2>/dev/null | head -1)
[ -n "$nro" ] || { echo "no .nro was produced, see ${out#$ROOT/}/ninja.log" >&2; exit 1; }

echo "== SD card folder"
sd="$out/sd/switch/ps2recomp"
rm -rf "$out/sd"; mkdir -p "$sd"
cp "$nro" "$sd/ps2recomp.nro"
cp "$elf" "$sd/$(basename "$elf")"
{
  echo "# arguments for ps2recomp.nro, one per line; see host/switch.cpp"
  echo "$(basename "$elf")"
  echo "--disc $name.iso"
  echo "--memcard $name.ps2"
  if [ -f "$game/patches.txt" ]; then cp "$game/patches.txt" "$sd/patches.txt"; echo "--patches patches.txt"; fi
  [ "$boost" = 1 ] && echo "@boost"
  [ "$log" = 1 ] && echo "@log"
} > "$sd/args.txt"
if [ "$with_disc" = 1 ]; then
  cp -L "$game/disc.iso" "$sd/$name.iso"
else
  echo "copy your disc image to the SD card as switch/ps2recomp/$name.iso"
fi
echo
echo "Done: copy the contents of ${out#$ROOT/}/sd to the root of the SD card and"
echo "open ps2recomp from the homebrew menu started over a game (hold R), not from the album."
