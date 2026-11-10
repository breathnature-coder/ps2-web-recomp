#!/usr/bin/env bash
# Builds a game that scripts/port.sh has already translated into a Nintendo
# Switch homebrew (.nro) and lays out the folder to copy to the SD card.
#
#   scripts/port.sh gt4 "Gran Turismo 4.iso" --native-only   # on the PC, once
#   scripts/build_switch.sh gt4 [--with-disc]
#
# Needs devkitPro with switch-dev, switch-sdl2 and switch-mesa (dkp-pacman -S
# switch-dev switch-sdl2 switch-mesa) and a devkitA64 whose GCC is version 15 or newer. Everything it
# produces contains the game's translated code: keep it on your own devices.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
  cat <<'EOF'
usage: scripts/build_switch.sh <name> [--with-disc] [--log] [--stats] [--record] [--title TEXT]
                               [--docker [IMAGE]] [--gl-driver nvc0|zink]

  <name>        the name given to scripts/port.sh; generated/<name> must exist
  --with-disc   also copy games/<name>/disc.iso into the SD folder (large; an
                SD card formatted FAT32 cannot hold files over 4 GiB, use exFAT)
  (no clock option: the system's boost mode slows the GPU to 76.8 MHz; raise
   clocks with an overclocking tool such as sys-clk, or add @fastload to
   args.txt only to test whether a scene is CPU-bound)
  --log         write the runner's output to log.txt on the SD card
  --stats       also log where the GPU renderer's time goes, once a second
  --record      record vector unit programs and code the PC runs never reached
                (into vu1/ and misses.txt on the card); bring them back with
                scripts/import_switch_run.sh to have them compiled in
  --title TEXT  name shown in the homebrew menu
  --docker [IMAGE]  build inside a container that carries a newer Mesa for the
                Switch (default ghcr.io/danfromtico/switch-dev: Mesa 26 with
                nvc0 OpenGL, and Zink on the NVK Vulkan driver) instead of the
                local devkitPro with Mesa 20.1. Output goes to
                build/switch-<name>-mesa26. Experimental.
  --gl-driver D with --docker: which OpenGL driver the console uses, nvc0
                (default) or zink (OpenGL on Vulkan); it is one line in
                args.txt (@env MESA_SWITCH_GL_DRIVER=...), so it can also be
                changed on the card without rebuilding
EOF
}

[ $# -ge 1 ] || { usage; exit 2; }
name="$1"; shift
with_disc=0; log=0; stats=0; record=0; title=""; docker_image=""; gl_driver=""; gl="${PS2_SWITCH_GL:-legacy}"; suffix="${PS2_SWITCH_SUFFIX:-}"
args=("$@")
while [ $# -gt 0 ]; do
  case "$1" in
    --with-disc) with_disc=1 ;;
    --boost) echo "--boost is gone: the system boost mode slows the GPU to 76.8 MHz; use sys-clk or similar" >&2 ;;
    --log) log=1 ;;
    --stats) stats=1; log=1 ;;
    --record) record=1 ;;
    --title) title="$2"; shift ;;
    --docker) docker_image="ghcr.io/danfromtico/switch-dev"; if [ $# -gt 1 ] && [ "${2#--}" = "$2" ]; then docker_image="$2"; shift; fi ;;
    --gl-driver) gl_driver="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

if [ -n "$docker_image" ]; then
  # Run this same script inside the container, on the repository mounted at the same path.
  inner=()
  skip=0
  for a in "${args[@]}"; do
    if [ "$skip" = 1 ]; then skip=0; [ "${a#--}" = "$a" ] && continue; fi
    if [ "$a" = "--docker" ]; then skip=1; continue; fi
    inner+=("$a")
  done
  proxy=()
  for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do [ -n "${!v:-}" ] && proxy+=(-e "$v=${!v}"); done
  exec docker run --rm "${proxy[@]}" -e PS2_SWITCH_GL=pkgconfig -e PS2_SWITCH_SUFFIX=-mesa26 -u "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$ROOT:$ROOT" -w "$ROOT" "$docker_image" bash "$ROOT/scripts/build_switch.sh" "$name" "${inner[@]}"
fi

game="$ROOT/games/$name"
generated="$ROOT/generated/$name"
out="$ROOT/build/switch-$name$suffix"
[ -d "$generated" ] && compgen -G "$generated/*.cpp" > /dev/null || {
  echo "generated/$name has no translated code; run scripts/port.sh $name <disc image> --native-only first" >&2; exit 1; }

: "${DEVKITPRO:=/opt/devkitpro}"
toolchain="$DEVKITPRO/cmake/Switch.cmake"
[ -f "$toolchain" ] || { echo "devkitPro not found at $DEVKITPRO (set DEVKITPRO, install switch-dev and switch-sdl2)" >&2; exit 1; }
export PATH="$DEVKITPRO/devkitA64/bin:$DEVKITPRO/tools/bin:$DEVKITPRO/portlibs/switch/bin:$PATH"
gcc_major=$(aarch64-none-elf-gcc -dumpversion | cut -d. -f1)
[ "$gcc_major" -ge 15 ] || { echo "devkitA64 GCC $gcc_major is too old, 15+ is needed for guaranteed tail calls (dkp-pacman -Syu)" >&2; exit 1; }

# The OpenGL renderer is built from the WebGPU headers (not the library).
[ -f "$ROOT/third_party/wgpu-native/include/webgpu/webgpu.h" ] || {
  echo "the WebGPU headers are missing: run scripts/fetch_wgpu.sh (port.sh normally does)" >&2; exit 1; }

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
cmake_args=(-DCMAKE_BUILD_TYPE=Release -DPS2_GENERATED="$generated" -DPS2_SWITCH_GL="$gl")
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
  echo "# @env PS2_GS=cpu      software renderer, for comparison"
  echo "# @env PS2_SCALE=2     twice the PS2 resolution inside (slower)"
  if [ -f "$game/patches.txt" ]; then cp "$game/patches.txt" "$sd/patches.txt"; echo "--patches patches.txt"; fi
  [ "$log" = 1 ] && echo "@log"
  [ "$stats" = 1 ] && echo "@env PS2_GL_STATS=1"
  [ -n "$gl_driver" ] && echo "@env MESA_SWITCH_GL_DRIVER=$gl_driver"
  if [ "$record" = 1 ]; then echo "@env PS2_VU_RECORD=vu1"; echo "--misses misses.txt"; fi
} > "$sd/args.txt"
if [ "$with_disc" = 1 ]; then
  cp -L "$game/disc.iso" "$sd/$name.iso"
else
  echo "copy your disc image to the SD card as switch/ps2recomp/$name.iso"
fi
echo
echo "Done: copy the contents of ${out#$ROOT/}/sd to the root of the SD card and"
echo "open ps2recomp from the homebrew menu started over a game (hold R), not from the album."
