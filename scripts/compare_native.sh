#!/usr/bin/env bash
# Checks the native renderer (PS2_GPU_NATIVE, runtime/src/gs/gpu/native.wgsl) against the compute rasterizer: runs
# the synthetic GS scenes, the general ones and the race-like ones, through both on each backend and compares all GS
# memory and display output (see --against in tests/gs_gpu_compare_test.cpp), twice:
#
#   exact     with blending left to the compute rasterizer (PS2_NATIVE_SKIP=2), every byte must be the same: what
#             the native renderer draws without blending is the compute rasterizer's picture bit for bit, and so is
#             the hand-over between the two.
#   blending  with the native renderer blending too, race-like scenes only, every byte within 8 but for a few (16 a
#             scene, where a difference went through several textures and blends). Fixed-function
#             blending rounds where the GS truncates, one step per blend, and a texture function can double it.
#             Blending by a texture's alpha stays with the compute rasterizer (PS2_NATIVE_SKIP=128): the scenes'
#             frames, read back as textures, have alphas above 0x80, which the native renderer does not blend with.
#
# Needs what scripts/compare_gpu_backends.sh needs. No game data is involved.
#
#   scripts/compare_native.sh [scenes] [scale...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
scenes="${1:-100}"; shift || true
scales=("$@"); [ ${#scales[@]} -gt 0 ] || scales=(1 2)
for backend in webgpu gl; do
  cmake -S "$ROOT" -B "$ROOT/build/compare-$backend" -G Ninja -DCMAKE_BUILD_TYPE=Release -DPS2_BUILD_TESTS=ON -DPS2_GPU_BACKEND=$backend > /dev/null
  ninja -C "$ROOT/build/compare-$backend" gs_gpu_compare_test > /dev/null
done
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
status=0
for scale in "${scales[@]}"; do
  for backend in webgpu gl; do
    for kind in "" --race; do
      test="$ROOT/build/compare-$backend/gs_gpu_compare_test"
      PS2_SCALE=$scale PS2_GPU_NATIVE=0 PS2_COMPARE_MIN_PERCENT=0 "$test" "$scenes" $kind --dump "$tmp/compute.bin" > "$tmp/compute.log" 2>&1 \
        || { echo "$backend failed:"; tail -5 "$tmp/compute.log"; exit 1; }
      for check in exact blending; do
        if [ $check = exact ]; then skip=2; options="--tolerance 0 --allow 0"; else skip=128; options="--tolerance 8 --allow 16"; [ -n "$kind" ] || continue; fi
        if PS2_SCALE=$scale PS2_GPU_NATIVE=1 PS2_NATIVE_SKIP=$skip "$test" "$scenes" $kind --against "$tmp/compute.bin" $options > "$tmp/native.log" 2> "$tmp/native.err"; then verdict=ok; else verdict=DIFFERS; status=1; fi
        grep -q "native renderer on" "$tmp/native.err" || { echo "$backend: the native renderer did not start"; tail -5 "$tmp/native.err"; exit 1; }
        echo "scale $scale, $backend, ${kind:-general} scenes, $check: $verdict"
        cat "$tmp/native.log" "$tmp/native.err" | grep -E "^against |native: " | sed "s#$tmp/compute.bin#the compute rasterizer#; s/^/    /"
        grep -E "^scene +[0-9]+: " "$tmp/native.log" | head -3 | sed "s#$tmp/compute.bin#the compute rasterizer#; s/^/    /" || true
      done
    done
  done
done
exit $status
