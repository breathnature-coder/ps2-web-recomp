#!/usr/bin/env bash
# Checks that the OpenGL renderer (runtime/src/gs/gpu/gl) draws exactly what
# the WebGPU one draws: builds the synthetic GS test both ways, runs the same
# scenes through each and compares all GS memory and display output byte for
# byte. Needs third_party/wgpu-native, Vulkan (lavapipe is fine) and EGL with
# OpenGL 4.3 (llvmpipe is fine). No game data is involved. On a machine without
# a Vulkan GPU, point VK_ICD_FILENAMES at lavapipe's ICD file.
#
#   scripts/compare_gpu_backends.sh [scenes] [scale...]
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
    PS2_SCALE=$scale PS2_COMPARE_MIN_PERCENT=0 "$ROOT/build/compare-$backend/gs_gpu_compare_test" "$scenes" --dump "$tmp/$backend.bin" > "$tmp/$backend.log" 2>&1 \
      || { echo "$backend failed:"; tail -5 "$tmp/$backend.log"; exit 1; }
  done
  if cmp -s "$tmp/webgpu.bin" "$tmp/gl.bin"; then
    echo "scale $scale: $scenes scenes, OpenGL output identical to WebGPU"
  else
    echo "scale $scale: OpenGL output DIFFERS from WebGPU ($(cmp "$tmp/webgpu.bin" "$tmp/gl.bin" | head -1))"
    status=1
  fi
done
exit $status
