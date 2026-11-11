#!/usr/bin/env python3
"""Translates the GS shaders from WGSL to GLSL for the OpenGL renderer.

The OpenGL renderer (runtime/src/gs/gpu/gl) runs the same compute shaders as
the WebGPU one, so instead of keeping a second copy by hand this asks naga, the
translator inside wgpu-native, for the GLSL it makes when wgpu itself runs on
OpenGL, and writes it into runtime/src/gs/gpu/gl/shaders.cpp. The native renderer's vertex and
fragment shaders (native.wgsl) go the same way. Run it again whenever gs.wgsl,
native.wgsl or blit.wgsl changes:

    scripts/gen_glsl.py            # regenerate
    scripts/gen_glsl.py --check    # fail if shaders.cpp is out of date

It needs third_party/wgpu-native (scripts/fetch_wgpu.sh), a C++ compiler and
an EGL/OpenGL driver; Mesa's software llvmpipe is enough.
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
gpu = root / 'runtime/src/gs/gpu'
out_path = gpu / 'gl/shaders.cpp'
wgpu = root / 'third_party/wgpu-native'
entries = ['upload', 'copy_read', 'copy_write', 'load_clut', 'display', 'raster', 'resolve', 'snapshot', 'decode', 'lut_frame', 'lut_scratch']
scales = [1, 2, 3]
tiles = [16, 8, 4]  # raster tile sizes (PS2_TILE); only the raster entry depends on it
stores = ['store_color', 'store_depth']  # compute entries that read a native target's texture
natives = ['native_fs', 'native_depth_fs', 'load_color_fs', 'load_depth_fs']  # render pipelines, by fragment entry


def build_tool(work: Path) -> Path:
    tool = work / 'wgsl_to_glsl'
    cxx = os.environ.get('CXX') or 'clang++'
    subprocess.run([cxx, '-std=c++20', '-O1', '-I', str(wgpu / 'include'), str(root / 'scripts/wgsl_to_glsl.cpp'),
                    str(wgpu / 'lib/libwgpu_native.a'), '-ldl', '-lpthread', '-lm', '-o', str(tool)], check=True)
    return tool


def shaders(tool: Path, wgsl: Path, args: list[str], tile: int = 16) -> dict[str, list[str]]:
    run = subprocess.run([str(tool), str(wgsl), *args], check=True, capture_output=True, text=True, env={**os.environ, 'WGSL_TILE': str(tile)})
    parts = re.split(r'@@ENTRY (\w+)\n', run.stdout)
    # parts: [before, name, logs, name, logs, ...]; a pipeline's GLSL is logged after its marker
    return {parts[i]: re.findall(r'Naga generated shader:\n(.*?)\n@@END', parts[i + 1], re.S) for i in range(1, len(parts), 2)}


def fix(src: str, stage: str) -> str:
    """Makes naga's desktop GLSL fit an OpenGL 4.3 core context with fixed bindings."""
    lines = []
    for line in src.splitlines():
        if line.startswith('#version'):
            line = '#version 430 core'
        elif line.startswith('#extension GL_ARB_compute_shader') or line.startswith('#extension GL_ARB_shader_storage_buffer_object'):
            continue  # core in 4.3
        elif stage == 'vs' and line.strip().startswith('gl_Position.yz = vec2(-gl_Position.y'):
            continue  # wgpu flips into its own texture; the renderer draws straight to the window
        elif stage == 'native_vs' and line.strip().startswith('gl_Position.yz = vec2(-gl_Position.y'):
            # Into a texture, whose first row is OpenGL's bottom one: flipped, as wgpu does. Depth stays as it is:
            # the renderer sets glClipControl to WebGPU's 0 to 1, so a Z24 value keeps all its bits.
            line = line[:len(line) - len(line.lstrip())] + 'gl_Position.y = -gl_Position.y;'
        # Bind every buffer at its WGSL binding number, so the renderer needs no table per pipeline.
        m = re.match(r'layout\((std430|std140), binding = \d+\)(.*\{ \w+ _group_0_binding_(\d+)_\w+(?:\[\])?; \};)$', line)
        if m:
            line = f'layout({m.group(1)}, binding = {m.group(3)}){m.group(2)}'
        lines.append(line)
    text = '\n'.join(lines) + '\n'
    if ')glsl"' in text:
        raise SystemExit('shader text contains the raw string delimiter')
    return text


def generate() -> str:
    if not (wgpu / 'lib/libwgpu_native.a').exists():
        raise SystemExit('third_party/wgpu-native is missing: run scripts/fetch_wgpu.sh')
    with tempfile.TemporaryDirectory(prefix='ps2-glsl-') as temp:
        tool = build_tool(Path(temp))
        table = []
        for scale in scales:
            found = shaders(tool, gpu / 'gs.wgsl', [str(scale), *entries])
            for entry in entries:
                if len(found.get(entry, [])) != 1:
                    raise SystemExit(f'naga produced no GLSL for {entry} at scale {scale}')
                table.append((entry, scale, 16 if entry == 'raster' else 0, fix(found[entry][0], 'cs')))
            for tile in tiles[1:]:
                found = shaders(tool, gpu / 'gs.wgsl', [str(scale), 'raster'], tile)
                if len(found.get('raster', [])) != 1:
                    raise SystemExit(f'naga produced no GLSL for raster at scale {scale}, tile {tile}')
                table.append(('raster', scale, tile, fix(found['raster'][0], 'cs')))
            found = shaders(tool, gpu / 'gs.wgsl', [str(scale), '--store'])
            for entry in stores:
                if len(found.get(entry, [])) != 1:
                    raise SystemExit(f'naga produced no GLSL for {entry} at scale {scale}')
                table.append((entry, scale, 0, fix(found[entry][0], 'cs')))
            # The native renderer's module: gs.wgsl and native.wgsl together, as gpu.cpp builds it.
            combined = Path(temp) / 'native.wgsl'
            combined.write_text('enable dual_source_blending;\n' + (gpu / 'gs.wgsl').read_text() + (gpu / 'native.wgsl').read_text())
            found = shaders(tool, combined, [str(scale), '--native'])
            for entry in natives:
                if len(found.get(entry, [])) != 2:
                    raise SystemExit(f'naga produced no GLSL for {entry} at scale {scale}')
                table.append((entry + '@vs', scale, 0, fix(found[entry][0], 'native_vs')))
                table.append((entry, scale, 0, fix(found[entry][1], 'fs')))
        blit = shaders(tool, gpu / 'blit.wgsl', ['1', '--blit']).get('blit', [])
        if len(blit) != 2:
            raise SystemExit('naga produced no GLSL for the blit pipeline')
        table.append(('blit_vs', 0, 0, fix(blit[0], 'vs')))
        table.append(('blit_fs', 0, 0, fix(blit[1], 'fs')))
    digest = hashlib.sha256(((gpu / 'gs.wgsl').read_bytes() + (gpu / 'native.wgsl').read_bytes() + (gpu / 'blit.wgsl').read_bytes())).hexdigest()[:16]
    body = ['// Generated by scripts/gen_glsl.py from gs.wgsl, native.wgsl and blit.wgsl; do not edit.',
            f'// Source digest: {digest}',
            '#include "shaders.hpp"', '', 'namespace ps2::gs::gpu::gl {', '',
            'const Shader kShaders[] = {']
    for entry, scale, tile, text in table:
        body.append(f'    {{"{entry}", {scale}u, {tile}u, R"glsl({text})glsl"}},')
    body += ['};', '', 'const std::size_t kShaderCount = sizeof(kShaders) / sizeof(kShaders[0]);', '', '}', '']
    return '\n'.join(body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    text = generate()
    if args.check:
        current = out_path.read_text() if out_path.exists() else ''
        if current != text:
            print(f'{out_path.relative_to(root)} is out of date: run scripts/gen_glsl.py', file=sys.stderr)
            return 1
        print('GLSL is up to date')
        return 0
    out_path.write_text(text)
    print(f'wrote {out_path.relative_to(root)} ({len(text) // 1024} KiB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
