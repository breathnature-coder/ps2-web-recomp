#pragma once

#include <cstddef>
#include <cstdint>

namespace ps2::gs::gpu::gl {

// GLSL translated from gs.wgsl (one per compute entry point and internal
// resolution scale, and for raster per tile size; tile 0 fits any),
// native.wgsl (per scale: a render pipeline's fragment shader under its entry
// point's name, its vertex shader under that name with "@vs" after it) and
// blit.wgsl (scale 0), see scripts/gen_glsl.py.
struct Shader {
    const char *entry;
    std::uint32_t scale, tile;
    const char *source;
};

extern const Shader kShaders[];
extern const std::size_t kShaderCount;

}
