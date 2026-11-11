// The native renderer: GS triangles and sprites drawn by the GPU's own rasterizer into textures that stand for a
// frame buffer and a Z buffer, instead of by the compute shader in gs.wgsl. This file is appended to gs.wgsl (after an
// "enable dual_source_blending;" line) and shares its texture, CLUT and address code: textures are still read
// straight from GS memory.
//
// A draw's parameters: words 0..24 are the setup, as in the data buffer; 32 is the word offset of its first primitive
// record in the data buffer, 33 the kind of primitive (0 triangles, 1 sprites), 34 and 35 the target's size in pixels,
// 36 how the shader scales the colour for blending (see native_fs).

fn param_setup() -> Setup {
    var s: Setup;
    s.flags = prm(0u);
    s.fbp = prm(1u);
    s.fbw = prm(2u);
    s.fpsm = prm(3u);
    s.fbmsk = prm(4u);
    s.zbp = prm(5u);
    s.zpsm = prm(6u);
    s.modes = prm(7u);
    s.aref = prm(8u);
    s.fix = prm(9u);
    s.scax0 = i32(prm(10u) & 0xFFFFu);
    s.scax1 = i32(prm(10u) >> 16u);
    s.scay0 = i32(prm(11u) & 0xFFFFu);
    s.scay1 = i32(prm(11u) >> 16u);
    s.tbp = prm(12u);
    s.tbw = prm(13u);
    s.tpsm = prm(14u);
    s.tw = prm(15u) & 0xFFu;
    s.th = prm(15u) >> 8u;
    s.cpsm = prm(16u);
    s.csa = prm(17u);
    s.minu = prm(18u) & 0xFFFFu;
    s.maxu = prm(18u) >> 16u;
    s.minv = prm(19u) & 0xFFFFu;
    s.maxv = prm(19u) >> 16u;
    s.ta0 = prm(20u) & 0xFFu;
    s.ta1 = (prm(20u) >> 8u) & 0xFFu;
    s.fogcol = prm(21u);
    s.clut = prm(22u);
    s.zmax = prm(23u);
    s.dec = prm(24u);
    s.tf = fmt_of(s.tpsm, s.tbp, s.tbw);
    s.ff = fmt_of(s.fpsm, s.fbp, s.fbw);
    s.zf = fmt_of(s.zpsm, s.zbp, s.fbw);
    return s;
}

// The rasterizer only decides which pixels a primitive may cover. What it gives a pixel is worked out in the fragment
// stage with the compute rasterizer's own arithmetic (triangle_at and sprite_at), from the primitive's values handed
// over unchanged: colours, texture coordinates and depth then agree with the compute rasterizer bit for bit, so the
// two can draw the same scene in turns without their depths fighting or texels shifting.
struct NativeVertex {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) pa: vec4<f32>,
    @location(1) @interpolate(flat) pb: vec4<f32>,
    @location(2) @interpolate(flat) c0: vec4<f32>,
    @location(3) @interpolate(flat) c1: vec4<f32>,
    @location(4) @interpolate(flat) c2: vec4<f32>,
    @location(5) @interpolate(flat) t0: vec4<f32>,
    @location(6) @interpolate(flat) t1: vec4<f32>,
    @location(7) @interpolate(flat) t2: vec4<f32>,
    @location(8) @interpolate(flat) dz: vec2<f32>,
    @location(9) @interpolate(flat) zi: vec2<u32>,
}

const EXPAND: f32 = 0.125;

// Vertices come from the primitive records the compute rasterizer uses too, picked by vertex index: three per
// triangle, six per sprite. The GS samples a pixel at its integer coordinate, the GPU at its centre, hence the half.
@vertex
fn native_vs(@builtin(vertex_index) index: u32) -> NativeVertex {
    var out: NativeVertex;
    var x: f32;
    var y: f32;
    var z: f32;
    if (prm(33u) == 0u) {
        let p = prm(32u) + (index / 3u) * PRIM_WORDS;
        let n = index % 3u;
        out.pa = vec4<f32>(fat(p + 4u), fat(p + 5u), fat(p + 6u), fat(p + 7u));
        out.pb = vec4<f32>(fat(p + 8u), fat(p + 9u), fat(p + 10u), 0.0);
        // The triangle is drawn a little larger than it is, each edge moved out by EXPAND pixels, so that every
        // pixel the compute rasterizer's edge arithmetic counts as inside gets a fragment: its floats and the GPU's
        // fixed point do not agree on pixels that lie on an edge or within a rounding error of one.
        var corner: array<vec2<f32>, 3>;
        corner[0] = out.pa.xy;
        corner[1] = out.pa.zw;
        corner[2] = out.pb.xy;
        let here = corner[n];
        let before = normalize(here - corner[(n + 2u) % 3u]);
        let after = normalize(corner[(n + 1u) % 3u] - here);
        let moved = here + (before - after) * (EXPAND / max(abs(before.x * after.y - before.y * after.x), 0.0001));
        x = moved.x;
        y = moved.y;
        // The depth there, on the triangle's plane, kept inside the view volume.
        let w0 = ((out.pa.z - x) * (out.pb.y - y) - (out.pb.x - x) * (out.pa.w - y)) * out.pb.z;
        let w1 = ((out.pb.x - x) * (out.pa.y - y) - (out.pa.x - x) * (out.pb.y - y)) * out.pb.z;
        z = clamp(f32(data[p + 24u]) + fat(p + 25u) * w1 + fat(p + 26u) * (1.0 - w0 - w1), 0.0, 16777215.0);
        out.c0 = vertex_attr(p, 0u);
        out.c1 = vertex_attr(p, 1u);
        out.c2 = vertex_attr(p, 2u);
        out.t0 = vertex_tex(p, 0u);
        out.t1 = vertex_tex(p, 1u);
        out.t2 = vertex_tex(p, 2u);
        out.dz = vec2<f32>(fat(p + 25u), fat(p + 26u));
        out.zi = vec2<u32>(data[p + 24u], data[p + 27u]);
    } else {
        let p = prm(32u) + (index / 6u) * PRIM_WORDS;
        let n = index % 6u;
        out.pa = vec4<f32>(fat(p + 4u), fat(p + 5u), fat(p + 6u), fat(p + 7u));
        // A sprite covers whole pixels, every sample of them: those whose coordinate is inside it.
        let low = ceil(min(out.pa.xy, out.pa.xy + out.pa.zw));
        let high = ceil(max(out.pa.xy, out.pa.xy + out.pa.zw));
        x = select(low.x, high.x, n == 1u || n == 4u || n == 5u);
        y = select(low.y, high.y, n == 2u || n == 3u || n == 5u);
        z = f32(data[p + 24u]);
        out.pb = vec4<f32>(0.0);
        out.c0 = vertex_attr(p, 1u);
        out.c1 = out.c0;
        out.c2 = out.c0;
        out.t0 = vertex_tex(p, 0u);
        out.t1 = vertex_tex(p, 1u);
        out.t2 = out.t1;
        out.dz = vec2<f32>(0.0);
        out.zi = vec2<u32>(data[p + 24u], 0u);
    }
    if ((prm(0u) & F_ZTE) == 0u) {
        z = 0.0;  // no depth test: any Z is inside the view volume
    }
    let scale = f32(SCALE);
    out.position = vec4<f32>((x * scale + 0.5) / (f32(prm(34u)) * scale) * 2.0 - 1.0, 1.0 - (y * scale + 0.5) / (f32(prm(35u)) * scale) * 2.0, z / 16777216.0, 1.0);
    return out;
}

struct NativeColor {
    @location(0) @blend_src(0) color: vec4<f32>,
    // The source alpha as a blend factor: the GS's 0x80 is 1.0. The frame buffer keeps the alpha itself.
    @location(0) @blend_src(1) factor: vec4<f32>,
}

struct NativeShaded {
    color: NativeColor,
    z: u32,
}

// Depth test, blending and the channel mask are the pipeline's; only an alpha test that discards is here.
fn native_shade(in: NativeVertex) -> NativeShaded {
    let s = param_setup();
    let sx = u32(in.position.x);
    let sy = u32(in.position.y);
    let x = i32(sx / SCALE);
    let y = i32(sy / SCALE);
    let px = f32(x) + f32(sx % SCALE) / f32(SCALE);
    let py = f32(y) + f32(sy % SCALE) / f32(SCALE);
    var hit: Covered;
    if (prm(33u) == 0u) {
        hit = triangle_at(in.pa, in.pb, in.zi.y, in.c0, in.c1, in.c2, in.t0, in.t1, in.t2, in.zi.x, in.dz, px, py);
    } else {
        hit = sprite_at(s, in.pa, in.t0, in.t1, in.c0, in.zi.x, x, y, px, py);
    }
    if (!hit.inside) {
        discard;  // the two rasterizers disagree about an edge: the compute one's word counts
    }
    let c = shade_color(s, hit.at, (sy % SCALE) * SCALE + sx % SCALE);
    if (has(s, F_ATE) && !alpha_test(s, c.w)) {
        discard;
    }
    var a = c.w & 0xFF;
    if (has(s, F_FBA)) {
        a = a | 0x80;
    }
    // A blend equation where C scales only the source colour, (Cs - 0) * C or (0 - Cs) * C: the product is made here,
    // rounded as the GS rounds it, and the blender adds it to or takes it from the destination.
    var rgb = vec3<i32>(c.x, c.y, c.z) & vec3<i32>(0xFF);
    let scaled = prm(36u);
    if ((scaled & 1u) != 0u) {
        var factor = c.w;
        if ((scaled & 2u) != 0u) {
            factor = i32(s.fix);
        }
        rgb = rgb * factor;
        if ((scaled & 4u) != 0u) {
            rgb = rgb + vec3<i32>(127);
        }
        rgb = min(rgb >> vec3<u32>(7u), vec3<i32>(255));
    }
    var out: NativeShaded;
    out.color.color = vec4<f32>(f32(rgb.x), f32(rgb.y), f32(rgb.z), f32(a)) / 255.0;
    out.color.factor = vec4<f32>(f32(c.w) / 128.0);
    out.z = min(hit.at.z, s.zmax);
    return out;
}

// With the depth the rasterizer interpolates: the GPU can reject hidden pixels before shading them, but a depth can
// be one step off the compute rasterizer's (PS2_NATIVE_Z=fast, and every draw without a depth test).
@fragment
fn native_fs(in: NativeVertex) -> NativeColor {
    return native_shade(in).color;
}

struct NativeDepth {
    @location(0) @blend_src(0) color: vec4<f32>,
    @location(0) @blend_src(1) factor: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

// With the compute rasterizer's own depth, exact in a 32-bit float.
@fragment
fn native_depth_fs(in: NativeVertex) -> NativeDepth {
    let shaded = native_shade(in);
    return NativeDepth(shaded.color.color, shaded.color.factor, f32(shaded.z) / 16777216.0);
}

// GS memory into a target's texture, over the scissor rectangle. Parameters: block pointer, buffer width, and a
// depth buffer's format.
@vertex
fn load_vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32((index << 1u) & 2u) * 2.0 - 1.0, f32(index & 2u) * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn load_color_fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let sx = u32(position.x);
    let sy = u32(position.y);
    let v = load_at(addr(fmt_of(PSMCT32, prm(0u), prm(1u)), sx / SCALE, sy / SCALE), (sy % SCALE) * SCALE + sx % SCALE);
    return vec4<f32>(f32(v & 0xFFu), f32((v >> 8u) & 0xFFu), f32((v >> 16u) & 0xFFu), f32(v >> 24u)) / 255.0;
}

@fragment
fn load_depth_fs(@builtin(position) position: vec4<f32>) -> @builtin(frag_depth) f32 {
    let sx = u32(position.x);
    let sy = u32(position.y);
    let v = load_at(addr(fmt_of(prm(2u), prm(0u), prm(1u)), sx / SCALE, sy / SCALE), (sy % SCALE) * SCALE + sx % SCALE);
    // A Z32 buffer can hold more than a float does exactly, and than any native draw's depth: such a pixel is 1.0
    // here, which no draw's depth reaches, and store_depth leaves GS memory's value alone where it still is.
    return select(f32(v) / 16777216.0, 1.0, v > 0xFFFFFFu);
}
