#include "gpu.hpp"
#include "ps2/system.hpp"
#include "ps2/video.hpp"

#include <webgpu/webgpu.h>
#ifndef __EMSCRIPTEN__
#include <webgpu/wgpu.h>
#endif

#include <algorithm>
#include <chrono>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <map>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>

extern const char kGsShader[];
extern const char kBlitShader[];
extern const char kNativeShader[];

#ifdef PS2_GPU_GL
// The OpenGL layer's (gl/webgpu_gl.cpp): a write to this buffer from its start replaces all of it, so the driver
// may hand out new storage instead of waiting for the GPU to be done with the old.
extern "C" void ps2GlDiscardOnWrite(WGPUBuffer buffer);
#endif

namespace ps2::gs {
void swizzle_tables(std::vector<std::uint32_t> &out);
}

namespace ps2::gs::gpu {

namespace {

// kNative is not a compute pipeline: such a command is a render pass of the native renderer (g_native_passes).
enum Pipeline { kUpload, kCopyRead, kCopyWrite, kLoadClut, kDisplay, kRaster, kResolve, kSnapshot, kDecode, kLutFrame, kLutScratch, kStoreColor, kStoreDepth, kPipelines, kNative = kPipelines };
const char *const kEntryPoints[kPipelines] = {"upload", "copy_read", "copy_write", "load_clut", "display", "raster", "resolve", "snapshot", "decode", "lut_frame", "lut_scratch", "store_color", "store_depth"};

constexpr std::uint32_t kParamStride = 256;
constexpr std::uint32_t kClutSlots = 2048;
constexpr std::uint32_t kClutEntries = 512;
constexpr std::uint32_t kPrimWords = 28;
constexpr std::uint32_t kSetupWords = 25;
constexpr std::uint32_t kDecodedWords = 4u << 20;
constexpr std::uint32_t kDecodePrims = 64;

Mode g_mode = Mode::off;
std::uint64_t g_dispatches[kPipelines] = {}, g_submits = 0;

WGPUInstance g_instance = nullptr;
WGPUAdapter g_adapter = nullptr;
WGPUSurface g_surface = nullptr;
WGPUTextureFormat g_surface_format = WGPUTextureFormat_Undefined;
WGPUPresentMode g_present_mode = WGPUPresentMode_Fifo;
std::uint32_t g_surface_width = 0, g_surface_height = 0;
WGPURenderPipeline g_blit = nullptr;
WGPUBindGroup g_blit_group = nullptr;
WGPUBuffer g_blit_params = nullptr;
std::uint32_t g_field = 0, g_last_width = 0, g_last_height = 0;
bool g_have_previous = false;
std::uint32_t g_scale = 1, g_copy_words = 1u << 20, g_field_words = 1u << 19;

std::uint32_t field_base(std::uint32_t field) { return g_copy_words + (field & 1u) * g_field_words; }

bool g_timestamps = false;
WGPUQuerySet g_queries = nullptr;
WGPUBuffer g_query_resolve = nullptr, g_query_read = nullptr;
bool g_query_pending = false, g_query_ready = false;
std::uint64_t g_measured_ns = 0, g_measured = 0, g_total_passes = 0;
WGPUDevice g_device = nullptr;
WGPUQueue g_queue = nullptr;
WGPUBuffer g_vram = nullptr, g_tables = nullptr, g_params = nullptr, g_data = nullptr, g_clut = nullptr, g_scratch = nullptr, g_readback = nullptr, g_decoded = nullptr;
std::uint32_t g_decoded_used = 0;
bool g_rendered[512] = {};
bool g_decode = true;
std::uint64_t g_params_size = 0, g_data_size = 0;
WGPUBindGroupLayout g_layout = nullptr;
WGPUPipelineLayout g_pipeline_layout = nullptr;
WGPUBindGroup g_group = nullptr;
WGPUComputePipeline g_pipelines[kPipelines] = {};

struct Surface;

struct Command {
    Pipeline pipeline;
    std::uint32_t x, y;
    std::uint32_t param;
    Surface *target = nullptr;  // the texture a store writes back
};

struct CommandInfo {
    std::size_t index;
    std::uint32_t values[8];
};
std::vector<CommandInfo> g_command_info;
std::FILE *g_profile_out = nullptr;
std::uint64_t g_profile_from = 0, g_profile_to = 0;
WGPUQuerySet g_profile_queries = nullptr;
WGPUBuffer g_profile_resolve = nullptr, g_profile_read = nullptr;

std::vector<Command> g_commands;
std::vector<std::array<std::uint32_t, kParamStride / 4>> g_param_records;
struct Arena {
    std::unique_ptr<std::uint32_t[]> words;
    std::size_t used = 0, capacity = 0;

    std::uint32_t *grow(std::size_t count) {
        if (used + count > capacity) {
            std::size_t next = std::max<std::size_t>(capacity * 2, 1u << 20);
            while (next < used + count) next *= 2;
            auto bigger = std::make_unique_for_overwrite<std::uint32_t[]>(next);
            if (used) std::memcpy(bigger.get(), words.get(), used * 4);
            words = std::move(bigger);
            capacity = next;
        }
        std::uint32_t *at = words.get() + used;
        used += count;
        return at;
    }
    std::size_t size() const { return used; }
    bool empty() const { return !used; }
    void clear() { used = 0; }
    std::uint32_t *data() { return words.get(); }
    std::uint32_t &operator[](std::size_t i) { return words[i]; }
} g_arena;
std::uint32_t g_clut_slot = 0, g_clut_loads = 0, g_texflushes = 0;
// What each CLUT slot was loaded from: block pointer, and whether as the 256 entries of a 32-bit CSM1 CLUT.
struct ClutSource {
    std::uint32_t cbp = 0;
    bool plain = false;
};
ClutSource g_clut_source[kClutSlots];

struct Image {
    bool active = false;
    std::uint32_t dbp = 0, dbw = 0, dpsm = 0, dsax = 0, dsay = 0, rrw = 0, total = 0, done = 0, bpp = 32, lead_bits = 0, id = 0;
    std::vector<std::uint8_t> pending;
} g_image;
std::uint32_t g_image_ids = 0;

struct Range {
    std::uint64_t lo = ~0ull, hi = 0;
    bool overlaps(std::uint64_t a, std::uint64_t b) const { return lo < b && a < hi; }
    void add(std::uint64_t a, std::uint64_t b) {
        lo = std::min(lo, a);
        hi = std::max(hi, b);
    }
};

struct Batch {
    std::vector<std::uint32_t> prims;
    std::vector<std::uint32_t> setups;
    std::vector<std::uint32_t> boxes;
    std::uint64_t key = ~0ull;
    int minx = 1 << 30, miny = 1 << 30, maxx = -1, maxy = -1;
    std::uint32_t fbp = 0, fbw = 0, fpsm = 0, zbp = 0, zpsm = 0;
    bool zte = false, zwrite = false;
    std::uint32_t texflushes = 0;
    int rx0 = 0, ry0 = 0, rx1 = -2, ry1 = -2;
    Range fb_range, z_range;
    bool native = false;
} g_batch;

struct SetupCache {
    std::uint32_t generation = ~0u, clut = ~0u;
    std::uint64_t attr = ~0ull;
    std::uint32_t setup[kSetupWords] = {};
    std::uint32_t flags = 0;
    std::uint64_t key = 0;
    Range tex;
    bool native = false, alpha_factor = false;
    std::uint32_t why = 0;
} g_setup_cache;

struct PreOp {
    bool clut = false;
    std::uint32_t rec[12] = {};
    Range range;
    std::uint32_t x0 = 0, y0 = 0, x1 = 0, y1 = 0, image = 0;
};

bool uploads_overlap(const PreOp &a, const PreOp &b) {
    if (a.image && a.image == b.image) return false;
    if (!a.range.overlaps(b.range.lo, b.range.hi)) return false;
    if (a.rec[0] != b.rec[0] || a.rec[1] != b.rec[1] || a.rec[2] != b.rec[2]) return true;
    return a.x0 <= b.x1 && b.x0 <= a.x1 && a.y0 <= b.y1 && b.y0 <= a.y1;
}

// Uploads and CLUT loads run before the pending primitive batch unless they touch memory it reads or writes.
std::vector<PreOp> g_pre;
std::vector<Range> g_batch_reads;

int g_tiles_per_row = 128;
// Side of a raster tile in pixels (PS2_TILE: 16, 8 or 4). Every pixel of a tile walks the tile's whole primitive list, so
// smaller tiles mean shorter walks at the price of more tiles to bin and dispatch.
int g_tile = 16;
std::vector<std::uint32_t> g_tile_count(128 * 128), g_tile_fill(128 * 128);
std::vector<std::uint32_t> g_touched;

WGPUStringView view(const char *s) { return WGPUStringView{s, WGPU_STRLEN}; }

std::uint32_t bits_per_pixel(std::uint32_t psm) {
    switch (psm) {
    case PSMCT24: case PSMZ24: return 24;
    case PSMCT16: case PSMCT16S: case PSMZ16: case PSMZ16S: return 16;
    case PSMT8: case PSMT8H: return 8;
    case PSMT4: case PSMT4HL: case PSMT4HH: return 4;
    default: return 32;
    }
}

void page_size(std::uint32_t psm, std::uint32_t &pw, std::uint32_t &ph) {
    switch (psm) {
    case PSMCT16: case PSMCT16S: case PSMZ16: case PSMZ16S: pw = 64; ph = 64; break;
    case PSMT8: pw = 128; ph = 64; break;
    case PSMT4: pw = 128; ph = 128; break;
    default: pw = 64; ph = 32; break;
    }
}

Range region(std::uint32_t bp, std::uint32_t bw, std::uint32_t psm, int x0, int y0, int x1, int y1) {
    std::uint32_t pw, ph;
    page_size(psm, pw, ph);
    const std::uint64_t per_row = std::max<std::uint32_t>(1, bw * 64 / pw);
    const auto page = [&](int x, int y) {
        return static_cast<std::uint64_t>(bp) * 256 + (static_cast<std::uint64_t>(std::max(x, 0)) / pw + static_cast<std::uint64_t>(std::max(y, 0)) / ph * per_row) * 8192;
    };
    Range r;
    r.add(std::min(page(x0, y0), static_cast<std::uint64_t>(bp) * 256), page(x1, y1) + 8192);
    return r;
}

std::uint32_t z_max(std::uint32_t psm) {
    switch (psm) {
    case PSMZ24: return 0xFFFFFFu;
    case PSMZ16: case PSMZ16S: return 0xFFFFu;
    default: return 0xFFFFFFFFu;
    }
}

std::uint32_t f2u(float f) {
    std::uint32_t v;
    std::memcpy(&v, &f, 4);
    return v;
}

std::uint32_t push_params(const std::initializer_list<std::uint32_t> values) {
    std::array<std::uint32_t, kParamStride / 4> record{};
    std::size_t i = 0;
    for (std::uint32_t v : values) record[i++] = v;
    g_param_records.push_back(record);
    return static_cast<std::uint32_t>(g_param_records.size() - 1);
}

void dispatch_linear(Pipeline p, std::uint32_t items, std::uint32_t param) {
    const std::uint32_t groups = (items + 63) / 64;
    if (!groups) return;
    g_commands.push_back({p, std::min(groups, 1024u), (groups + 1023) / 1024, param});
}

constexpr std::uint32_t kCacheWords = 4u << 20;
constexpr std::size_t kCacheMinBytes = 256;

struct CacheEntry {
    std::uint64_t key;
    std::uint32_t offset, words;
    std::size_t bytes;
    std::uint64_t used;
};

WGPUBuffer g_cache = nullptr;
std::vector<std::uint32_t> g_cache_copy;
std::deque<CacheEntry> g_cache_ring;
std::unordered_multimap<std::uint64_t, CacheEntry *> g_cache_index;
std::unordered_map<std::uint64_t, std::uint32_t> g_cache_seen;
std::uint32_t g_cache_head = 0;
std::uint64_t g_serial = 1;

std::uint64_t content_key(const std::uint8_t *p, std::size_t n) {
    std::uint64_t h = static_cast<std::uint64_t>(n) * 0x9E3779B97F4A7C15ull;
    const auto mix = [&h](std::uint64_t v) {
        h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
        h *= 0xBF58476D1CE4E5B9ull;
    };
    const std::size_t step = std::max<std::size_t>(8, (n / 64) & ~static_cast<std::size_t>(7));
    std::uint64_t v;
    for (std::size_t i = 0; i + 8 <= n; i += step) {
        std::memcpy(&v, p + i, 8);
        mix(v);
    }
    v = 0;
    std::memcpy(&v, p + n - std::min<std::size_t>(n, 8), std::min<std::size_t>(n, 8));
    mix(v);
    return h;
}

bool cache_find(std::uint64_t key, const std::uint8_t *bytes, std::size_t n, std::uint32_t &offset) {
    const auto [first, last] = g_cache_index.equal_range(key);
    for (auto it = first; it != last; ++it) {
        CacheEntry &e = *it->second;
        if (e.bytes != n || std::memcmp(g_cache_copy.data() + e.offset, bytes, n) != 0) continue;
        e.used = g_serial;
        offset = e.offset;
        return true;
    }
    return false;
}

bool cache_insert(std::uint64_t key, const std::uint8_t *bytes, std::size_t n, std::uint32_t &offset) {
    const auto words = static_cast<std::uint32_t>((n + 3) / 4 + 1);
    if (words > kCacheWords / 4) return false;
    const auto evict_front = [] {
        CacheEntry &front = g_cache_ring.front();
        if (front.used == g_serial) return false;
        const auto [first, last] = g_cache_index.equal_range(front.key);
        for (auto it = first; it != last; ++it)
            if (it->second == &front) {
                g_cache_index.erase(it);
                break;
            }
        g_cache_ring.pop_front();
        return true;
    };
    if (g_cache_head + words > kCacheWords) {
        while (!g_cache_ring.empty() && g_cache_ring.front().offset >= g_cache_head)
            if (!evict_front()) return false;
        g_cache_head = 0;
    }
    while (!g_cache_ring.empty() && g_cache_ring.front().offset < g_cache_head + words)
        if (!evict_front()) return false;
    offset = g_cache_head;
    std::uint32_t *to = g_cache_copy.data() + offset;
    to[words - 1] = 0;
    to[words - 2] = 0;
    std::memcpy(to, bytes, n);
    wgpuQueueWriteBuffer(g_queue, g_cache, static_cast<std::uint64_t>(offset) * 4, to, static_cast<std::uint64_t>(words) * 4);
    g_cache_ring.push_back({key, offset, words, n, g_serial});
    g_cache_index.emplace(key, &g_cache_ring.back());
    g_cache_head += words;
    return true;
}

std::uint32_t arena_append(const void *bytes, std::size_t size) {
    const std::uint32_t offset = static_cast<std::uint32_t>(g_arena.size());
    const std::size_t words = (size + 3) / 4 + 1;
    std::uint32_t *to = g_arena.grow(words);
    to[words - 2] = 0;
    to[words - 1] = 0;
    std::memcpy(to, bytes, size);
    return offset;
}

float u2f(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Span {
    std::int16_t ty, tx0, tx1;
};
std::vector<Span> g_spans;
std::vector<std::uint32_t> g_span_start;

void add_spans(const std::uint32_t *rec, std::uint32_t lo, std::uint32_t hi) {
    const int sc = static_cast<int>(g_scale);
    const int tx0 = static_cast<int>(lo & 0xFFFFu) * sc / g_tile, ty0 = static_cast<int>(lo >> 16) * sc / g_tile;
    const int tx1 = (static_cast<int>(hi & 0xFFFFu) * sc + sc - 1) / g_tile, ty1 = (static_cast<int>(hi >> 16) * sc + sc - 1) / g_tile;
    const auto whole = [&] {
        for (int ty = ty0; ty <= ty1; ++ty) g_spans.push_back({static_cast<std::int16_t>(ty), static_cast<std::int16_t>(tx0), static_cast<std::int16_t>(tx1)});
    };
    if (rec[1] != 0 || tx1 - tx0 < 2) {
        whole();
        return;
    }
    const double x[3] = {u2f(rec[4]), u2f(rec[6]), u2f(rec[8])}, y[3] = {u2f(rec[5]), u2f(rec[7]), u2f(rec[9])};
    const double inv = u2f(rec[10]);
    static const int kEdges[3][2] = {{1, 2}, {2, 0}, {0, 1}};
    double m[3], q[3];
    int kind[3];
    for (int e = 0; e < 3; ++e) {
        const int a = kEdges[e][0], b = kEdges[e][1];
        const double ca = (y[a] - y[b]) * inv, cb = (x[b] - x[a]) * inv, cc = (x[a] * y[b] - x[b] * y[a]) * inv;
        if (!std::isfinite(ca) || !std::isfinite(cb) || !std::isfinite(cc)) {
            whole();
            return;
        }
        kind[e] = ca > 0.0 ? 1 : ca < 0.0 ? -1 : 0;
        m[e] = kind[e] ? -cb / ca : cb;
        q[e] = kind[e] ? -cc / ca : cc;
    }
    const double span = static_cast<double>(g_tile) / static_cast<double>(sc);
    for (int ty = ty0; ty <= ty1; ++ty) {
        const double y0 = ty * span - 1.0, y1 = (ty + 1) * span + 1.0;
        double left = -1e30, right = 1e30;
        bool empty = false;
        for (int e = 0; e < 3; ++e) {
            const double a0 = m[e] * y0, a1 = m[e] * y1;
            if (kind[e] == 0) empty = empty || std::max(a0, a1) + q[e] < 0.0;
            else if (kind[e] > 0) left = std::max(left, std::min(a0, a1) + q[e]);
            else right = std::min(right, std::max(a0, a1) + q[e]);
        }
        left -= 1.0;
        right += 1.0;
        if (empty || left > right) continue;
        const int rx0 = std::max(tx0, static_cast<int>(std::floor(std::max(left, -1.0) / span)));
        const int rx1 = std::min(tx1, static_cast<int>(std::floor(std::min(right, 4096.0) / span)));
        if (rx0 <= rx1) g_spans.push_back({static_cast<std::int16_t>(ty), static_cast<std::int16_t>(rx0), static_cast<std::int16_t>(rx1)});
    }
}

bool reads_own_word(const std::uint32_t *rec, const std::uint32_t *setup) {
    const std::uint32_t fpsm = setup[3], tpsm = setup[14];
    if (rec[1] != 1 || (setup[0] & (16u | 16384u)) != 16u || setup[12] != setup[1] || setup[13] != setup[2]) return false;
    float f[6], uv[6];
    std::memcpy(f, rec + 4, sizeof f);
    std::memcpy(uv, rec + 15, sizeof uv);
    const float du = (uv[3] - uv[0]) / 16.0f, dv = (uv[4] - uv[1]) / 16.0f;
    if (f[2] == 0.0f || f[3] == 0.0f || std::fabs(du - f[2]) > 0.0005f * std::fabs(f[2]) || std::fabs(dv - f[3]) > 0.0005f * std::fabs(f[3])) return false;
    const int x0 = static_cast<int>(rec[2] & 0xFFFFu), y0 = static_cast<int>(rec[2] >> 16), x1 = static_cast<int>(rec[3] & 0xFFFFu), y1 = static_cast<int>(rec[3] >> 16);
    const auto texel = [](float a, float b, float origin, float span, int p) {
        return static_cast<int>(std::floor((a + (b - a) * ((static_cast<float>(p) - origin) / span)) / 16.0f));
    };
    for (const int y : {y0, y1})
        if (texel(uv[1], uv[4], f[1], f[3], y) != y) return false;
    const int ox = texel(uv[0], uv[3], f[0], f[2], x0) - x0;
    if (texel(uv[0], uv[3], f[0], f[2], x1) - x1 != ox) return false;
    const auto wide = [](std::uint32_t psm) { return psm == PSMCT32 || psm == PSMCT24 || psm == PSMT8H || psm == PSMT4HL || psm == PSMT4HH; };
    // A 16-bit sprite reading the other half of its word (x ^ 8, a channel shuffle) reads the copy like any other: nothing
    // makes the pixel holding that half read before this one writes, and PCSX2 too reads a draw's texture as it was before it.
    return ox == 0 && (tpsm == fpsm || (wide(fpsm) && wide(tpsm)));
}

bool read_rect(const std::uint32_t *rec, const std::uint32_t *setup, int &u0, int &v0, int &u1, int &v1) {
    const int tw = std::max(1 << (setup[15] & 0xFFu), static_cast<int>(setup[18] >> 16) + 1), th = std::max(1 << (setup[15] >> 8), static_cast<int>(setup[19] >> 16) + 1);
    u0 = v0 = 0;
    u1 = tw - 1;
    v1 = th - 1;
    if (rec[1] != 1 || !(setup[0] & 16u)) return false;
    float uv[6];
    std::memcpy(uv, rec + 15, sizeof uv);
    int a0 = static_cast<int>(std::floor(std::min(uv[0], uv[3]) / 16.0f)) - 1, a1 = static_cast<int>(std::ceil(std::max(uv[0], uv[3]) / 16.0f)) + 1;
    int b0 = static_cast<int>(std::floor(std::min(uv[1], uv[4]) / 16.0f)) - 1, b1 = static_cast<int>(std::ceil(std::max(uv[1], uv[4]) / 16.0f)) + 1;
    const auto clamped = [](std::uint32_t mode, int &lo, int &hi, int size) {
        if (lo >= 0 && hi < size) return true;
        if (mode != 1 || lo + 1 < 0 || hi - 1 >= size) return false;
        lo = std::max(lo, 0);
        hi = std::min(hi, size - 1);
        return true;
    };
    if (!clamped((setup[7] >> 24) & 3u, a0, a1, tw) || !clamped((setup[7] >> 26) & 3u, b0, b1, th)) return false;
    u0 = a0;
    v0 = b0;
    u1 = a1;
    v1 = b1;
    return true;
}

Range pages(std::uint32_t bp, std::uint32_t bw, std::uint32_t psm, int x0, int y0, int x1, int y1) {
    std::uint32_t pw, ph;
    page_size(psm, pw, ph);
    const std::uint64_t per_row = std::max<std::uint32_t>(1, bw * 64 / pw);
    const auto page = [&](int x, int y) {
        return static_cast<std::uint64_t>(bp) * 256 + (static_cast<std::uint64_t>(std::max(x, 0)) / pw + static_cast<std::uint64_t>(std::max(y, 0)) / ph * per_row) * 8192;
    };
    Range r;
    r.add(page(x0, y0), page(x1, y1) + 8192);
    return r;
}

// When each page of GS memory was last written, by a clock that counts the writes as they are queued (which is the
// order the GPU makes them in): a texture decoded after the last write to its pages still holds (see plan_decodes).
std::uint64_t g_write_clock = 0, g_page_written[512] = {};

void wrote(const Range &r) {
    ++g_write_clock;
    if (r.hi - r.lo >= kMemorySize) {
        std::fill(std::begin(g_page_written), std::end(g_page_written), g_write_clock);
        return;
    }
    for (std::uint64_t at = r.lo & ~8191ull; at < r.hi; at += 8192) g_page_written[(at >> 13) & 511u] = g_write_clock;
}

std::uint64_t last_written(const Range &r) {
    std::uint64_t last = 0;
    for (std::uint64_t at = r.lo & ~8191ull; at < r.hi; at += 8192) last = std::max(last, g_page_written[(at >> 13) & 511u]);
    return last;
}

// A decoded texture is stale as soon as any of its GS pages was written
// after the decode was queued. Unlike last_written(), this can stop at
// the first invalidating page instead of scanning the entire texture.
bool written_since(const Range &r, std::uint64_t when) {
    for (std::uint64_t at = r.lo & ~8191ull; at < r.hi; at += 8192)
        if (g_page_written[(at >> 13) & 511u] >= when) return true;
    return false;
}

// Texture rectangles may cover only a few pages per GS row. The bounding
// Range includes untouched pages between those rows, causing false cache
// invalidations. Check only the texture's pages when rows do not wrap.
// For unusual widths, retain the conservative contiguous-range check.
bool texture_written_since(std::uint32_t bp, std::uint32_t bw, std::uint32_t psm,
                           std::uint32_t tw, std::uint32_t th,
                           const Range &bounds, std::uint64_t when) {
    std::uint32_t pw, ph;
    page_size(psm, pw, ph);
    const std::uint64_t row_pages = std::max<std::uint32_t>(1, bw * 64 / pw);
    const std::uint64_t width_pages = ((1ull << tw) + pw - 1) / pw;
    const std::uint64_t height_pages = ((1ull << th) + ph - 1) / ph;
    // GS base pointers are 256-byte aligned, not necessarily 8 KiB aligned.
    // An unaligned GS page straddles two tracked physical pages.
    if (width_pages >= row_pages || ((static_cast<std::uint64_t>(bp) * 256) & 8191ull))
        return written_since(bounds, when);
    const std::uint64_t base_page = (static_cast<std::uint64_t>(bp) * 256) >> 13;
    for (std::uint64_t y = 0; y < height_pages; ++y) {
        const std::uint64_t row = base_page + y * row_pages;
        for (std::uint64_t x = 0; x < width_pages; ++x)
            if (g_page_written[(row + x) & 511u] >= when) return true;
    }
    return false;
}

void mark_rendered(const Range &r) {
    for (std::uint64_t at = r.lo; at < r.hi; at += 8192) g_rendered[(at >> 13) & 511u] = true;
    wrote(r);
}

bool rendered(const Range &r) {
    for (std::uint64_t at = r.lo; at < r.hi; at += 8192)
        if (g_rendered[(at >> 13) & 511u]) return true;
    return false;
}

struct Decode {
    std::uint32_t setup, offset, words;
};
std::vector<Decode> g_decodes;

// What a CLUT slot holds, as a number that is the same only for the same entries: each load mixes into the number of
// the slot it copies the one of what it reads, and when that was last written (load_clut).
std::uint64_t g_clut_content[kClutSlots] = {};

std::uint64_t mix(std::uint64_t h, std::uint64_t v) {
    h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    h *= 0xFF51AFD7ED558CCDull;
    return h ^ (h >> 33);
}

// Textures decoded to 32-bit texels (decode in gs.wgsl), kept across batches and frames for as long as GS memory under
// them is not written and, for a palette texture, while the CLUT is the same: a race frame reads the same textures
// in many batches, and each decode was a dispatch and as many texels as the texture has. When the buffer is full
// it starts over. PS2_DECODE_CACHE=0 decodes each batch's textures for that batch alone, as before.
struct DecodeKey {
    std::uint32_t tbp, tbw, tpsm, size, cpsm, csa, ta, aem;
    std::uint64_t clut;
    bool operator==(const DecodeKey &o) const {
        return tbp == o.tbp && tbw == o.tbw && tpsm == o.tpsm && size == o.size && cpsm == o.cpsm && csa == o.csa && ta == o.ta && aem == o.aem && clut == o.clut;
    }
};
struct DecodeKeyHash {
    std::size_t operator()(const DecodeKey &k) const {
        std::uint64_t h = mix(mix(mix(mix(k.tbp, k.tbw), k.tpsm), k.size), k.cpsm);
        return static_cast<std::size_t>(mix(mix(mix(mix(h, k.csa), k.ta), k.aem), k.clut));
    }
};
struct Decoded {
    std::uint32_t offset;
    std::uint64_t when;  // the write clock when it was queued
    std::uint32_t words;
    std::uint64_t last_use_serial;
};
std::unordered_map<DecodeKey, Decoded, DecodeKeyHash> g_decoded_cache;
struct DecodeBlock { std::uint32_t offset, words; std::uint64_t retire_serial; };
std::multimap<std::uint64_t, DecodeBlock> g_decode_retired;
std::vector<DecodeBlock> g_decode_free;
std::uint64_t g_decode_completed_serial = 0;
std::uint64_t g_decode_evictions = 0, g_decode_reclaimed_words = 0;
void decode_retire(const Decoded &d) {
    // Keep pending blocks ordered by the GPU serial that protects them.
    // Collection only visits blocks that can actually be recycled.
    g_decode_retired.emplace(d.last_use_serial, DecodeBlock{d.offset, d.words, d.last_use_serial});
}
void decode_collect() {
    // Keep free blocks sorted and coalesced as each completed retirement
    // arrives, instead of re-sorting the entire list for every LRU victim.
    while (!g_decode_retired.empty() &&
           g_decode_retired.begin()->first <= g_decode_completed_serial) {
        const DecodeBlock block = g_decode_retired.begin()->second;
        g_decode_retired.erase(g_decode_retired.begin());
        auto it = std::lower_bound(g_decode_free.begin(), g_decode_free.end(), block.offset,
                                   [](const DecodeBlock &free, std::uint32_t offset) {
                                       return free.offset < offset;
                                   });
        const auto index = static_cast<std::size_t>(it - g_decode_free.begin());
        if (index && g_decode_free[index - 1].offset + g_decode_free[index - 1].words == block.offset) {
            g_decode_free[index - 1].words += block.words;
            if (index < g_decode_free.size() &&
                g_decode_free[index - 1].offset + g_decode_free[index - 1].words == g_decode_free[index].offset) {
                g_decode_free[index - 1].words += g_decode_free[index].words;
                g_decode_free.erase(g_decode_free.begin() + index);
            }
        } else if (index < g_decode_free.size() &&
                   block.offset + block.words == g_decode_free[index].offset) {
            g_decode_free[index].offset = block.offset;
            g_decode_free[index].words += block.words;
        } else {
            g_decode_free.insert(it, block);
        }
    }
    // A free block at the end of the append arena can be returned to the
    // cursor outright. Repeated tail retirements can then release more space,
    // avoiding needless LRU eviction and reducing fragmentation.
    while (!g_decode_free.empty()) {
        const DecodeBlock &tail = g_decode_free.back();
        if (tail.offset + tail.words != g_decoded_used) break;
        g_decoded_used = tail.offset;
        g_decode_free.pop_back();
    }
}
bool decode_allocate(std::uint32_t words, std::uint32_t &offset) {
    decode_collect();
    const auto take_free = [&]() {
        // Best fit preserves larger contiguous slots for future textures.
        // Free blocks are ordered by offset, not by size.
        std::size_t best = g_decode_free.size();
        for (std::size_t i = 0; i < g_decode_free.size(); ++i) {
            const auto available = g_decode_free[i].words;
            if (available < words) continue;
            if (best == g_decode_free.size() || available < g_decode_free[best].words)
                best = i;
            if (available == words) break;
        }
        if (best == g_decode_free.size()) return false;
        auto &block = g_decode_free[best];
        offset = block.offset;
        block.offset += words;
        block.words -= words;
        if (!block.words) g_decode_free.erase(g_decode_free.begin() + best);
        g_decode_reclaimed_words += words;
        return true;
    };
    if (take_free()) return true;
    if (g_decoded_used + words <= kDecodedWords) {
        offset = g_decoded_used;
        g_decoded_used += words;
        return true;
    }
    // Select reclaim candidates once, oldest first. Repeatedly scanning the
    // entire hash table for each victim was O(entries * evictions) per miss.
    // Never evict textures referenced by queued or unfinished GPU work.
    // Keep scratch capacity across allocation misses to avoid repeated heap
    // allocation when the decoded cache is under pressure.
    using DecodeCacheIterator = decltype(g_decoded_cache.begin());
    static std::vector<DecodeCacheIterator> candidates;
    candidates.clear();
    candidates.reserve(g_decoded_cache.size());
    for (auto it = g_decoded_cache.begin(); it != g_decoded_cache.end(); ++it)
        if (it->second.last_use_serial <= g_decode_completed_serial)
            candidates.push_back(it);
    std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) {
        return a->second.last_use_serial < b->second.last_use_serial;
    });
    for (const auto victim : candidates) {
        decode_retire(victim->second);
        g_decoded_cache.erase(victim);
        ++g_decode_evictions;
        decode_collect();
        if (take_free()) return true;
    }
    return false;
}

// Admission history is independent of the decoded buffer: a full buffer reset
// must not erase evidence that a texture is used repeatedly.
std::unordered_map<DecodeKey, std::uint8_t, DecodeKeyHash> g_decode_seen;
std::uint64_t g_decode_hits = 0, g_decode_misses = 0, g_decode_words = 0, g_decode_restarts = 0;
std::uint64_t g_decode_saved_words = 0;  // texels reused instead of decoded again
std::uint64_t g_decode_invalidated = 0, g_decode_cold = 0, g_decode_small = 0;
std::uint64_t g_decode_admission_skips = 0, g_decode_admission_texels = 0;
std::uint64_t g_decode_capacity_bypasses = 0, g_decode_capacity_texels = 0;

bool palette_psm(std::uint32_t psm) { return psm == PSMT8 || psm == PSMT4 || psm == PSMT8H || psm == PSMT4HL || psm == PSMT4HH; }

bool decode_admission_on() {
    static const bool on = !std::getenv("PS2_DECODE_ADMISSION") ||
                           std::atoi(std::getenv("PS2_DECODE_ADMISSION")) != 0;
    return on;
}

bool decode_cache_on() {
    static const bool on = !std::getenv("PS2_DECODE_CACHE") || std::atoi(std::getenv("PS2_DECODE_CACHE")) != 0;
    return on;
}

void plan_decodes(std::uint32_t prims) {
    Batch &b = g_batch;
    g_decodes.clear();
    if (!g_decode) return;
    const bool cache = decode_cache_on();
    if (!cache) {
        // The uncached path resets the append cursor after each submit.
        // Recycled offsets from the persistent cache must never be mixed
        // with those transient allocations.
        g_decoded_cache.clear();
        g_decode_free.clear();
        g_decode_retired.clear();
            if (prims <= kDecodePrims) return;
    }
    const std::uint32_t setups = static_cast<std::uint32_t>(b.setups.size() / kSetupWords);
    for (std::uint32_t i = 0; i < setups; ++i) {
        std::uint32_t *w = b.setups.data() + i * kSetupWords;
        if (!(w[0] & 2u)) continue;
        const std::uint32_t tw = w[15] & 0xFFu, th = w[15] >> 8;
        const std::uint32_t mu = (w[7] >> 24) & 3u, mv = (w[7] >> 26) & 3u;
        const auto inside = [](std::uint32_t mode, std::uint32_t region, std::uint32_t size) { return mode < 2 || ((region & 0x3FFu) < (1u << size) && (region >> 16) < (1u << size)); };
        if (!inside(mu, w[18], tw) || !inside(mv, w[19], th)) continue;
        const Range texels = pages(w[12], w[13], w[14], 0, 0, (1 << tw) - 1, (1 << th) - 1);
        if (g_scale > 1 && rendered(texels)) continue;
        const bool palette = palette_psm(w[14]);
        const DecodeKey key{w[12], w[13], w[14], w[15], palette ? w[16] : 0u, palette ? w[17] : 0u, w[20], (w[0] & 32768u) ? 1u : 0u, palette ? g_clut_content[w[22] % kClutSlots] : 0u};
        const auto found = g_decoded_cache.find(key);
        if (found != g_decoded_cache.end()) {
            if (!texture_written_since(w[12], w[13], w[14], tw, th, texels, found->second.when)) {
                w[24] = found->second.offset;
                found->second.last_use_serial = g_serial;
                ++g_decode_hits;
                g_decode_saved_words += 1ull << (tw + th);
                continue;
            }
            ++g_decode_invalidated;
            decode_retire(found->second);
            g_decoded_cache.erase(found);
        } else {
            ++g_decode_cold;
        }
        // Decoding costs a dispatch and every texel: for a few primitives reading GS memory directly is cheaper.
        if (prims <= kDecodePrims) {
            ++g_decode_small;
            continue;
        }
        const std::uint32_t words = 1u << (tw + th);
        if (words > kDecodedWords / 4) continue;
        // Large, one-off textures consume the decoded buffer and trigger
        // expensive full-cache resets. Leave their sampling on the direct
        // GS-memory path until a second encounter establishes reuse.
        // Small textures stay eager: their decode and footprint are cheap.
        if (cache && decode_admission_on() && words >= 4096u) {
            if (g_decode_seen.size() >= 8192u) g_decode_seen.clear();
            auto [seen, inserted] = g_decode_seen.try_emplace(key, 0u);
            if (seen->second < 2u) ++seen->second;
            if (seen->second < 2u) {
                ++g_decode_admission_skips;
                g_decode_admission_texels += words;
                continue;
            }
        }
        std::uint32_t decoded_offset = 0;
        if (!(cache ? decode_allocate(words, decoded_offset)
                    : (g_decoded_used + words <= kDecodedWords
                        ? (decoded_offset = g_decoded_used, g_decoded_used += words, true)
                        : false))) {
            // Never overwrite a decoded slot while queued native passes (or
            // already submitted GPU work) may still reference its offset.
            // Keep existing cached textures alive; this miss uses the direct
            // GS-memory sampler until a synchronized reclamation is possible.
            ++g_decode_capacity_bypasses;
            g_decode_capacity_texels += words;
            continue;
        }
        w[24] = decoded_offset;
        g_decodes.push_back({i, decoded_offset, words});
        g_decoded_cache[key] = {decoded_offset, g_write_clock + 1, words, g_serial};
        ++g_decode_misses;
        g_decode_words += words;
    }
}

std::uint32_t snapshot_batch(std::uint32_t prims, std::uint32_t &lo, std::uint32_t &hi, std::uint32_t &stride, std::uint32_t &run) {
    Batch &b = g_batch;
    lo = hi = stride = run = 0;
    if (g_batch_reads.empty()) return 0;
    const Range fb = pages(b.fbp, b.fbw, b.fpsm, b.minx, b.miny, b.maxx, b.maxy);
    const Range z = pages(b.zbp, b.fbw, b.zpsm, b.minx, b.miny, b.maxx, b.maxy);
    const auto written = [&](const Range &t) { return t.overlaps(fb.lo, fb.hi) || (b.zwrite && t.overlaps(z.lo, z.hi)); };
    bool any = false;
    for (const Range &t : g_batch_reads) any = any || written(t);
    if (!any) return 0;
    std::uint32_t pw, ph;
    page_size(b.fpsm, pw, ph);
    const int per_row = static_cast<int>(std::max<std::uint32_t>(1, b.fbw * 64 / pw));
    const int fx0 = std::max(b.minx, 0) / static_cast<int>(pw), fy0 = std::max(b.miny, 0) / static_cast<int>(ph);
    const int fx1 = b.maxx / static_cast<int>(pw), fy1 = b.maxy / static_cast<int>(ph);
    int px0 = 1 << 30, py0 = 1 << 30, px1 = -1, py1 = -1;
    bool rect = fx1 < per_row;
    Range shared;
    for (std::uint32_t i = 0; i < prims; ++i) {
        const std::uint32_t *rec = b.prims.data() + i * kPrimWords;
        const std::uint32_t *setup = b.setups.data() + rec[0] * kSetupWords;
        if (!(setup[0] & 2u) || reads_own_word(rec, setup)) continue;
        int u0, v0, u1, v1;
        read_rect(rec, setup, u0, v0, u1, v1);
        const Range t = region(setup[12], setup[13], setup[14], u0, v0, u1, v1);
        if (b.zwrite && t.overlaps(z.lo, z.hi)) {
            shared.add(std::max(t.lo, z.lo), std::min(t.hi, z.hi));
            rect = false;
        }
        if (!t.overlaps(fb.lo, fb.hi)) continue;
        shared.add(std::max(t.lo, fb.lo), std::min(t.hi, fb.hi));
        std::uint32_t tpw, tph;
        page_size(setup[14], tpw, tph);
        if (setup[12] != b.fbp || setup[13] != b.fbw || tpw != pw || tph != ph || u1 / static_cast<int>(pw) >= per_row) {
            rect = false;
            continue;
        }
        const int x0 = std::max(u0 / static_cast<int>(pw), fx0), y0 = std::max(v0 / static_cast<int>(ph), fy0);
        const int x1 = std::min(u1 / static_cast<int>(pw), fx1), y1 = std::min(v1 / static_cast<int>(ph), fy1);
        if (x0 > x1 || y0 > y1) continue;
        px0 = std::min(px0, x0);
        py0 = std::min(py0, y0);
        px1 = std::max(px1, x1);
        py1 = std::max(py1, y1);
    }
    if (shared.lo >= shared.hi) return 0;
    if (rect && px1 >= 0) {
        const std::uint64_t base = static_cast<std::uint64_t>(b.fbp) * 256 + (static_cast<std::uint64_t>(px0) + static_cast<std::uint64_t>(py0) * per_row) * 8192;
        if (base + static_cast<std::uint64_t>(py1 - py0 + 1) * per_row * 8192 <= kMemorySize) {
            lo = static_cast<std::uint32_t>(base / 4);
            run = static_cast<std::uint32_t>(px1 - px0 + 1) * 2048u;
            stride = static_cast<std::uint32_t>(per_row) * 2048u;
            const std::uint32_t rows = static_cast<std::uint32_t>(py1 - py0 + 1);
            hi = lo + (rows - 1) * stride + run;
            const std::uint32_t words = rows * run * g_scale * g_scale;
            dispatch_linear(kSnapshot, words, push_params({lo, words, stride, run}));
            return words;
        }
    }
    lo = static_cast<std::uint32_t>(std::min<std::uint64_t>(shared.lo, kMemorySize) / 4);
    hi = static_cast<std::uint32_t>(std::min<std::uint64_t>(shared.hi, kMemorySize) / 4);
    if (hi <= lo) return 0;
    stride = run = hi - lo;
    const std::uint32_t words = (hi - lo) * g_scale * g_scale;
    dispatch_linear(kSnapshot, words, push_params({lo, words, stride, run}));
    return words;
}

void note_raster(std::uint32_t prims, std::uint32_t tiles, std::uint32_t entries, std::uint32_t snapshot) {
    if (!g_profile_out) return;
    const Batch &b = g_batch;
    const std::uint32_t *first = b.setups.data();
    std::uint32_t textured = 0;
    for (std::uint32_t i = 0; i < prims; ++i) textured += (b.setups[b.prims[i * kPrimWords] * kSetupWords] & 2u) ? 1u : 0u;
    g_command_info.push_back({g_commands.size(), {prims, tiles, entries, snapshot, b.prims[1], first[3] | (first[14] << 8) | (first[6] << 16), first[0], textured}});
}


// The native renderer. Triangles and sprites whose state the GPU's fixed functions can express are drawn by its own
// rasterizer (native.wgsl) into textures standing for the frame and Z buffers, which is several times cheaper than
// rasterizing in a compute shader. GS memory stays the meeting point: a texture is written back (stored) before
// anything else reads or writes the memory under it, and loaded again before it is drawn to once something else has
// written there. Textures a draw samples are still read from GS memory. Everything else, and every state not listed
// in native_state, goes through the compute rasterizer as before, a whole batch at a time.
struct Surface {
    std::uint32_t bp = 0, bw = 0, width = 0, height = 0;  // block pointer, buffer width in 64s, size in pixels
    bool depth = false;
    std::uint32_t psm = PSMCT32;  // a depth buffer's format; colour is kept as CT32 words
    // A colour target's copy for draws that read the pixel already there, made when the first such draw comes.
    WGPUTexture backdrop = nullptr;
    WGPUTextureView backdrop_view = nullptr;
    WGPUBindGroup backdrop_group = nullptr;
    WGPUTexture texture = nullptr;
    WGPUTextureView view = nullptr;
    WGPUBindGroup group = nullptr;
    bool dirty = false;  // the texture is newer than GS memory inside this rectangle
    int dx0 = 0, dy0 = 0, dx1 = 0, dy1 = 0;
    int stale0 = 0, stale1 = -1;  // GS memory is newer than the texture in these rows
    std::uint64_t used = 0;
};

// A draw of a native pass, or with no pipeline a copy of the colour target's rectangle to its backdrop, which the
// feedback draws after it read (see native_feedback in native.wgsl).
struct NativeDraw {
    WGPURenderPipeline pipeline;
    std::uint32_t param, vertices, x, y, w, h;
    float constant;
    bool feedback = false;
};

struct NativePass {
    Surface *color, *depth;
    std::vector<NativeDraw> draws;
};

struct NativeState {
    std::uint32_t key = 0, flags = 0;
    std::uint32_t premultiply = 0;  // the shader multiplies the colour by C itself: 1, with 2 C is FIX, with 4 rounding up
    float constant = 0.0f;
    bool draws = true;
    bool alpha_factor = false;  // the source alpha is a blend factor, which the blender clamps to one (0x80)
    std::uint32_t why = 0;      // an index into kWhy when the state is not for the native renderer
    bool feedback = false;      // the shader makes the whole pixel from the one already there
};

constexpr std::size_t kMaxSurfaces = 24;

bool g_native = false;
std::vector<std::unique_ptr<Surface>> g_surfaces, g_retired;
std::vector<NativePass> g_native_passes;
std::uint64_t g_surface_clock = 0;
std::uint64_t g_native_batches = 0, g_native_prims = 0, g_compute_batches = 0, g_compute_prims = 0, g_stores = 0, g_loads = 0, g_native_draws = 0;
WGPUBindGroupLayout g_native_layout = nullptr, g_target_layout = nullptr;
WGPUPipelineLayout g_native_pipeline_layout = nullptr, g_store_layout = nullptr, g_feedback_pipeline_layout = nullptr;
WGPUBindGroupLayout g_backdrop_layout = nullptr;
std::uint64_t g_feedback_draws = 0, g_feedback_copies = 0;
std::uint64_t g_feedback_copy_pixels = 0;
std::uint64_t g_feedback_group_calls = 0;
std::uint64_t g_feedback_outline_tests = 0;
std::uint64_t g_feedback_group_prims = 0;  // unscaled source pixels copied for feedback
bool g_refusal_counted = false;  // flush_native said why it gave a batch up
WGPUBindGroup g_native_group = nullptr;
WGPUShaderModule g_native_module = nullptr;
WGPURenderPipeline g_load_pipelines[2] = {};
std::unordered_map<std::uint32_t, WGPURenderPipeline> g_native_pipelines;

// Whether the GPU's rasterizer and fixed functions can draw this setup, and with which pipeline state. The key: bit 0
// a depth attachment, 1-2 the depth test (never, always, >=, >), 3 depth writes, 4-7 the channels written, 8 blending,
// 9-10 its operation (add, source minus destination, destination minus source), 11-12 and 13-14 the source and
// destination factors (0, 1, C, 1 - C), 15 C is FIX and not the source alpha, 16 the shader writes the depth.
// PS2_NATIVE_SKIP=<sum>: leave a kind of draw to the compute rasterizer, to find which one a wrong picture comes from.
// 1 blending by a factor, 2 any blending, 4 textures, 8 alpha tests, 16 depth tests, 32 triangles, 64 sprites,
// 128 blending by a factor that has a texture's alpha in it (see draw), 256 what needs the pixel already there;
// and 512 does not leave anything out (see native_state).
std::uint32_t native_skip() {
    static const std::uint32_t skip = std::getenv("PS2_NATIVE_SKIP") ? static_cast<std::uint32_t>(std::atoi(std::getenv("PS2_NATIVE_SKIP"))) : 0u;
    return skip;
}

// Where the GS thread's own time goes inside the renderer, for the [native] report: each part's time without what
// it calls that is itself timed.
enum CpuPart { kCpuDraw, kCpuClut, kCpuImage, kCpuPre, kCpuNative, kCpuCompute, kCpuSubmit, kCpuParts };
const char *const kCpuNames[kCpuParts] = {"draw", "clut", "image", "uploads", "native flush", "compute flush", "submit"};
struct Cpu {
    double ms[kCpuParts] = {};
    int current = -1;
    std::chrono::steady_clock::time_point since;
} g_cpu;
bool cpu_on() {
    static const bool on = std::getenv("PS2_GL_STATS") || std::getenv("PS2_NATIVE_STATS");
    return on;
}
struct Timed {
    int before = -1;
    explicit Timed(int part) {
        if (!cpu_on()) return;
        const auto now = std::chrono::steady_clock::now();
        if (g_cpu.current >= 0) g_cpu.ms[g_cpu.current] += std::chrono::duration<double, std::milli>(now - g_cpu.since).count();
        before = g_cpu.current;
        g_cpu.current = part;
        g_cpu.since = now;
    }
    ~Timed() {
        if (!cpu_on()) return;
        const auto now = std::chrono::steady_clock::now();
        g_cpu.ms[g_cpu.current] += std::chrono::duration<double, std::milli>(now - g_cpu.since).count();
        g_cpu.current = before;
        g_cpu.since = now;
    }
};

// PS2_NATIVE_TRACE=1 with the [native] report (--stats): once a race has run for a while, everything the renderer
// does over a few frames, a line each, to see what a frame is made of on hardware. Writing those thousands of lines
// to the SD card stops the game for seconds, so it is off unless asked for.
int g_trace = 0;
#define TRACE(...) do { if (g_trace > 0) std::fprintf(stderr, "[trace] " __VA_ARGS__); } while (0)
const Range *g_sync_range = nullptr;
bool native_debug() {
    static const bool on = std::getenv("PS2_NATIVE_DEBUG") != nullptr;
    return on;
}

// What a target was written back to GS memory for, counted for the [native] line.
const char *const kStoreCause[] = {"copy, readback or display", "texture of a native batch", "target of a compute batch", "texture of a compute batch", "upload or CLUT", ""};
std::uint64_t g_store_cause[6] = {};
std::uint32_t g_sync_cause = 0;

// Why a draw is left to the compute rasterizer, counted for the [native] line.
const char *const kWhy[] = {"",                "PS2_NATIVE_SKIP",        "16-bit target",     "DATE",           "scan mask",         "",         
                            "mask in channel", "alpha fail writes part", "PABE or no clamp",  "blend by Ad",    "FIX above 0x80",    "blend 1+C",
                            "highlight alpha", "line or point",          "wider than buffer", "Z above range" , "vertex alpha>0x80", "overlapping, reads target", "small, target in memory", "no native target"};
constexpr std::size_t kWhyCount = sizeof kWhy / sizeof kWhy[0];
std::uint64_t g_why_prims[kWhyCount] = {}, g_why_batches[kWhyCount] = {};

bool native_fixed(const std::uint32_t *w, bool has_depth, NativeState &out) {
    const std::uint32_t flags = w[0], fpsm = w[3], modes = w[7];
    const std::uint32_t skip = native_skip();
    out.why = 0;
    const auto why = [&](std::uint32_t reason) {
        out.why = reason;
        return false;
    };
    if (((skip & 2u) && (flags & 8u)) || ((skip & 4u) && (flags & 2u)) || ((skip & 8u) && (flags & 64u)) || ((skip & 16u) && (flags & 512u))) return why(1);
    if ((fpsm != PSMCT32 && fpsm != PSMCT24) || w[2] == 0) return why(2);
    if (flags & 128u) return why(3);       // destination alpha test
    if ((modes >> 28) & 2u) return why(4); // every other line masked
    const bool zte = flags & 512u;
    std::uint32_t mask = 0;
    for (int ch = 0; ch < (fpsm == PSMCT24 ? 3 : 4); ++ch) {
        const std::uint32_t m = (w[4] >> (ch * 8)) & 0xFFu;
        if (m == 0) mask |= 1u << ch;
        else if (m != 0xFFu) return why(6);  // a mask inside a channel
    }
    bool write_z = zte && !(flags & 32u);
    out.flags = flags;
    if (flags & 64u) {
        const std::uint32_t atst = modes & 7u, afail = (modes >> 4) & 3u;
        if (atst == 1) {
            out.flags &= ~64u;
        } else if (atst == 0) {
            // The test never passes, so what a failed test leaves alone is known here.
            out.flags &= ~64u;
            if (afail == 0) mask = 0;
            if (afail == 2) mask = 0;
            else write_z = false;
            if (afail == 3) mask &= 7u;
        } else if (afail != 0) {
            return why(7);  // a failed test writes some of the pixel: needs the test per pixel and two write masks
        }
    }
    const std::uint32_t ztst = (modes >> 8) & 3u;
    std::uint32_t blend = 0;
    out.constant = 0.0f;
    out.premultiply = 0;
    out.alpha_factor = false;
    if ((flags & 8u) && (mask & 7u)) {
        if ((flags & 1024u) || !(flags & 2048u)) return why(8);  // blending per pixel alpha, or colours that wrap
        const std::uint32_t a = (modes >> 12) & 3u, b = (modes >> 14) & 3u, c = (modes >> 16) & 3u, d = (modes >> 18) & 3u;
        // (A - B) * C + D as what multiplies the source and the destination colour: so many times C, plus one or not.
        const int source_c = (a == 0) - (b == 0), dest_c = (a == 1) - (b == 1);
        const int source_1 = d == 0, dest_1 = d == 1;
        std::uint32_t op = 0;
        int sf, df;
        if (!source_c && !dest_c) {
            sf = source_1;
            df = dest_1;
        } else if (c == 1) {
            return why(9);  // destination alpha: the texture keeps alpha, not alpha as a factor
        } else if (!dest_c && !source_1) {
            // C only scales the source colour: the shader does it, with the GS's own rounding and for any C.
            out.premultiply = 1u | (c >= 2 ? 2u : 0u) | (source_c < 0 ? 4u : 0u);
            sf = 1;
            df = dest_1;
            op = source_c < 0 ? 2u : 0u;
        } else {
            if ((c >= 2 && w[9] > 0x80u) || (skip & 1u)) return why(10);  // a factor above one
            bool source_minus = false, dest_minus = false;
            const auto factor = [](int times, int one, bool &minus) {
                if (!times) return one;
                if (times == 1 && !one) return 2;
                if (times == -1 && one) return 3;
                if (times == -1) {
                    minus = true;
                    return 2;
                }
                return -1;  // 1 + C
            };
            sf = factor(source_c, source_1, source_minus);
            df = factor(dest_c, dest_1, dest_minus);
            if (sf < 0 || df < 0 || (source_minus && dest_minus)) return why(11);
            op = source_minus ? 2u : dest_minus ? 1u : 0u;
            out.alpha_factor = c == 0;
            if (c >= 2) {
                blend |= 1u << 7;
                out.constant = static_cast<float>(w[9]) / 128.0f;
            }
        }
        if (sf != 1 || df != 0 || op) blend |= 1u | (op << 1) | (static_cast<std::uint32_t>(sf) << 3) | (static_cast<std::uint32_t>(df) << 5);
        else blend = 0;
        if (!(blend & 1u)) out.constant = 0.0f;
    }
    out.draws = !(zte && ztst == 0) && (mask || write_z);
    // PS2_NATIVE_Z=fast leaves depth to the rasterizer's interpolation: quicker where much is hidden, and a step off
    // the compute rasterizer's depth here and there, which shows where both draw surfaces lying on each other.
    static const bool exact = !std::getenv("PS2_NATIVE_Z") || std::strcmp(std::getenv("PS2_NATIVE_Z"), "fast") != 0;
    out.key = (has_depth ? 1u : 0u) | ((zte ? ztst : 1u) << 1) | (write_z ? 8u : 0u) | (mask << 4) | (blend << 8) | (has_depth && zte && exact ? 1u << 16 : 0u);
    return true;
}

// What the native renderer does with a draw state. What the fixed functions cannot express (native_fixed says why)
// a draw that reads the pixel already there can, the shader making the whole pixel: a destination alpha test, a
// blend the blender has no factors for, a mask inside a channel, a failed alpha test that still writes the colour.
// PS2_NATIVE_FEEDBACK=0, or 256 in PS2_NATIVE_SKIP, leaves those to the compute rasterizer.
bool native_state(const std::uint32_t *w, bool has_depth, NativeState &out) {
    out.feedback = false;
    const std::uint32_t flags = w[0], modes = w[7];
    // 512 in PS2_NATIVE_SKIP has every blend made this way, which is the compute rasterizer's to the bit where the
    // blender's is a step off: for telling the two apart.
    const bool fixed = native_fixed(w, has_depth, out), exact_blend = (native_skip() & 512u) && (flags & 8u);
    if (fixed && !exact_blend) return true;
    static const bool on = !std::getenv("PS2_NATIVE_FEEDBACK") || std::atoi(std::getenv("PS2_NATIVE_FEEDBACK")) != 0;
    const std::uint32_t reason = out.why;
    if (!fixed && (!on || (native_skip() & 256u) || !(reason == 3 || (reason >= 6 && reason <= 11)))) return false;
    if ((modes >> 28) & 2u) {
        out.why = 4;
        return false;
    }
    const bool zte = flags & 512u, write_z = zte && !(flags & 32u);
    const std::uint32_t atst = modes & 7u, afail = (modes >> 4) & 3u, ztst = (modes >> 8) & 3u;
    // A failed alpha test that writes the colour and not the depth: the depth write is the pipeline's, all or nothing.
    if ((flags & 64u) && atst != 1 && (afail == 1 || afail == 3) && write_z) {
        out.why = 7;
        return false;
    }
    static const bool exact = !std::getenv("PS2_NATIVE_Z") || std::strcmp(std::getenv("PS2_NATIVE_Z"), "fast") != 0;
    out.why = 0;
    out.flags = flags;
    out.constant = 0.0f;
    out.premultiply = 0;
    out.alpha_factor = false;
    out.feedback = true;
    out.draws = !(zte && ztst == 0);
    out.key = (has_depth ? 1u : 0u) | ((zte ? ztst : 1u) << 1) | (write_z ? 8u : 0u) | (0xFu << 4) | (has_depth && zte && exact ? 1u << 16 : 0u) | (1u << 17);
    return true;
}

WGPURenderPipeline native_pipeline(std::uint32_t key) {
    const auto found = g_native_pipelines.find(key);
    if (found != g_native_pipelines.end()) return found->second;
    WGPUConstantEntry constant = WGPU_CONSTANT_ENTRY_INIT;
    constant.key = view("SCALE");
    constant.value = static_cast<double>(g_scale);
    const bool from_fix = (key >> 15) & 1u;
    const WGPUBlendFactor factors[4] = {WGPUBlendFactor_Zero, WGPUBlendFactor_One, from_fix ? WGPUBlendFactor_Constant : WGPUBlendFactor_Src1Alpha,
                                        from_fix ? WGPUBlendFactor_OneMinusConstant : WGPUBlendFactor_OneMinusSrc1Alpha};
    const WGPUBlendOperation ops[3] = {WGPUBlendOperation_Add, WGPUBlendOperation_Subtract, WGPUBlendOperation_ReverseSubtract};
    WGPUBlendState blend = WGPU_BLEND_STATE_INIT;
    blend.color.operation = ops[std::min((key >> 9) & 3u, 2u)];
    blend.color.srcFactor = factors[(key >> 11) & 3u];
    blend.color.dstFactor = factors[(key >> 13) & 3u];
    blend.alpha.operation = WGPUBlendOperation_Add;  // the GS writes the source alpha, blended or not
    blend.alpha.srcFactor = WGPUBlendFactor_One;
    blend.alpha.dstFactor = WGPUBlendFactor_Zero;
    WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
    target.format = WGPUTextureFormat_RGBA8Unorm;
    target.writeMask = (key >> 4) & 0xFu;
    const bool feedback = (key >> 17) & 1u;
    if (((key >> 8) & 1u) && !feedback) target.blend = &blend;
    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    fragment.module = g_native_module;
    fragment.entryPoint = view(feedback ? ((key >> 16) & 1u ? "native_feedback_depth_fs" : "native_feedback_fs") : (key >> 16) & 1u ? "native_depth_fs" : "native_fs");
    fragment.constantCount = 1;
    fragment.constants = &constant;
    fragment.targetCount = 1;
    fragment.targets = &target;
    static const WGPUCompareFunction compares[4] = {WGPUCompareFunction_Never, WGPUCompareFunction_Always, WGPUCompareFunction_GreaterEqual, WGPUCompareFunction_Greater};
    WGPUDepthStencilState depth = WGPU_DEPTH_STENCIL_STATE_INIT;
    depth.format = WGPUTextureFormat_Depth32Float;
    depth.depthWriteEnabled = (key & 8u) ? WGPUOptionalBool_True : WGPUOptionalBool_False;
    depth.depthCompare = compares[(key >> 1) & 3u];
    WGPURenderPipelineDescriptor rd = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    rd.layout = feedback ? g_feedback_pipeline_layout : g_native_pipeline_layout;
    rd.vertex.module = g_native_module;
    rd.vertex.entryPoint = view("native_vs");
    rd.vertex.constantCount = 1;
    rd.vertex.constants = &constant;
    rd.fragment = &fragment;
    if (key & 1u) rd.depthStencil = &depth;
    const WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(g_device, &rd);
    g_native_pipelines.emplace(key, pipeline);
    return pipeline;
}

// The pages of GS memory under rows y0 to y1 of a surface. Pages are 64 by 32 pixels, and 64 by 64 of a 16-bit format.
Range surface_range(const Surface &s, int y0, int y1) { return pages(s.bp, s.bw, s.psm, 0, y0, static_cast<int>(s.width) - 1, y1); }

int page_rows(std::uint32_t psm) { return psm == PSMZ16 || psm == PSMZ16S ? 64 : 32; }

bool surface_rows(const Surface &s, const Range &r, int &y0, int &y1) {
    const std::uint64_t base = static_cast<std::uint64_t>(s.bp) * 256, row = static_cast<std::uint64_t>(s.bw) * 8192;
    const int tall = page_rows(s.psm);
    const std::uint64_t lo = std::max(r.lo, base), hi = std::min(r.hi, base + row * ((s.height + tall - 1) / tall));
    if (lo >= hi) return false;
    y0 = static_cast<int>((lo - base) / row) * tall;
    y1 = std::min(static_cast<int>((hi - 1 - base) / row) * tall + tall - 1, static_cast<int>(s.height) - 1);
    return true;
}

void store_surface(Surface &s) {
    if (!s.dirty) return;
    s.dirty = false;
    const std::uint32_t w = static_cast<std::uint32_t>(s.dx1 - s.dx0 + 1), h = static_cast<std::uint32_t>(s.dy1 - s.dy0 + 1);
    Command c{s.depth ? kStoreDepth : kStoreColor, (w * g_scale + 7) / 8, (h * g_scale + 7) / 8,
              push_params({s.bp, s.bw, static_cast<std::uint32_t>(s.dx0), static_cast<std::uint32_t>(s.dy0), w, h, s.psm})};
    c.target = &s;
    g_commands.push_back(c);
    wrote(pages(s.bp, s.bw, s.psm, s.dx0, s.dy0, s.dx1, s.dy1));
    ++g_stores;
    ++g_store_cause[g_sync_cause];
    TRACE("store %s %X/%u/%02X rect %d,%d-%d,%d for %s", s.depth ? "depth" : "color", s.bp, s.bw, s.psm, s.dx0, s.dy0, s.dx1, s.dy1, kStoreCause[g_sync_cause]);
    if (g_trace > 0 && g_sync_range) std::fprintf(stderr, " %llX-%llX", static_cast<unsigned long long>(g_sync_range->lo), static_cast<unsigned long long>(g_sync_range->hi));
    if (g_trace > 0) std::fprintf(stderr, "\n");
}

// Something other than a native draw to `self` is about to read, or write, this GS memory: textures holding newer
// pixels there are written back first, and after a write they are stale where it landed.
void sync_memory(const Range &r, bool write, const Surface *self = nullptr) {
    g_sync_range = &r;
    for (const auto &p : g_surfaces) {
        Surface &s = *p;
        if (&s == self) continue;
        if (s.dirty && surface_range(s, s.dy0, s.dy1).overlaps(r.lo, r.hi)) store_surface(s);
        int y0, y1;
        if (write && surface_rows(s, r, y0, y1)) {
            // One span of stale rows is kept, so it can take in rows nothing wrote to. They are loaded again all the
            // same: pixels only the texture has in them are written back before that.
            if (s.stale1 >= s.stale0) {
                y0 = std::min(y0, s.stale0);
                y1 = std::max(y1, s.stale1);
            }
            if (s.dirty && y0 <= s.dy1 && s.dy0 <= y1) store_surface(s);
            s.stale0 = y0;
            s.stale1 = y1;
        }
    }
}

void sync_all() {
    for (const auto &p : g_surfaces) store_surface(*p);
}

void retire_surface(std::size_t index) {
    store_surface(*g_surfaces[index]);
    g_retired.push_back(std::move(g_surfaces[index]));  // commands not yet submitted may still name it
    g_surfaces.erase(g_surfaces.begin() + static_cast<std::ptrdiff_t>(index));
}

void release_retired() {
    for (const auto &s : g_retired) {
        if (s->backdrop) {
            wgpuBindGroupRelease(s->backdrop_group);
            wgpuTextureViewRelease(s->backdrop_view);
            wgpuTextureRelease(s->backdrop);
        }
        wgpuBindGroupRelease(s->group);
        wgpuTextureViewRelease(s->view);
        wgpuTextureRelease(s->texture);
    }
    g_retired.clear();
}

Surface *surface_for(std::uint32_t bp, std::uint32_t bw, bool depth, std::uint32_t psm, std::uint32_t rows, const Surface *keep) {
    for (std::size_t i = 0; i < g_surfaces.size(); ++i) {
        Surface &s = *g_surfaces[i];
        if (s.bp != bp || s.bw != bw || s.depth != depth || s.psm != psm) continue;
        if (s.height >= rows) {
            s.used = ++g_surface_clock;
            return &s;
        }
        retire_surface(i);  // too short: a taller one takes over
        break;
    }
    // Whole pages, and no further than GS memory goes.
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(bw) * 8192, base = static_cast<std::uint64_t>(bp) * 256;
    if (!bw || base >= kMemorySize) return nullptr;
    const std::uint32_t most = static_cast<std::uint32_t>(std::min<std::uint64_t>((kMemorySize - base) / row_bytes * static_cast<std::uint64_t>(page_rows(psm)), 2048));
    const std::uint32_t height = std::min((rows + 63) & ~63u, most);
    if (height < rows) return nullptr;
    if (g_surfaces.size() >= kMaxSurfaces) {
        std::size_t oldest = g_surfaces.size();
        for (std::size_t i = 0; i < g_surfaces.size(); ++i)
            if (g_surfaces[i].get() != keep && (oldest == g_surfaces.size() || g_surfaces[i]->used < g_surfaces[oldest]->used)) oldest = i;
        retire_surface(oldest);
    }
    auto s = std::make_unique<Surface>();
    s->bp = bp;
    s->bw = bw;
    s->depth = depth;
    s->psm = psm;
    s->width = bw * 64;
    s->height = height;
    WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
    td.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc;
    td.size = WGPUExtent3D{s->width * g_scale, s->height * g_scale, 1};
    td.format = depth ? WGPUTextureFormat_Depth32Float : WGPUTextureFormat_RGBA8Unorm;
    s->texture = wgpuDeviceCreateTexture(g_device, &td);
    s->view = wgpuTextureCreateView(s->texture, nullptr);
    WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
    entry.binding = 0;
    entry.textureView = s->view;
    WGPUBindGroupDescriptor bd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bd.layout = g_target_layout;
    bd.entryCount = 1;
    bd.entries = &entry;
    s->group = wgpuDeviceCreateBindGroup(g_device, &bd);
    s->stale0 = 0;
    s->stale1 = static_cast<int>(height) - 1;  // nothing of GS memory is in it yet
    s->used = ++g_surface_clock;
    g_surfaces.push_back(std::move(s));
    return g_surfaces.back().get();
}

void load_surface(Surface &s) {
    if (s.stale1 < s.stale0) return;
    NativePass pass{s.depth ? nullptr : &s, s.depth ? &s : nullptr, {}};
    pass.draws.push_back({g_load_pipelines[s.depth ? 1 : 0], push_params({s.bp, s.bw, s.psm}), 3, 0, static_cast<std::uint32_t>(s.stale0), s.width,
                          static_cast<std::uint32_t>(s.stale1 - s.stale0 + 1), 0.0f});
    g_commands.push_back({kNative, static_cast<std::uint32_t>(g_native_passes.size()), 0, 0});
    g_native_passes.push_back(std::move(pass));
    TRACE("load %s %X/%u/%02X rows %d-%d\n", s.depth ? "depth" : "color", s.bp, s.bw, s.psm, s.stale0, s.stale1);
    s.stale0 = 0;
    s.stale1 = -1;
    ++g_loads;
}

void plan_decodes(std::uint32_t prims);

// Draws the pending batch with the GPU's rasterizer; false leaves it to the compute rasterizer.
// A primitive's outline, for telling whether two of a feedback draw can cover the same pixel.
struct Outline {
    float x[4], y[4];
    float min_x, max_x, min_y, max_y;
    int corners;
};

Outline outline(const std::uint32_t *rec) {
    Outline o;
    if (rec[1]) {  // a sprite: x, y, width, height
        const float x0 = u2f(rec[4]), y0 = u2f(rec[5]), x1 = x0 + u2f(rec[6]), y1 = y0 + u2f(rec[7]);
        const float xs[4] = {x0, x1, x1, x0}, ys[4] = {y0, y0, y1, y1};
        std::copy(xs, xs + 4, o.x);
        std::copy(ys, ys + 4, o.y);
        o.corners = 4;
    } else {
        for (int i = 0; i < 3; ++i) {
            o.x[i] = u2f(rec[4 + i * 2]);
            o.y[i] = u2f(rec[5 + i * 2]);
        }
        o.corners = 3;
    }
    o.min_x = o.max_x = o.x[0];
    o.min_y = o.max_y = o.y[0];
    for (int i = 1; i < o.corners; ++i) {
        o.min_x = std::min(o.min_x, o.x[i]);
        o.max_x = std::max(o.max_x, o.x[i]);
        o.min_y = std::min(o.min_y, o.y[i]);
        o.max_y = std::max(o.max_y, o.y[i]);
    }
    return o;
}

// Whether the insides of two convex outlines meet (touching along an edge is not meeting: the fill rule gives a
// pixel on a shared edge to one of them). A separating axis is along some edge of one of the two.
bool outlines_meet(const Outline &a, const Outline &b) {
    // AABB is a conservative broad phase: disjoint boxes cannot share a
    // pixel, so avoid expensive separating-axis tests for distant primitives.
    if (a.max_x <= b.min_x || b.max_x <= a.min_x ||
        a.max_y <= b.min_y || b.max_y <= a.min_y) return false;
    for (int pass = 0; pass < 2; ++pass) {
        const Outline &p = pass ? b : a;
        for (int i = 0; i < p.corners; ++i) {
            const int j = (i + 1) % p.corners;
            const float nx = p.y[j] - p.y[i], ny = p.x[i] - p.x[j];
            if (nx == 0.0f && ny == 0.0f) continue;
            float amin = 1e30f, amax = -1e30f, bmin = 1e30f, bmax = -1e30f;
            for (int k = 0; k < a.corners; ++k) {
                const float d = a.x[k] * nx + a.y[k] * ny;
                amin = std::min(amin, d);
                amax = std::max(amax, d);
            }
            for (int k = 0; k < b.corners; ++k) {
                const float d = b.x[k] * nx + b.y[k] * ny;
                bmin = std::min(bmin, d);
                bmax = std::max(bmax, d);
            }
            if (amax <= bmin || bmax <= amin) return false;
        }
    }
    return true;
}

// Where the group of a feedback draw's primitives that starts at `from` ends: before the first one, short of `to`,
// that can cover a pixel one of the group covers.
std::uint32_t feedback_group(std::uint32_t from, std::uint32_t to, const Outline *precomputed = nullptr) {
    // The eligibility prepass already owns contiguous outlines for this run.
    // Compare them in place instead of copying every outline into a second
    // vector for each feedback group. The fallback retains its own workspace.
    static std::vector<Outline> fallback_group;
    if (!precomputed) fallback_group.clear();
    float group_min_x = 0, group_max_x = 0, group_min_y = 0, group_max_y = 0;
    ++g_feedback_group_calls;
    for (std::uint32_t q = from; q < to; ++q) {
        ++g_feedback_group_prims;
        Outline computed;
        if (!precomputed) computed = outline(g_batch.prims.data() + q * kPrimWords);
        const Outline &o = precomputed ? precomputed[q - from] : computed;
        const bool may_overlap = q != from &&
            o.max_x > group_min_x && o.min_x < group_max_x &&
            o.max_y > group_min_y && o.min_y < group_max_y;
        if (may_overlap) {
            for (std::uint32_t prev = from; prev < q; ++prev) {
                const Outline &other = precomputed ? precomputed[prev - from] : fallback_group[prev - from];
                ++g_feedback_outline_tests;
                if (outlines_meet(o, other)) return q;
            }
        }
        if (q == from) {
            group_min_x = o.min_x; group_max_x = o.max_x;
            group_min_y = o.min_y; group_max_y = o.max_y;
        } else {
            group_min_x = std::min(group_min_x, o.min_x);
            group_max_x = std::max(group_max_x, o.max_x);
            group_min_y = std::min(group_min_y, o.min_y);
            group_max_y = std::max(group_max_y, o.max_y);
        }
        if (!precomputed) fallback_group.push_back(o);
    }
    return to;
}

bool flush_native(std::uint32_t prims) {
    const Timed timed(kCpuNative);
    Batch &b = g_batch;
    // Cache the feedback partition computed during native eligibility checking.
    // A zero entry means the range was not evaluated in the prepass.
    static std::vector<std::uint32_t> feedback_ends;
    feedback_ends.assign(prims, 0);
    static std::vector<Outline> run_outlines;  // reused across batches and state runs
    // Draws that read the pixel already there cost a copy of the target and a new pass per group of primitives that
    // keep clear of each other. Where a batch's primitives lie all over each other that is more than handing the
    // batch to the compute rasterizer costs.
    {
        std::uint32_t groups = 0, among = 0;
        for (std::uint32_t i = 0; i < prims;) {
            const std::uint32_t *rec = b.prims.data() + i * kPrimWords;
            std::uint32_t j = i + 1;
            while (j < prims && b.prims[j * kPrimWords] == rec[0] && b.prims[j * kPrimWords + 1] == rec[1]) ++j;
            NativeState state;
            if (native_state(b.setups.data() + rec[0] * kSetupWords, b.zte, state) && state.feedback && state.draws) {
                among += j - i;
                // Build outlines once for the entire state run. The previous
                // partitioner rebuilt outlines for each new feedback group.
                run_outlines.clear();
                run_outlines.reserve(j - i);
                for (std::uint32_t q = i; q < j; ++q)
                    run_outlines.push_back(outline(b.prims.data() + q * kPrimWords));
                for (std::uint32_t from = i; from < j;) {
                    const std::uint32_t end = feedback_group(from, j, run_outlines.data() + (from - i));
                    feedback_ends[from] = end;
                    from = end;
                    ++groups;
                    // Even if every remaining primitive requires feedback,
                    // the allowed group budget cannot catch up. Fall back
                    // immediately instead of completing costly SAT grouping.
                    if (groups > 16u + (among + (prims - j)) / 4u) {
                        g_why_prims[kWhyCount - 3] += prims;
                        ++g_why_batches[kWhyCount - 3];
                        g_refusal_counted = true;
                        return false;
                    }
                }
            }
            i = j;
        }
        if (groups > 16 + among / 4) {
            g_why_prims[kWhyCount - 3] += prims;
            ++g_why_batches[kWhyCount - 3];
            g_refusal_counted = true;
            return false;
        }
    }
    const std::uint32_t rows = static_cast<std::uint32_t>(b.maxy) + 1;
    Surface *color = surface_for(b.fbp, b.fbw, false, PSMCT32, rows, nullptr);
    if (!color) return false;
    Surface *depth = b.zte ? surface_for(b.zbp, b.fbw, true, b.zpsm, rows, color) : nullptr;
    if (b.zte && !depth) return false;
    if (depth && depth->height != color->height) {
        // A pass's attachments are of one size: the shorter is made again as tall as the other.
        const std::uint32_t tall = std::max(color->height, depth->height);
        color = surface_for(b.fbp, b.fbw, false, PSMCT32, tall, depth);
        depth = color ? surface_for(b.zbp, b.fbw, true, b.zpsm, tall, color) : nullptr;
        if (!color || !depth || depth->height != color->height) return false;
    }
    const Range fb = surface_range(*color, b.miny, b.maxy);
    const Range z = depth ? surface_range(*depth, b.miny, b.maxy) : Range{};
    if (depth && fb.overlaps(z.lo, z.hi)) return false;
    // The textures are read from GS memory as it is before the batch: whichever target holds them newer writes them
    // back, the batch's own included (a draw reading the frame it draws to).
    g_sync_cause = 1;
    for (const Range &t : g_batch_reads) sync_memory(t, false);
    g_sync_cause = 0;
    sync_memory(fb, true, color);
    if (depth) sync_memory(z, b.zwrite, depth);
    // PS2_NATIVE_SYNC=1 reloads the targets before every batch and writes them back after it: slow, and what the
    // picture should be if the tracking of who holds the newer pixels were at fault.
    static const bool paranoid = std::getenv("PS2_NATIVE_SYNC") != nullptr;
    if (paranoid && std::atoi(std::getenv("PS2_NATIVE_SYNC")) & 1) {
        color->stale0 = 0;
        color->stale1 = static_cast<int>(color->height) - 1;
        if (depth) depth->stale0 = 0, depth->stale1 = static_cast<int>(depth->height) - 1;
    }
    load_surface(*color);
    if (depth) load_surface(*depth);
    plan_decodes(prims);
    const std::uint32_t prim_offset = arena_append(b.prims.data(), b.prims.size() * 4);
    const std::uint32_t setup_offset = arena_append(b.setups.data(), b.setups.size() * 4);
    for (const Decode &d : g_decodes) dispatch_linear(kDecode, d.words, push_params({setup_offset + d.setup * kSetupWords, d.offset, d.words}));
    NativePass pass{color, depth, {}};
    if (native_debug()) std::fprintf(stderr, "[batch] %llu: %u prims, fbp %X zbp %X\n", static_cast<unsigned long long>(g_native_batches), prims, b.fbp, b.zbp);
    for (std::uint32_t i = 0; i < prims;) {
        const std::uint32_t *rec = b.prims.data() + i * kPrimWords;
        std::uint32_t j = i + 1;
        while (j < prims && b.prims[j * kPrimWords] == rec[0] && b.prims[j * kPrimWords + 1] == rec[1]) ++j;
        const std::uint32_t *w = b.setups.data() + rec[0] * kSetupWords;
        NativeState state;
        native_state(w, depth != nullptr, state);
        const std::uint32_t x0 = std::min(w[10] & 0xFFFFu, color->width), x1 = std::min((w[10] >> 16) + 1, color->width);
        const std::uint32_t y0 = std::min(w[11] & 0xFFFFu, color->height), y1 = std::min((w[11] >> 16) + 1, color->height);
        if (native_debug()) for (std::uint32_t q = i; q < j; ++q) std::fprintf(stderr, "[prim] %u,%u-%u,%u z %06X tex %04X/%u/%02X %ux%u clamp %X dec %X\n", b.prims[q * kPrimWords + 2] & 0xFFFF, b.prims[q * kPrimWords + 2] >> 16, b.prims[q * kPrimWords + 3] & 0xFFFF, b.prims[q * kPrimWords + 3] >> 16, b.prims[q * kPrimWords + 24], w[12], w[13], w[14], 1u << (w[15] & 0xFF), 1u << (w[15] >> 8), (w[7] >> 24) & 0xF, w[24]);
        if (native_debug()) std::fprintf(stderr, "[native] prims %u..%u kind %u key %05X flags %05X modes %08X fpsm %02X tpsm %02X sc %u,%u-%u,%u box %d,%d-%d,%d draws %d\n", i, j, rec[1], state.key, w[0], w[7], w[3], w[14], x0, y0, x1, y1, b.minx, b.miny, b.maxx, b.maxy, state.draws);
        if (state.draws && x0 < x1 && y0 < y1) {
            // One state run uses the same pipeline for every feedback group.
            const WGPURenderPipeline pipeline = native_pipeline(state.key);
            std::array<std::uint32_t, kParamStride / 4> record{};
            std::memcpy(record.data(), w, kSetupWords * 4);
            record[0] = state.flags;
            record[32] = prim_offset + i * kPrimWords;
            record[33] = rec[1];
            record[34] = color->width;
            record[35] = color->height;
            record[36] = state.premultiply;
            if (!state.feedback) {
                g_param_records.push_back(record);
                pass.draws.push_back({pipeline, static_cast<std::uint32_t>(g_param_records.size() - 1), (j - i) * (rec[1] ? 6u : 3u), x0, y0, x1 - x0,
                                      y1 - y0, state.constant});
            } else {
                // The draw's primitives go in groups of which no two can cover the same pixel: each group reads the
                // target as the groups before it left it, copied to the backdrop over the group's box.
                if (!color->backdrop) {
                    WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
                    td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
                    td.size = WGPUExtent3D{color->width * g_scale, color->height * g_scale, 1};
                    td.format = WGPUTextureFormat_RGBA8Unorm;
                    color->backdrop = wgpuDeviceCreateTexture(g_device, &td);
                    color->backdrop_view = wgpuTextureCreateView(color->backdrop, nullptr);
                    WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
                    entry.binding = 0;
                    entry.textureView = color->backdrop_view;
                    WGPUBindGroupDescriptor bd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
                    bd.layout = g_backdrop_layout;
                    bd.entryCount = 1;
                    bd.entries = &entry;
                    color->backdrop_group = wgpuDeviceCreateBindGroup(g_device, &bd);
                }
                std::uint32_t from = i;
                while (from < j) {
                    int gx0 = 1 << 30, gy0 = 1 << 30, gx1 = -1, gy1 = -1;
                    // Reuse the exact partition validated above; preserve the
                    // original calculation as a fallback for unplanned ranges.
                    const std::uint32_t cached_end = feedback_ends[from];
                    const std::uint32_t to = cached_end && cached_end <= j
                        ? cached_end : feedback_group(from, j);
                    for (std::uint32_t q = from; q < to; ++q) {
                        const std::uint32_t *prim = b.prims.data() + q * kPrimWords;
                        gx0 = std::min(gx0, static_cast<int>(prim[2] & 0xFFFFu));
                        gy0 = std::min(gy0, static_cast<int>(prim[2] >> 16));
                        gx1 = std::max(gx1, static_cast<int>(prim[3] & 0xFFFFu));
                        gy1 = std::max(gy1, static_cast<int>(prim[3] >> 16));
                    }
                    // The copy over the group's box inside the scissor rectangle; nothing is drawn outside it.
                    const std::uint32_t cx0 = std::max(static_cast<std::uint32_t>(std::max(gx0, 0)), x0), cy0 = std::max(static_cast<std::uint32_t>(std::max(gy0, 0)), y0);
                    const std::uint32_t cx1 = std::min(static_cast<std::uint32_t>(gx1 + 1), x1), cy1 = std::min(static_cast<std::uint32_t>(gy1 + 1), y1);
                    if (gx1 >= 0 && cx0 < cx1 && cy0 < cy1) {
                        record[32] = prim_offset + from * kPrimWords;
                        g_param_records.push_back(record);
                        pass.draws.push_back({nullptr, 0, 0, cx0, cy0, cx1 - cx0, cy1 - cy0, 0.0f, false});
                        // The feedback group's bounding box already encloses every
                        // primitive in this draw and is clipped to the setup
                        // scissor. Use it for rasterization as well as copy.
                        pass.draws.push_back({pipeline, static_cast<std::uint32_t>(g_param_records.size() - 1), (to - from) * (rec[1] ? 6u : 3u), cx0, cy0, cx1 - cx0,
                                              cy1 - cy0, 0.0f, true});
                        ++g_feedback_copies;
                        g_feedback_copy_pixels += static_cast<std::uint64_t>(cx1 - cx0) * (cy1 - cy0);
                        g_feedback_draws += to - from;
                    }
                    from = to;
                }
            }
        }
        i = j;
    }
    g_native_draws += pass.draws.size();
    TRACE("native %u prims %zu draws %zu decodes fb %X/%u/%02X z %X/%02X%s box %d,%d-%d,%d reads %zu\n", prims, pass.draws.size(), g_decodes.size(), b.fbp, b.fbw, b.fpsm, b.zbp, b.zpsm,
          b.zte ? (b.zwrite ? " zw" : " zt") : " -", b.minx, b.miny, b.maxx, b.maxy, g_batch_reads.size());
    if (!pass.draws.empty()) {
        g_commands.push_back({kNative, static_cast<std::uint32_t>(g_native_passes.size()), 0, 0});
        g_native_passes.push_back(std::move(pass));
    }
    const auto touched = [&](Surface &s) {
        const int x1 = std::min(b.maxx, static_cast<int>(s.width) - 1);
        s.dx0 = s.dirty ? std::min(s.dx0, b.minx) : b.minx;
        s.dy0 = s.dirty ? std::min(s.dy0, b.miny) : b.miny;
        s.dx1 = s.dirty ? std::max(s.dx1, x1) : x1;
        s.dy1 = s.dirty ? std::max(s.dy1, b.maxy) : b.maxy;
        s.dirty = true;
    };
    touched(*color);
    if (depth && b.zwrite) touched(*depth);
    if (paranoid && std::atoi(std::getenv("PS2_NATIVE_SYNC")) & 2) sync_all();
    return true;
}

// PS2_GPU_CENSUS=1 (or a number of seconds): every ten seconds, which draw states the game used and how much screen each covered. It is the
// shopping list for a renderer that draws with the GPU's own rasterizer: the states at the top are the ones worth
// supporting first.
struct CensusKey {
    std::uint32_t kind, flags, fpsm, zpsm, test, tpsm, masked;
    bool operator<(const CensusKey &o) const {
        return std::tie(kind, flags, fpsm, zpsm, test, tpsm, masked) < std::tie(o.kind, o.flags, o.fpsm, o.zpsm, o.test, o.tpsm, o.masked);
    }
};
struct CensusCount {
    std::uint64_t prims = 0, area = 0;
};
std::map<CensusKey, CensusCount> g_census;
std::uint64_t g_census_batches = 0, g_census_feedback = 0, g_census_prims = 0, g_census_area = 0;
std::chrono::steady_clock::time_point g_census_since = std::chrono::steady_clock::now();

void census(std::uint32_t prims, bool feedback) {
    static const char *const spec = std::getenv("PS2_GPU_CENSUS");
    if (!spec) return;
    static const double interval = std::atof(spec) > 1.0 ? std::atof(spec) : 10.0;  // PS2_GPU_CENSUS=<seconds>, 10 by default
    const Batch &b = g_batch;
    ++g_census_batches;
    g_census_feedback += feedback ? 1 : 0;
    for (std::uint32_t i = 0; i < prims; ++i) {
        const std::uint32_t *rec = b.prims.data() + i * kPrimWords;
        const std::uint32_t *w = b.setups.data() + rec[0] * kSetupWords;
        const bool textured = (w[0] & 2u) != 0;
        // Without a texture the texture bits of the state say nothing; without blending the blend equation does not either.
        std::uint32_t test = w[7] & ~(3u << 28);
        if (!textured) test &= ~((3u << 20) | (0xFu << 24));
        if (!(w[0] & 8u)) test &= ~(0xFFu << 12);
        if (!(w[0] & 64u)) test &= ~0x37u;
        if (!(w[0] & 512u)) test &= ~(3u << 8);
        std::uint32_t flags = w[0];
        if (!textured) flags &= ~(16u | 8192u | 16384u | 32768u);
        const CensusKey key{rec[1], flags, w[3], w[6], test, textured ? w[14] : 0xFFu, w[4] ? 1u : 0u};
        const std::uint64_t width = (rec[3] & 0xFFFFu) - (rec[2] & 0xFFFFu) + 1, height = (rec[3] >> 16) - (rec[2] >> 16) + 1;
        CensusCount &c = g_census[key];
        ++c.prims;
        c.area += width * height;
        ++g_census_prims;
        g_census_area += width * height;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_census_since).count();
    if (seconds < interval) return;
    std::vector<std::pair<CensusKey, CensusCount>> rows(g_census.begin(), g_census.end());
    std::sort(rows.begin(), rows.end(), [](const auto &x, const auto &y) { return x.second.area > y.second.area; });
    std::fprintf(stderr, "[census] %.0f s: %llu batches (%llu read their own target), %llu prims, %llu states; box area %.1f Mpx\n", seconds,
                 (unsigned long long)g_census_batches, (unsigned long long)g_census_feedback, (unsigned long long)g_census_prims,
                 (unsigned long long)rows.size(), static_cast<double>(g_census_area) / 1e6);
    static const char *const kAtst[] = {"never", "always", "<", "<=", "==", ">=", ">", "!="};
    static const char *const kAfail[] = {"keep", "fb", "zb", "rgb"};
    static const char *const kZtst[] = {"never", "always", ">=", ">"};
    static const char *const kTfx[] = {"modulate", "decal", "highlight", "highlight2"};
    for (std::size_t i = 0; i < rows.size() && i < 40; ++i) {
        const CensusKey &k = rows[i].first;
        char text[256];
        int n = std::snprintf(text, sizeof text, "kind %u fb %02X%s z %02X%s", k.kind, k.fpsm, k.masked ? " masked" : "", k.zpsm, (k.flags & 32u) ? " nowrite" : "");
        if (k.flags & 512u) n += std::snprintf(text + n, sizeof text - n, " ztst %s", kZtst[(k.test >> 8) & 3u]);
        if (k.flags & 64u) n += std::snprintf(text + n, sizeof text - n, " atst %s fail %s", kAtst[k.test & 7u], kAfail[(k.test >> 4) & 3u]);
        if (k.flags & 128u) n += std::snprintf(text + n, sizeof text - n, " date%u", (k.flags >> 8) & 1u);
        if (k.flags & 8u) {
            const std::uint32_t a = (k.test >> 12) & 0xFFu;
            n += std::snprintf(text + n, sizeof text - n, " blend (%u-%u)*%u+%u%s", a & 3u, (a >> 2) & 3u, (a >> 4) & 3u, (a >> 6) & 3u, (k.flags & 1024u) ? " pabe" : "");
        }
        if (k.flags & 2u)
            n += std::snprintf(text + n, sizeof text - n, " tex %02X %s%s%s%s clamp %X", k.tpsm, kTfx[(k.test >> 20) & 3u], (k.flags & 8192u) ? " tcc" : "",
                               (k.flags & 16384u) ? " linear" : "", (k.flags & 16u) ? " fst" : "", (k.test >> 24) & 0xFu);
        if (k.flags & 4u) n += std::snprintf(text + n, sizeof text - n, " fog");
        if (k.flags & 1u) n += std::snprintf(text + n, sizeof text - n, " gouraud");
        if (!(k.flags & 2048u)) n += std::snprintf(text + n, sizeof text - n, " wrapcolor");
        if (k.flags & 4096u) n += std::snprintf(text + n, sizeof text - n, " fba");
        std::fprintf(stderr, "[census] %5.1f%% area %7llu prims: %s\n", 100.0 * static_cast<double>(rows[i].second.area) / static_cast<double>(std::max<std::uint64_t>(g_census_area, 1)),
                     (unsigned long long)rows[i].second.prims, text);
    }
    g_census.clear();
    g_census_batches = g_census_feedback = g_census_prims = g_census_area = 0;
    g_census_since = std::chrono::steady_clock::now();
}

// The pending batch through the compute rasterizer.
void raster_batch(std::uint32_t prims) {
    const Timed timed(kCpuCompute);
    Batch &b = g_batch;
    ++g_compute_batches;
    if (g_trace > 0) {
        const std::uint32_t *w = b.setups.data() + b.prims[0] * kSetupWords;
        TRACE("compute %u prims kind %u flags %05X modes %08X fb %X/%u/%02X msk %08X z %X/%02X tex %X/%u/%02X box %d,%d-%d,%d reads %zu\n", prims, b.prims[1], w[0], w[7], b.fbp, b.fbw, b.fpsm, w[4],
              b.zbp, b.zpsm, w[12], w[13], w[14], b.minx, b.miny, b.maxx, b.maxy, g_batch_reads.size());
        // The first few batches of GT4's colour correction (alpha masked) in full, to work out what it computes.
        static int dumps = 0;
        if (w[4] == 0xFF000000u && prims <= 64 && dumps < 8) {
            ++dumps;
            for (std::size_t i = 0; i < b.setups.size(); ++i) std::fprintf(stderr, "%s%08X%s", i % kSetupWords ? " " : "[trace] setup ", b.setups[i], (i + 1) % kSetupWords ? "" : "\n");
            for (std::size_t i = 0; i < b.prims.size(); ++i) std::fprintf(stderr, "%s%08X%s", i % kPrimWords ? " " : "[trace] prim ", b.prims[i], (i + 1) % kPrimWords ? "" : "\n");
        }
    }
    g_compute_prims += prims;
    if (!g_surfaces.empty()) {
        // It reads and writes GS memory itself: native targets under it go back there first.
        g_sync_cause = 2;
        sync_memory(pages(b.fbp, b.fbw, b.fpsm, b.minx, b.miny, b.maxx, b.maxy), true);
        if (b.zte) sync_memory(pages(b.zbp, b.fbw, b.zpsm, b.minx, b.miny, b.maxx, b.maxy), b.zwrite);
        g_sync_cause = 3;
        for (const Range &r : g_batch_reads) sync_memory(r, false);
        g_sync_cause = 0;
    }
    g_touched.clear();
    g_spans.clear();
    g_span_start.resize(prims + 1);
    for (std::uint32_t i = 0; i < prims; ++i) {
        g_span_start[i] = static_cast<std::uint32_t>(g_spans.size());
        add_spans(b.prims.data() + i * kPrimWords, b.boxes[i * 2], b.boxes[i * 2 + 1]);
        for (std::size_t k = g_span_start[i]; k < g_spans.size(); ++k) {
            const Span &sp = g_spans[k];
            std::uint32_t t = static_cast<std::uint32_t>(sp.ty * g_tiles_per_row + sp.tx0);
            for (int tx = sp.tx0; tx <= sp.tx1; ++tx, ++t)
                if (!g_tile_count[t]++) g_touched.push_back(t);
        }
    }
    g_span_start[prims] = static_cast<std::uint32_t>(g_spans.size());
    plan_decodes(prims);
    const std::uint32_t prim_offset = arena_append(b.prims.data(), b.prims.size() * 4);
    const std::uint32_t setup_offset = arena_append(b.setups.data(), b.setups.size() * 4);
    for (const Decode &d : g_decodes) dispatch_linear(kDecode, d.words, push_params({setup_offset + d.setup * kSetupWords, d.offset, d.words}));
    std::uint32_t total = 0;
    for (std::uint32_t t : g_touched) total += g_tile_count[t];
    const std::uint32_t tiles = static_cast<std::uint32_t>(g_touched.size());
    const std::uint32_t tile_offset = static_cast<std::uint32_t>(g_arena.size());
    const std::uint32_t list_offset = tile_offset + tiles * 2 + 2;
    const std::uint32_t list_words = (total + 1) / 2;
    std::uint32_t *grown = g_arena.grow(tiles * 2 + 2 + list_words);
    if (list_words) grown[tiles * 2 + 2 + list_words - 1] = 0;
    std::uint32_t cursor = 0;
    for (std::uint32_t i = 0; i < tiles; ++i) {
        const std::uint32_t t = g_touched[i];
        grown[i * 2] = (t % static_cast<std::uint32_t>(g_tiles_per_row)) | ((t / static_cast<std::uint32_t>(g_tiles_per_row)) << 16);
        grown[i * 2 + 1] = cursor;
        g_tile_fill[t] = cursor;
        cursor += g_tile_count[t];
    }
    grown[tiles * 2] = 0;
    grown[tiles * 2 + 1] = cursor;
    auto *halves = reinterpret_cast<unsigned char *>(grown + tiles * 2 + 2);
    for (std::uint32_t i = 0; i < prims; ++i) {
        const auto index = static_cast<std::uint16_t>(i);
        for (std::uint32_t k = g_span_start[i]; k < g_span_start[i + 1]; ++k) {
            const Span &sp = g_spans[k];
            std::uint32_t t = static_cast<std::uint32_t>(sp.ty * g_tiles_per_row + sp.tx0);
            for (int tx = sp.tx0; tx <= sp.tx1; ++tx, ++t) std::memcpy(halves + 2u * g_tile_fill[t]++, &index, 2);
        }
    }
    for (std::uint32_t t : g_touched) g_tile_count[t] = 0;
    std::uint32_t snapshot_lo, snapshot_hi, snapshot_stride, snapshot_run;
    const std::uint32_t snapshot = snapshot_batch(prims, snapshot_lo, snapshot_hi, snapshot_stride, snapshot_run);
    const std::uint32_t param = push_params({tile_offset, prim_offset, setup_offset, tiles, list_offset, snapshot_lo, snapshot_hi, snapshot_stride, snapshot_run});
    note_raster(prims, tiles, total, snapshot);
    census(prims, snapshot != 0);
    g_commands.push_back({kRaster, std::min(tiles, 1024u), (tiles + 1023) / 1024, param});
}

void emit_pre(std::size_t upto = ~static_cast<std::size_t>(0));

// ---- GT4's colour correction ----
//
// After a race frame is drawn GT4 puts every pixel's red, green and blue through a 256-entry table. It has the GS do
// it, a 64x32 tile (one page of the frame) at a time, 70 tiles a frame:
//   1. 48 sprites read the tile's page as an 8-bit texture, a byte of a pixel at a time, with the table as CLUT,
//      into a scratch page: 16 sprites for red, then 16 adding blue and 16 adding green.
//   2. 8 sprites copy the scratch page back over the tile, each through an eighth of it as a CLUT and a texture that
//      counts (kLutCount), which undoes the order step 1 left the pixels in.
// That is 140 batches and 630 CLUT loads a frame, each reading what the one before wrote. Here the two steps are
// recognized, exactly as traced on hardware and nothing looser, and not drawn: the pages they cover are collected
// and the table is applied to them in one dispatch (lut_frame in gs.wgsl) when something else comes along. The
// scratch page is left as the game's last step 1 would have left it (lut_scratch). tests/gs_gt4_lut_test.cpp draws
// the real sequence both ways. PS2_GT4_LUT=0 turns this off.
const std::uint8_t kLutCount[256] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x0C, 0x0D, 0x0E, 0x0F, 0x08, 0x09, 0x0A, 0x0B, 0x1C, 0x1D, 0x1E, 0x1F, 0x18, 0x19, 0x1A, 0x1B,
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
    0x2C, 0x2D, 0x2E, 0x2F, 0x28, 0x29, 0x2A, 0x2B, 0x3C, 0x3D, 0x3E, 0x3F, 0x38, 0x39, 0x3A, 0x3B,
    0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57,
    0x4C, 0x4D, 0x4E, 0x4F, 0x48, 0x49, 0x4A, 0x4B, 0x5C, 0x5D, 0x5E, 0x5F, 0x58, 0x59, 0x5A, 0x5B,
    0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77,
    0x6C, 0x6D, 0x6E, 0x6F, 0x68, 0x69, 0x6A, 0x6B, 0x7C, 0x7D, 0x7E, 0x7F, 0x78, 0x79, 0x7A, 0x7B,
    0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x8C, 0x8D, 0x8E, 0x8F, 0x88, 0x89, 0x8A, 0x8B, 0x9C, 0x9D, 0x9E, 0x9F, 0x98, 0x99, 0x9A, 0x9B,
    0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7,
    0xAC, 0xAD, 0xAE, 0xAF, 0xA8, 0xA9, 0xAA, 0xAB, 0xBC, 0xBD, 0xBE, 0xBF, 0xB8, 0xB9, 0xBA, 0xBB,
    0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7,
    0xCC, 0xCD, 0xCE, 0xCF, 0xC8, 0xC9, 0xCA, 0xCB, 0xDC, 0xDD, 0xDE, 0xDF, 0xD8, 0xD9, 0xDA, 0xDB,
    0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7,
    0xEC, 0xED, 0xEE, 0xEF, 0xE8, 0xE9, 0xEA, 0xEB, 0xFC, 0xFD, 0xFE, 0xFF, 0xF8, 0xF9, 0xFA, 0xFB,
};
// Step 2's eight CLUTs, in blocks from the scratch page's start.
const std::uint32_t kLutClut[8] = {0x00, 0x04, 0x10, 0x14, 0x08, 0x0C, 0x18, 0x1C};

struct Lut {
    bool count_ok = false;  // the counting texture is in GS memory, at count_bp
    std::uint32_t count_bp = 0;
    Range count_range;
    bool half = false;  // a step 1 was taken and its step 2 has not come
    std::uint32_t page = 0, scratch = 0, scratch_bw = 0, slot = 0, table = 0;  // of the latest step 1
    std::size_t pre = 0;                                                       // uploads and CLUT loads queued before it
    bool any = false;  // pages are waiting for the table
    std::uint32_t fbp = 0, fbw = 0, mask[8] = {};
    int minx = 0, miny = 0, maxx = 0, maxy = 0;
} g_lut;
std::uint64_t g_lut_tiles = 0, g_lut_passes = 0;

void clear_batch() {
    Batch &b = g_batch;
    b.prims.clear();
    b.setups.clear();
    b.boxes.clear();
    g_batch_reads.clear();
    b.key = ~0ull;
    b.minx = b.miny = 1 << 30;
    b.maxx = b.maxy = -1;
}

// Is the pending batch step 1, and of which page?
bool lut_step1(std::uint32_t &page, std::uint32_t &scratch, std::uint32_t &scratch_bw, std::uint32_t &slot) {
    const Batch &b = g_batch;
    if (b.prims.size() != 48 * kPrimWords) return false;
    static const struct { std::uint32_t color; float u, v; bool blend, region; } kChannel[3] = {{0x00000080u, 0, 0, false, false}, {0x00800000u, 128, 0, true, false}, {0x00008000u, 64, 32, true, true}};
    for (std::uint32_t n = 0; n < 48; ++n) {
        const std::uint32_t *rec = b.prims.data() + n * kPrimWords;
        const std::uint32_t *w = b.setups.data() + rec[0] * kSetupWords;
        const auto &c = kChannel[n / 16];
        const std::uint32_t i = n % 16;
        // The state: a 32-bit scratch page with its alpha masked, the page as a 128x64 8-bit texture, modulated,
        // point sampled, nothing tested; red plain, blue and green added.
        const std::uint32_t flags = w[0], want = 2u | 16u | 2048u | 8192u | (c.blend ? 8u : 0u);
        if ((flags & (2u | 4u | 8u | 16u | 64u | 128u | 512u | 1024u | 2048u | 4096u | 8192u | 16384u)) != want) return false;
        if (w[3] != PSMCT32 || w[4] != 0xFF000000u || w[1] % 32u || w[12] % 32u || w[13] != 2 || w[14] != PSMT8 || w[15] != 0x0607u) return false;
        if (((w[7] >> 20) & 3u) != 0 || ((w[7] >> 28) & 3u) != 0) return false;
        if (c.blend && (((w[7] >> 12) & 0xFFu) != 0x68u || w[9] != 0x80u)) return false;
        if (((w[7] >> 24) & 0xFu) != (c.region ? 3u : 0u) || (c.region && w[18] != 0x3F7u)) return false;
        if ((w[10] & 0xFFFFu) != 0 || (w[10] >> 16) < 63 || (w[11] & 0xFFFFu) != 0 || (w[11] >> 16) < 31) return false;
        if (n == 0) {
            page = w[12];
            scratch = w[1];
            scratch_bw = w[2];
            slot = w[22];
        } else if (w[12] != page || w[1] != scratch || w[2] != scratch_bw || w[22] != slot) {
            return false;
        }
        // The sprite: 8x16 pixels of the scratch page from 8 texels by 64, which is every fourth row of bytes.
        const float u = c.u + static_cast<float>(i / 2) * 256.0f, v = c.v + static_cast<float>(i % 2) * 16.0f;
        const float expect[4] = {static_cast<float>((i % 8) * 8), static_cast<float>((i / 8) * 16), 8.0f, 16.0f};
        if (rec[1] != 1 || std::memcmp(rec + 4, expect, sizeof expect) != 0 || rec[13] != c.color) return false;
        if (u2f(rec[15]) != u || u2f(rec[16]) != v || u2f(rec[18]) != u + 128.0f || u2f(rec[19]) != v + 1024.0f) return false;
    }
    return g_clut_source[slot].plain;
}

// Is the pending batch step 2 of the step 1 taken, and over which page of the frame?
bool lut_step2(std::uint32_t &tile) {
    const Batch &b = g_batch;
    if (b.prims.size() != 8 * kPrimWords || !g_lut.count_ok) return false;
    const std::uint32_t *first = b.setups.data() + b.prims[0] * kSetupWords;
    const std::uint32_t fbp = first[1], fbw = first[2];
    if (b.minx % 64 || b.miny % 32 || b.maxx != b.minx + 63 || b.maxy != b.miny + 31 || !fbw) return false;
    tile = static_cast<std::uint32_t>(b.minx / 64) + static_cast<std::uint32_t>(b.miny / 32) * fbw;
    if (tile >= 256 || fbp + tile * 32u != g_lut.page) return false;
    for (std::uint32_t k = 0; k < 8; ++k) {
        const std::uint32_t *rec = b.prims.data() + k * kPrimWords;
        const std::uint32_t *w = b.setups.data() + rec[0] * kSetupWords;
        // The state: the 24-bit frame, the counting texture, decal, point sampled, nothing tested or blended.
        if ((w[0] & (2u | 4u | 8u | 16u | 64u | 128u | 512u | 1024u | 2048u | 4096u | 8192u | 16384u)) != (2u | 16u | 2048u | 8192u)) return false;
        if (w[3] != PSMCT24 || w[4] != 0xFF000000u || w[1] != fbp || w[2] != fbw) return false;
        if (w[12] != g_lut.count_bp || w[13] != 1 || w[14] != PSMT8 || w[15] != 0x0503u) return false;
        if (((w[7] >> 20) & 3u) != 1 || ((w[7] >> 24) & 0xFu) != 0 || ((w[7] >> 28) & 3u) != 0) return false;
        if (static_cast<int>(w[10] & 0xFFFFu) > b.minx || static_cast<int>(w[10] >> 16) < b.maxx || static_cast<int>(w[11] & 0xFFFFu) > b.miny || static_cast<int>(w[11] >> 16) < b.maxy) return false;
        const ClutSource &clut = g_clut_source[w[22]];
        if (!clut.plain || clut.cbp != g_lut.scratch + kLutClut[k]) return false;
        // The sprite: strip k of the tile, 8x32 pixels from the texture's 8x32.
        const float expect[4] = {static_cast<float>(b.minx + static_cast<int>(k) * 8), static_cast<float>(b.miny), 8.0f, 32.0f};
        if (rec[1] != 1 || std::memcmp(rec + 4, expect, sizeof expect) != 0) return false;
        if (u2f(rec[15]) != 0.0f || u2f(rec[16]) != 0.0f || u2f(rec[18]) != 128.0f || u2f(rec[19]) != 512.0f) return false;
    }
    return true;
}

void lut_flush();

// Takes the pending batch if it is the next step of the colour correction; true when it did.
bool lut_try() {
    static const bool on = !std::getenv("PS2_GT4_LUT") || std::atoi(std::getenv("PS2_GT4_LUT")) != 0;
    if (!on || g_scale != 1 || g_batch.prims.empty()) return false;
    std::uint32_t page = 0, scratch = 0, scratch_bw = 0, slot = 0, tile = 0;
    if (!g_lut.half && lut_step1(page, scratch, scratch_bw, slot)) {
        // Every step 1 of a run has to use the same table: the same CLUT, which nothing can have changed since the
        // run began (an upload ends it).
        if (g_lut.any && (g_clut_source[slot].cbp != g_lut.table || scratch != g_lut.scratch)) return false;
        // A page whose lookup is still to be done, read again: the lookup first.
        if (g_lut.any && page >= g_lut.fbp && (page - g_lut.fbp) / 32u < 256u && ((g_lut.mask[(page - g_lut.fbp) / 32u / 32u] >> ((page - g_lut.fbp) / 32u % 32u)) & 1u)) lut_flush();
        g_lut.half = true;
        g_lut.page = page;
        g_lut.scratch = scratch;
        g_lut.scratch_bw = scratch_bw;
        g_lut.slot = slot;
        g_lut.table = g_clut_source[slot].cbp;
        g_lut.pre = g_pre.size();
        wrote(pages(scratch, scratch_bw, PSMCT32, 0, 0, 63, 31));  // as the step would have
        clear_batch();
        return true;
    }
    if (g_lut.half && lut_step2(tile)) {
        const std::uint32_t *w = g_batch.setups.data() + g_batch.prims[0] * kSetupWords;
        if (g_lut.any && (w[1] != g_lut.fbp || w[2] != g_lut.fbw)) return false;
        g_lut.minx = g_lut.any ? std::min(g_lut.minx, g_batch.minx) : g_batch.minx;
        g_lut.miny = g_lut.any ? std::min(g_lut.miny, g_batch.miny) : g_batch.miny;
        g_lut.maxx = g_lut.any ? std::max(g_lut.maxx, g_batch.maxx) : g_batch.maxx;
        g_lut.maxy = g_lut.any ? std::max(g_lut.maxy, g_batch.maxy) : g_batch.maxy;
        g_lut.any = true;
        g_lut.half = false;
        g_lut.fbp = w[1];
        g_lut.fbw = w[2];
        g_lut.mask[tile / 32] |= 1u << (tile % 32);
        wrote(pages(g_lut.page, 1, PSMCT32, 0, 0, 63, 31));
        ++g_lut_tiles;
        clear_batch();
        return true;
    }
    return false;
}

// Does what the steps taken so far come to, before anything else touches GS memory.
void lut_flush() {
    if (!g_lut.any && !g_lut.half) return;
    // The table's CLUT load is among what was queued before the latest step 1; what was queued after it (the eight
    // loads from the scratch page) has to see what that step wrote.
    emit_pre(g_lut.half ? g_lut.pre : ~static_cast<std::size_t>(0));
    const auto scratch = [&] {
        sync_memory(pages(g_lut.page, 1, PSMCT32, 0, 0, 63, 31), false);
        sync_memory(pages(g_lut.scratch, g_lut.scratch_bw, PSMCT32, 0, 0, 63, 31), true);
        std::uint32_t where[256];
        for (std::uint32_t i = 0; i < 256; ++i) where[kLutCount[i]] = (i % 8) | ((i / 8) << 8);
        const std::uint32_t offset = arena_append(where, sizeof where);
        g_commands.push_back({kLutScratch, 8, 4, push_params({g_lut.scratch, g_lut.scratch_bw, g_lut.page, g_lut.slot, offset})});
    };
    // The scratch page holds the latest step 1's pixels: those of a page before its lookup, if that page's step 2
    // came, so this goes first.
    if (g_lut.any && !g_lut.half) scratch();
    if (g_lut.any) {
        sync_memory(pages(g_lut.fbp, g_lut.fbw, PSMCT32, g_lut.minx, g_lut.miny, g_lut.maxx, g_lut.maxy), true);
        const std::uint32_t w = static_cast<std::uint32_t>(g_lut.maxx - g_lut.minx + 1), h = static_cast<std::uint32_t>(g_lut.maxy - g_lut.miny + 1);
        g_commands.push_back({kLutFrame, (w + 7) / 8, (h + 7) / 8,
                              push_params({g_lut.fbp, g_lut.fbw, static_cast<std::uint32_t>(g_lut.minx), static_cast<std::uint32_t>(g_lut.miny), w, h, g_lut.slot, 0, g_lut.mask[0], g_lut.mask[1],
                                           g_lut.mask[2], g_lut.mask[3], g_lut.mask[4], g_lut.mask[5], g_lut.mask[6], g_lut.mask[7]})});
        ++g_lut_passes;
        TRACE("colour correction over %d,%d-%d,%d of %X/%u\n", g_lut.minx, g_lut.miny, g_lut.maxx, g_lut.maxy, g_lut.fbp, g_lut.fbw);
    }
    if (g_lut.half) scratch();
    g_lut.any = g_lut.half = false;
    std::memset(g_lut.mask, 0, sizeof g_lut.mask);
}

void flush_batch() {
    Batch &b = g_batch;
    if (b.prims.empty()) {
        b.key = ~0ull;
        return;
    }
    const std::uint32_t prims = static_cast<std::uint32_t>(b.prims.size() / kPrimWords);
    // PS2_NATIVE_LIMIT=<n>: only the first n batches that could go to the native renderer do, to find the batch a
    // wrong picture starts with.
    static const std::uint64_t limit = std::getenv("PS2_NATIVE_LIMIT") ? std::strtoull(std::getenv("PS2_NATIVE_LIMIT"), nullptr, 10) : ~0ull;
    // A small batch whose targets GS memory holds as they are goes to the compute rasterizer, which draws there
    // without anything having to be written back afterwards for whoever reads it next. GT4's colour correction is
    // 140 such batches a frame, each read by the next: a tile of the picture, then a palette made from it.
    // PS2_NATIVE_SMALL=<pixels> sets how small (4096 by default, 0 for never).
    static const long small = std::getenv("PS2_NATIVE_SMALL") ? std::atol(std::getenv("PS2_NATIVE_SMALL")) : 4096;
    bool native = b.native && g_native_batches < limit;
    if (native && static_cast<long>(b.maxx - b.minx + 1) * (b.maxy - b.miny + 1) <= small) {
        bool in_memory = true;
        for (const auto &s : g_surfaces) {
            const bool mine = s->depth ? b.zte && s->bp == b.zbp && s->psm == b.zpsm : s->bp == b.fbp;
            if (mine && s->bw == b.fbw && s->dirty) in_memory = false;
        }
        if (in_memory) {
            native = false;
            g_why_prims[kWhyCount - 2] += prims;
            ++g_why_batches[kWhyCount - 2];
        }
    }
    if (native && flush_native(prims)) {
        ++g_native_batches;
        g_native_prims += prims;
        census(prims, false);
    } else {
        if (native && !g_refusal_counted) {  // no targets for it: frame and depth in the same memory, or no room
            g_why_prims[kWhyCount - 1] += prims;
            ++g_why_batches[kWhyCount - 1];
        }
        g_refusal_counted = false;
        raster_batch(prims);
    }
    mark_rendered(pages(b.fbp, b.fbw, b.fpsm, b.minx, b.miny, b.maxx, b.maxy));
    if (b.zwrite) mark_rendered(pages(b.zbp, b.fbw, b.zpsm, b.minx, b.miny, b.maxx, b.maxy));
    // Whatever was drawn over the counting texture of GT4's colour correction, it no longer counts.
    if (g_lut.count_ok && pages(b.fbp, b.fbw, b.fpsm, b.minx, b.miny, b.maxx, b.maxy).overlaps(g_lut.count_range.lo, g_lut.count_range.hi)) g_lut.count_ok = false;
    clear_batch();
}


void batch_ranges() {
    Batch &b = g_batch;
    if (b.rx0 == b.minx && b.ry0 == b.miny && b.rx1 == b.maxx && b.ry1 == b.maxy) return;
    b.rx0 = b.minx;
    b.ry0 = b.miny;
    b.rx1 = b.maxx;
    b.ry1 = b.maxy;
    b.fb_range = region(b.fbp, b.fbw, b.fpsm, b.minx, b.miny, b.maxx, b.maxy);
    b.z_range = region(b.zbp, b.fbw, b.zpsm, b.minx, b.miny, b.maxx, b.maxy);
}

bool batch_conflict(const Range &r, bool clut) {
    Batch &b = g_batch;
    if (b.prims.empty()) return false;
    batch_ranges();
    if (b.fb_range.overlaps(r.lo, r.hi)) return true;
    if (b.zte && b.z_range.overlaps(r.lo, r.hi)) return true;
    if (!clut)
        for (const Range &t : g_batch_reads)
            if (t.overlaps(r.lo, r.hi)) return true;
    return false;
}

// Emits the first `upto` of the queued uploads and CLUT loads (all of them by default).
void emit_pre(std::size_t upto) {
    const Timed timed(kCpuPre);
    std::vector<PreOp> later;
    if (upto < g_pre.size()) {
        later.assign(g_pre.begin() + static_cast<std::ptrdiff_t>(upto), g_pre.end());
        g_pre.resize(upto);
    }
    if (!g_surfaces.empty()) {
        g_sync_cause = 4;
        for (const PreOp &op : g_pre) sync_memory(op.range, !op.clut);
        g_sync_cause = 0;
    }
    if (g_trace > 0)
        for (const PreOp &op : g_pre)
            TRACE("%s %llX-%llX rec %X %X %X %X %X %X %X %X %X %X %X rect %u,%u-%u,%u\n", op.clut ? "clut" : "upload", static_cast<unsigned long long>(op.range.lo),
                  static_cast<unsigned long long>(op.range.hi), op.rec[0], op.rec[1], op.rec[2], op.rec[3], op.rec[4], op.rec[5], op.rec[6], op.rec[7], op.rec[8], op.rec[9], op.rec[10], op.x0, op.y0,
                  op.x1, op.y1);
    // Uploads go ahead of the CLUT loads queued before them, as far as no such load reads what the upload writes:
    // a game alternates the two, and every change from one kind to the other is a dispatch.
    {
        std::vector<PreOp> ordered;
        ordered.reserve(g_pre.size());
        for (const PreOp &op : g_pre) {
            std::size_t at = ordered.size();
            if (!op.clut)
                while (at > 0 && ordered[at - 1].clut && !ordered[at - 1].range.overlaps(op.range.lo, op.range.hi)) --at;
            ordered.insert(ordered.begin() + static_cast<std::ptrdiff_t>(at), op);
        }
        g_pre = std::move(ordered);
    }
    std::size_t i = 0;
    while (i < g_pre.size()) {
        const bool clut = g_pre[i].clut;
        std::size_t j = i + 1;
        while (j < g_pre.size() && g_pre[j].clut == clut) {
            bool overlap = false;
            if (!clut)
                for (std::size_t k = i; k < j && !overlap; ++k) overlap = uploads_overlap(g_pre[k], g_pre[j]);
            if (overlap) break;
            ++j;
        }
        const std::uint32_t count = static_cast<std::uint32_t>(j - i);
        std::uint32_t *records = g_arena.grow(count * 12u);
        const std::uint32_t offset = static_cast<std::uint32_t>(records - g_arena.data());
        std::uint32_t total = 0;
        for (std::size_t k = i; k < j; ++k) {
            std::memcpy(records + (k - i) * 12, g_pre[k].rec, sizeof g_pre[k].rec);
            if (!clut) {
                records[(k - i) * 12 + 11] = total;
                total += g_pre[k].rec[7];
            }
        }
        if (clut) g_commands.push_back({kLoadClut, 1, 1, push_params({offset, count})});
        else dispatch_linear(kUpload, total, push_params({offset, count, total}));
        i = j;
    }
    g_pre = std::move(later);
}

void emit_all() {
    if (lut_try()) return;
    lut_flush();
    emit_pre();
    flush_batch();
}

// emit_all, and nothing kept back: a taken step of the colour correction leaves its lookup and the queued uploads
// and CLUT loads waiting for the next step, and they cannot wait past a submit (which empties what they point into)
// or anything else that reads or writes GS memory.
void emit_everything() {
    emit_all();
    lut_flush();
    emit_pre();
}

void queue_pre(const PreOp &op) {
    if (batch_conflict(op.range, op.clut)) emit_all();
    // An upload could change the colour correction's table or what it reads: what is collected is done first.
    if (!op.clut) {
        lut_flush();
        wrote(op.range);
    }
    g_pre.push_back(op);
}

void submit(WGPUTextureView present_view);

std::uint64_t queue_upload(const std::uint8_t *bytes, std::size_t size, std::uint32_t lead_bits) {
    Image &im = g_image;
    const std::uint64_t bits = static_cast<std::uint64_t>(size) * 8;
    if (bits <= lead_bits) return 0;
    const std::uint32_t pixels = static_cast<std::uint32_t>(std::min<std::uint64_t>((bits - lead_bits) / im.bpp, im.total - im.done));
    if (!pixels) return 0;
    const std::uint64_t used = lead_bits + static_cast<std::uint64_t>(pixels) * im.bpp;
    PreOp op;
    const auto n = static_cast<std::size_t>((used + 7) / 8);
    std::uint32_t offset = 0, source = 0;
    if (n >= kCacheMinBytes) {
        const std::uint64_t key = content_key(bytes, n);
        bool cached = cache_find(key, bytes, n, offset);
        if (!cached && ++g_cache_seen[key] >= 2) {
            cached = cache_insert(key, bytes, n, offset);
            // The ring cannot evict its oldest entry while the pending submit reads it. A texture drawn every frame
            // is read again before the next insert, so without a submit here the cache would take nothing new
            // once it wraps, and every upload from then on goes through the per-submit buffer.
            if (!cached && !g_cache_ring.empty() && g_cache_ring.front().used == g_serial) {
                emit_everything();
                submit(nullptr);
                cached = cache_insert(key, bytes, n, offset);
            }
        }
        if (cached) source = 0x80000000u;
        if (g_cache_seen.size() > 65536) g_cache_seen.clear();
    }
    if (!source) offset = arena_append(bytes, n);
    const std::uint32_t rec[11] = {im.dbp, im.dbw, im.dpsm, im.dsax, im.dsay, im.rrw, im.done, pixels, offset, lead_bits, im.bpp | source};
    std::memcpy(op.rec, rec, sizeof rec);
    const std::uint32_t first = im.done / im.rrw, last = (im.done + pixels - 1) / im.rrw;
    op.range = region(im.dbp, im.dbw, im.dpsm, static_cast<int>(im.dsax), static_cast<int>(im.dsay + first), static_cast<int>(im.dsax + im.rrw - 1),
                      static_cast<int>(im.dsay + last));
    // GT4's counting texture for its colour correction, as a whole or written over.
    if (im.dpsm == PSMT8 && im.dbw == 1 && im.rrw == 8 && !im.dsax && !im.dsay && !im.done && pixels == 256 && !lead_bits && std::memcmp(bytes, kLutCount, 256) == 0) {
        g_lut.count_ok = true;
        g_lut.count_bp = im.dbp;
        g_lut.count_range = op.range;
    } else if (g_lut.count_ok && op.range.overlaps(g_lut.count_range.lo, g_lut.count_range.hi)) {
        g_lut.count_ok = false;
    }
    op.x0 = im.dsax;
    op.x1 = im.dsax + im.rrw - 1;
    op.y0 = im.dsay + first;
    op.y1 = im.dsay + last;
    op.image = im.id;
    queue_pre(op);
    im.done += pixels;
    return used;
}

void finish_image_if_done() {
    Image &im = g_image;
    if (im.done < im.total) return;
    im.active = false;
    im.pending.clear();
    const std::uint32_t psm = im.dpsm;
    if (psm != PSMCT32 && psm != PSMCT16 && psm != PSMCT16S && psm != PSMT8 && psm != PSMT4 && psm != PSMZ32 && psm != PSMZ16 && psm != PSMZ16S) return;
    std::uint32_t pw, ph;
    page_size(psm, pw, ph);
    const std::uint32_t per_row = std::max<std::uint32_t>(1, im.dbw * 64 / pw), rows = im.rrw ? im.total / im.rrw : 0;
    for (std::uint32_t py = (im.dsay + ph - 1) / ph; py < (im.dsay + rows) / ph; ++py)
        for (std::uint32_t px = (im.dsax + pw - 1) / pw; px < (im.dsax + im.rrw) / pw; ++px)
            g_rendered[((im.dbp * 256 + (px + py * per_row) * 8192) >> 13) & 511u] = false;
}

void flush_image() {
    Image &im = g_image;
    if (!im.active || im.pending.empty()) return;
    const std::uint64_t used = queue_upload(im.pending.data(), im.pending.size(), im.lead_bits);
    if (!used) return;
    im.pending.erase(im.pending.begin(), im.pending.begin() + static_cast<std::ptrdiff_t>(used / 8));
    im.lead_bits = static_cast<std::uint32_t>(used % 8);
    finish_image_if_done();
}

void flush_all() {
    flush_image();
    emit_everything();
}

void on_error(WGPUDevice const *, WGPUErrorType type, WGPUStringView message, void *, void *) {
    std::fprintf(stderr, "[gpu] error %d: %.*s\n", static_cast<int>(type), static_cast<int>(message.length == WGPU_STRLEN ? std::strlen(message.data) : message.length), message.data);
}

void pump() {
#ifdef __EMSCRIPTEN__
    browser_yield();
    wgpuInstanceProcessEvents(g_instance);
#else
    if (g_device) wgpuDevicePoll(g_device, true, nullptr);
    else wgpuInstanceProcessEvents(g_instance);
#endif
}

void wait_until(const bool &flag) {
    while (!flag) pump();
}

WGPUBuffer make_buffer(std::uint64_t size, WGPUBufferUsage usage) {
    WGPUBufferDescriptor d = WGPU_BUFFER_DESCRIPTOR_INIT;
    d.size = size;
    d.usage = usage;
    return wgpuDeviceCreateBuffer(g_device, &d);
}

void rebuild_group() {
    if (g_group) wgpuBindGroupRelease(g_group);
    WGPUBindGroupEntry e[8] = {};
    WGPUBuffer buffers[8] = {g_vram, g_tables, g_params, g_data, g_clut, g_scratch, g_cache, g_decoded};
    for (int i = 0; i < 8; ++i) {
        e[i] = WGPU_BIND_GROUP_ENTRY_INIT;
        e[i].binding = static_cast<std::uint32_t>(i);
        e[i].buffer = buffers[i];
        e[i].size = i == 2 ? kParamStride : WGPU_WHOLE_SIZE;
    }
    WGPUBindGroupDescriptor d = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    d.layout = g_layout;
    d.entryCount = 8;
    d.entries = e;
    g_group = wgpuDeviceCreateBindGroup(g_device, &d);
    if (!g_native) return;
    // The same buffers for the native renderer's shaders, without the upload cache they do not read.
    if (g_native_group) wgpuBindGroupRelease(g_native_group);
    e[6] = e[7];
    d.layout = g_native_layout;
    d.entryCount = 7;
    g_native_group = wgpuDeviceCreateBindGroup(g_device, &d);
}

void ensure_capacity() {
    bool changed = false;
    const std::uint64_t data_bytes = std::max<std::uint64_t>(g_arena.size() * 4, 16);
    if (data_bytes > g_data_size) {
        if (g_data) wgpuBufferRelease(g_data);
        g_data_size = std::max<std::uint64_t>(g_data_size * 2, (data_bytes + (4u << 20) - 1) & ~((4ull << 20) - 1));
        g_data = make_buffer(g_data_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
#ifdef PS2_GPU_GL
        ps2GlDiscardOnWrite(g_data);
#endif
        changed = true;
    }
    const std::uint64_t param_bytes = std::max<std::uint64_t>(g_param_records.size(), 1) * kParamStride;
    if (param_bytes > g_params_size) {
        if (g_params) wgpuBufferRelease(g_params);
        g_params_size = std::max<std::uint64_t>(g_params_size * 2, std::max<std::uint64_t>(param_bytes, 64 * 1024));
        g_params = make_buffer(g_params_size, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
#ifdef PS2_GPU_GL
        ps2GlDiscardOnWrite(g_params);
#endif
        changed = true;
    }
    if (changed || !g_group) rebuild_group();
}

// The render passes of the native renderer from command i on; successive ones on the same targets share a pass.
// Returns the last command taken.
std::size_t encode_native(WGPUCommandEncoder enc, std::size_t i, bool merge) {
    const NativePass &first = g_native_passes[g_commands[i].x];
    WGPURenderPassColorAttachment color = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    WGPURenderPassDepthStencilAttachment depth = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
    WGPURenderPassDescriptor rd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    if (first.color) {
        color.view = first.color->view;
        color.loadOp = WGPULoadOp_Load;
        color.storeOp = WGPUStoreOp_Store;
        rd.colorAttachmentCount = 1;
        rd.colorAttachments = &color;
    }
    if (first.depth) {
        depth.view = first.depth->view;
        depth.depthLoadOp = WGPULoadOp_Load;
        depth.depthStoreOp = WGPUStoreOp_Store;
        rd.depthStencilAttachment = &depth;
    }
    WGPURenderPassEncoder rp = nullptr;
    WGPURenderPipeline current = nullptr;
    float constant = 0.0f;
    std::uint32_t last_x = ~0u, last_y = ~0u, last_w = ~0u, last_h = ~0u;
    bool backdrop_bound = false;
    for (;;) {
        for (const NativeDraw &d : g_native_passes[g_commands[i].x].draws) {
            if (!d.pipeline) {
                // The colour target as it is now, for the feedback draw that follows: a copy cannot be made inside
                // a pass, so this one ends and another begins.
                if (rp) {
                    wgpuRenderPassEncoderEnd(rp);
                    wgpuRenderPassEncoderRelease(rp);
                    rp = nullptr;
                }
                WGPUTexelCopyTextureInfo from = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT, to = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
                from.texture = first.color->texture;
                to.texture = first.color->backdrop;
                from.origin = to.origin = WGPUOrigin3D{d.x * g_scale, d.y * g_scale, 0};
                const WGPUExtent3D size{d.w * g_scale, d.h * g_scale, 1};
                wgpuCommandEncoderCopyTextureToTexture(enc, &from, &to, &size);
                current = nullptr;
                constant = 0.0f;
                backdrop_bound = false;
                last_x = last_y = last_w = last_h = ~0u;
                continue;
            }
            if (!rp) {
                rp = wgpuCommandEncoderBeginRenderPass(enc, &rd);
                current = nullptr;
                constant = 0.0f;
                backdrop_bound = false;
                last_x = last_y = last_w = last_h = ~0u;
            }
            if (d.feedback && !backdrop_bound) {
                wgpuRenderPassEncoderSetBindGroup(rp, 1, first.color->backdrop_group, 0, nullptr);
                backdrop_bound = true;
            }
            if (d.pipeline != current) {
                wgpuRenderPassEncoderSetPipeline(rp, d.pipeline);
                current = d.pipeline;
            }
            const std::uint32_t offset = d.param * kParamStride;
            wgpuRenderPassEncoderSetBindGroup(rp, 0, g_native_group, 1, &offset);
            if (d.x != last_x || d.y != last_y || d.w != last_w || d.h != last_h) {
                wgpuRenderPassEncoderSetScissorRect(rp, d.x * g_scale, d.y * g_scale, d.w * g_scale, d.h * g_scale);
                last_x = d.x; last_y = d.y; last_w = d.w; last_h = d.h;
            }
            if (d.constant != constant) {
                constant = d.constant;
                const WGPUColor c{constant, constant, constant, constant};
                wgpuRenderPassEncoderSetBlendConstant(rp, &c);
            }
            wgpuRenderPassEncoderDraw(rp, d.vertices, 1, 0, 0);
        }
        if (!merge || i + 1 >= g_commands.size() || g_commands[i + 1].pipeline != kNative) break;
        const NativePass &next = g_native_passes[g_commands[i + 1].x];
        if (next.color != first.color || next.depth != first.depth) break;
        ++i;
    }
    if (rp) {
        wgpuRenderPassEncoderEnd(rp);
        wgpuRenderPassEncoderRelease(rp);
    }
    return i;
}

void submitted() {
    g_commands.clear();
    g_command_info.clear();
    ++g_serial;
    g_param_records.clear();
    g_arena.clear();
    if (!decode_cache_on()) g_decoded_used = 0;  // kept across submits when cached
    g_clut_loads = 0;
    g_native_passes.clear();
    release_retired();
    if (g_trace > 0) std::fprintf(stderr, "[trace] ---- submitted, %d to go\n", --g_trace);
    // PS2_GL_STATS (the Switch build's --stats) or PS2_NATIVE_STATS: once a second, how much each rasterizer drew
    // and how often a target went to GS memory and back. What stays with the compute rasterizer is what costs.
    static const bool report = g_native && (std::getenv("PS2_GL_STATS") || std::getenv("PS2_NATIVE_STATS"));
    if (!report) return;
    static auto since = std::chrono::steady_clock::now();
    static std::uint64_t last[6] = {};
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
    if (seconds < 1.0) return;
    static std::uint64_t last_draws = 0;
    const std::uint64_t now[6] = {g_native_batches, g_native_prims, g_compute_batches, g_compute_prims, g_stores, g_loads};
    std::fprintf(stderr, "[native] per second: %.0f batches (%.0f prims, %.0f draws) native, %.0f batches (%.0f prims) compute, %.0f stores, %.0f loads\n",
                 static_cast<double>(now[0] - last[0]) / seconds, static_cast<double>(now[1] - last[1]) / seconds, static_cast<double>(g_native_draws - last_draws) / seconds,
                 static_cast<double>(now[2] - last[2]) / seconds, static_cast<double>(now[3] - last[3]) / seconds, static_cast<double>(now[4] - last[4]) / seconds,
                 static_cast<double>(now[5] - last[5]) / seconds);
    // Twenty busy seconds in a row are a race: trace a few frames of it, twice, half a minute apart.
    static const bool tracing = std::getenv("PS2_NATIVE_TRACE") && std::atoi(std::getenv("PS2_NATIVE_TRACE")) != 0;
    static int busy = 0, traces = 0;
    busy = static_cast<double>(now[1] - last[1]) / seconds > 50000.0 ? busy + 1 : 0;
    if (tracing && traces < 2 && busy == 20 + 30 * traces) {
        ++traces;
        g_trace = 3;
        std::fprintf(stderr, "[trace] ---- the next %d submits\n", g_trace);
    }
    last_draws = g_native_draws;
    std::copy(now, now + 6, last);
    // And why the compute rasterizer's share was not the native renderer's: prims and batches a second, most first.
    static std::uint64_t last_prims[kWhyCount] = {}, last_batches[kWhyCount] = {};
    std::size_t order[kWhyCount];
    for (std::size_t i = 0; i < kWhyCount; ++i) order[i] = i;
    std::sort(order, order + kWhyCount, [&](std::size_t a, std::size_t b) { return g_why_prims[a] - last_prims[a] > g_why_prims[b] - last_prims[b]; });
    std::string line;
    for (std::size_t i : order) {
        if (g_why_prims[i] == last_prims[i]) continue;
        char text[96];
        std::snprintf(text, sizeof text, "%s %s %.0f/%.0f", line.empty() ? "" : ",", kWhy[i], static_cast<double>(g_why_prims[i] - last_prims[i]) / seconds,
                      static_cast<double>(g_why_batches[i] - last_batches[i]) / seconds);
        line += text;
    }
    if (!line.empty()) std::fprintf(stderr, "[native] compute because (prims/batches):%s\n", line.c_str());
    {
        char text[256];
        int n = 0;
        for (int i = 0; i < kCpuParts; ++i) n += std::snprintf(text + n, sizeof text - static_cast<std::size_t>(n), "%s %s %.0f", i ? "," : "", kCpuNames[i], g_cpu.ms[i] / seconds);
        std::fprintf(stderr, "[native] GS thread ms a second in the renderer:%s\n", text);
        std::fill(g_cpu.ms, g_cpu.ms + kCpuParts, 0.0);
    }
    static std::uint64_t last_hits = 0, last_misses = 0, last_words = 0, last_restarts = 0, last_saved_words = 0;
    static std::uint64_t last_invalidated = 0, last_cold = 0, last_small = 0;
    static std::uint64_t last_admission_skips = 0, last_admission_texels = 0;
    static std::uint64_t last_capacity_bypasses = 0, last_capacity_texels = 0;
    static std::uint64_t last_decode_evictions = 0, last_decode_reclaimed = 0;
    const std::uint64_t hits = g_decode_hits - last_hits, misses = g_decode_misses - last_misses;
    if (hits + misses) {
        const double hit_rate = 100.0 * static_cast<double>(hits) / static_cast<double>(hits + misses);
        std::fprintf(stderr, "[native] decoded textures a second: %.0f found, %.0f decoded (%.0f texels), %.0f times the buffer full\n",
                     static_cast<double>(hits) / seconds, static_cast<double>(misses) / seconds,
                     static_cast<double>(g_decode_words - last_words) / seconds, static_cast<double>(g_decode_restarts - last_restarts) / seconds);
        std::fprintf(stderr, "[native] decode cache: %.1f%% hits, %.0f texels/s reused, %zu entries, %.1f/%.1f MiB used\n",
                     hit_rate, static_cast<double>(g_decode_saved_words - last_saved_words) / seconds,
                     g_decoded_cache.size(), static_cast<double>(g_decoded_used) * 4.0 / (1024.0 * 1024.0),
                     static_cast<double>(kDecodedWords) * 4.0 / (1024.0 * 1024.0));
        std::fprintf(stderr, "[native] decode cache misses: %.0f invalidated/s, %.0f absent/s, %.0f small-batch skips/s\n",
                     static_cast<double>(g_decode_invalidated - last_invalidated) / seconds,
                     static_cast<double>(g_decode_cold - last_cold) / seconds,
                     static_cast<double>(g_decode_small - last_small) / seconds);
    }
    last_invalidated = g_decode_invalidated;
    last_cold = g_decode_cold;
    last_small = g_decode_small;
    std::fprintf(stderr, "[native] decode admission: %.0f cold large textures/s bypassed, %.0f texels/s not predecoded\\n",
                 static_cast<double>(g_decode_admission_skips - last_admission_skips) / seconds,
                 static_cast<double>(g_decode_admission_texels - last_admission_texels) / seconds);
    std::fprintf(stderr, "[native] decode capacity: %.0f bypasses/s, %.0f texels/s direct, %.1f/%.1f MiB occupied\n",
                 static_cast<double>(g_decode_capacity_bypasses - last_capacity_bypasses) / seconds,
                 static_cast<double>(g_decode_capacity_texels - last_capacity_texels) / seconds,
                 static_cast<double>(g_decoded_used) * 4.0 / (1024.0 * 1024.0),
                 static_cast<double>(kDecodedWords) * 4.0 / (1024.0 * 1024.0));
    std::fprintf(stderr, "[native] decode reclaim: %.0f evictions/s, %.0f texels/s recycled, %zu pending blocks, %zu free spans\n",
                 static_cast<double>(g_decode_evictions - last_decode_evictions) / seconds,
                 static_cast<double>(g_decode_reclaimed_words - last_decode_reclaimed) / seconds,
                 g_decode_retired.size(), g_decode_free.size());
    last_decode_evictions = g_decode_evictions;
    last_decode_reclaimed = g_decode_reclaimed_words;
    last_capacity_bypasses = g_decode_capacity_bypasses;
    last_capacity_texels = g_decode_capacity_texels;
    last_admission_skips = g_decode_admission_skips;
    last_admission_texels = g_decode_admission_texels;
    last_hits = g_decode_hits;
    last_misses = g_decode_misses;
    last_words = g_decode_words;
    last_restarts = g_decode_restarts;
    last_saved_words = g_decode_saved_words;
    static std::uint64_t last_feedback = 0, last_copies = 0, last_copy_pixels = 0;
    const std::uint64_t feedback_copies = g_feedback_copies - last_copies;
    const std::uint64_t feedback_pixels = g_feedback_copy_pixels - last_copy_pixels;
    if (g_feedback_draws != last_feedback)
        std::fprintf(stderr, "[native] reading the target: %.0f prims a second in %.0f groups, each a copy\n", static_cast<double>(g_feedback_draws - last_feedback) / seconds,
                     static_cast<double>(feedback_copies) / seconds);
    if (feedback_copies)
        std::fprintf(stderr, "[native] feedback copies: %.0f/s, %.0f source pixels/s, %.0f pixels/copy, %.0f render pass restarts/s (scale %u)\n",
                     static_cast<double>(feedback_copies) / seconds,
                     static_cast<double>(feedback_pixels) / seconds,
                     static_cast<double>(feedback_pixels) / static_cast<double>(feedback_copies),
                     static_cast<double>(feedback_copies) / seconds, g_scale);
    last_feedback = g_feedback_draws;
    last_copies = g_feedback_copies;
    last_copy_pixels = g_feedback_copy_pixels;
    static std::uint64_t last_group_calls = 0;
    static std::uint64_t last_outline_tests = 0;
    static std::uint64_t last_group_prims = 0;
    const auto group_calls = g_feedback_group_calls - last_group_calls;
    const auto outline_tests = g_feedback_outline_tests - last_outline_tests;
    const auto group_prims = g_feedback_group_prims - last_group_prims;
    if (group_calls)
        std::fprintf(stderr,
            "[native] feedback grouping: %.0f calls/s, %.0f prim visits/s, "
            "%.0f overlap tests/s, %.1f tests/call\n",
            static_cast<double>(group_calls) / seconds,
            static_cast<double>(group_prims) / seconds,
            static_cast<double>(outline_tests) / seconds,
            static_cast<double>(outline_tests) / static_cast<double>(group_calls));
    last_group_calls = g_feedback_group_calls;
    last_outline_tests = g_feedback_outline_tests;
    last_group_prims = g_feedback_group_prims;
    static std::uint64_t last_tiles = 0, last_passes = 0;
    if (g_lut_tiles != last_tiles)
        std::fprintf(stderr, "[native] colour correction: %.0f tiles a second in %.0f passes\n", static_cast<double>(g_lut_tiles - last_tiles) / seconds, static_cast<double>(g_lut_passes - last_passes) / seconds);
    last_tiles = g_lut_tiles;
    last_passes = g_lut_passes;
    static std::uint64_t last_cause[6] = {};
    line.clear();
    for (std::size_t i = 0; i < 6; ++i) {
        if (g_store_cause[i] == last_cause[i]) continue;
        char text[96];
        std::snprintf(text, sizeof text, "%s %s %.0f", line.empty() ? "" : ",", kStoreCause[i], static_cast<double>(g_store_cause[i] - last_cause[i]) / seconds);
        line += text;
    }
    if (!line.empty()) std::fprintf(stderr, "[native] stores for:%s\n", line.c_str());
    std::copy(g_store_cause, g_store_cause + 6, last_cause);
    std::copy(g_why_prims, g_why_prims + kWhyCount, last_prims);
    std::copy(g_why_batches, g_why_batches + kWhyCount, last_batches);
    since = std::chrono::steady_clock::now();
}

void track_decode_completion(std::uint64_t serial) {
    auto *completed_serial = new std::uint64_t(serial);
    WGPUQueueWorkDoneCallbackInfo decode_done = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    decode_done.mode = WGPUCallbackMode_AllowProcessEvents;
    decode_done.callback = [](WGPUQueueWorkDoneStatus status, WGPUStringView, void *userdata, void *) {
        std::unique_ptr<std::uint64_t> completed(static_cast<std::uint64_t *>(userdata));
        if (status == WGPUQueueWorkDoneStatus_Success)
            g_decode_completed_serial = std::max(g_decode_completed_serial, *completed);
    };
    decode_done.userdata1 = completed_serial;
    wgpuQueueOnSubmittedWorkDone(g_queue, decode_done);
}

void profile_submit() {
    constexpr std::uint32_t kMax = 2048;
    if (!g_profile_queries) {
        WGPUQuerySetDescriptor qd = WGPU_QUERY_SET_DESCRIPTOR_INIT;
        qd.type = WGPUQueryType_Timestamp;
        qd.count = kMax * 2;
        g_profile_queries = wgpuDeviceCreateQuerySet(g_device, &qd);
        g_profile_resolve = make_buffer(kMax * 16, WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc);
        g_profile_read = make_buffer(kMax * 16, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    }
    ensure_capacity();
    // Native-only / present-only submissions can have no compute payload.
    // Avoid queue writes and backend validation for empty upload ranges.
    if (!g_arena.empty())
        wgpuQueueWriteBuffer(g_queue, g_data, 0, g_arena.data(), g_arena.size() * 4);
    if (!g_param_records.empty())
        wgpuQueueWriteBuffer(g_queue, g_params, 0, g_param_records.data(), g_param_records.size() * kParamStride);
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_device, nullptr);
    std::vector<std::uint32_t> measured_commands;
    measured_commands.reserve(std::min<std::size_t>(g_commands.size(), kMax));
    for (std::uint32_t i = 0; i < g_commands.size(); ++i) {
        const Command &c = g_commands[i];
        if (c.pipeline == kNative) {
            encode_native(enc, i, false);  // not timed
            continue;
        }
        WGPUPassTimestampWrites writes = WGPU_PASS_TIMESTAMP_WRITES_INIT;
        writes.querySet = g_profile_queries;
        const bool measure_command = measured_commands.size() < kMax;
        if (measure_command) {
            const auto query = static_cast<std::uint32_t>(measured_commands.size());
            writes.beginningOfPassWriteIndex = query * 2;
            writes.endOfPassWriteIndex = query * 2 + 1;
            measured_commands.push_back(i);
        }
        WGPUComputePassDescriptor cd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        if (measure_command) cd.timestampWrites = &writes;
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, &cd);
        wgpuComputePassEncoderSetPipeline(pass, g_pipelines[c.pipeline]);
        const std::uint32_t offset = c.param * kParamStride;
        wgpuComputePassEncoderSetBindGroup(pass, 0, g_group, 1, &offset);
        if (c.target) wgpuComputePassEncoderSetBindGroup(pass, 1, c.target->group, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, c.x, c.y, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }
    const auto n = static_cast<std::uint32_t>(measured_commands.size());
    if (n) {
        wgpuCommandEncoderResolveQuerySet(enc, g_profile_queries, 0, n * 2, g_profile_resolve, 0);
        wgpuCommandEncoderCopyBufferToBuffer(enc, g_profile_resolve, 0, g_profile_read, 0, n * 16);
    }
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(g_queue, 1, &cmd);
    track_decode_completion(g_serial);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    if (n) {
        bool done = false;
        WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        cb.mode = WGPUCallbackMode_AllowProcessEvents;
        cb.callback = [](WGPUMapAsyncStatus, WGPUStringView, void *flag, void *) { *static_cast<bool *>(flag) = true; };
        cb.userdata1 = &done;
        wgpuBufferMapAsync(g_profile_read, WGPUMapMode_Read, 0, n * 16, cb);
        wait_until(done);
        const auto *t = static_cast<const std::uint64_t *>(wgpuBufferGetConstMappedRange(g_profile_read, 0, n * 16));
        std::size_t info = 0;
        for (std::uint32_t query = 0; t && query < n; ++query) {
            const std::uint32_t i = measured_commands[query];
            while (info < g_command_info.size() && g_command_info[info].index < i) ++info;
            std::fprintf(g_profile_out, "%llu %u %s %llu %u %u", static_cast<unsigned long long>(frame_count()), i,
                         kEntryPoints[g_commands[i].pipeline],
                         static_cast<unsigned long long>(t[query * 2 + 1] > t[query * 2] ? t[query * 2 + 1] - t[query * 2] : 0), g_commands[i].x, g_commands[i].y);
            if (info < g_command_info.size() && g_command_info[info].index == i)
                for (std::uint32_t v : g_command_info[info].values) std::fprintf(g_profile_out, " %u", v);
            std::fprintf(g_profile_out, "\n");
        }
        wgpuBufferUnmap(g_profile_read);
    }
    submitted();
}

// Browsers copy writes above about 1 MiB through a slower path, so large uploads go in pieces.
void submit(WGPUTextureView present_view = nullptr) {
    const Timed timed(kCpuSubmit);
    if (g_commands.empty() && !present_view) return;
    if (g_profile_out && !present_view && g_timestamps && frame_count() >= g_profile_from && frame_count() < g_profile_to) {
        profile_submit();
        return;
    }
    ensure_capacity();
    // Native-only / present-only submissions can have no compute payload.
    // Avoid queue writes and backend validation for empty upload ranges.
    if (!g_arena.empty())
        wgpuQueueWriteBuffer(g_queue, g_data, 0, g_arena.data(), g_arena.size() * 4);
    if (!g_param_records.empty())
        wgpuQueueWriteBuffer(g_queue, g_params, 0, g_param_records.data(), g_param_records.size() * kParamStride);
    if (g_query_ready) {
        const void *mapped = wgpuBufferGetConstMappedRange(g_query_read, 0, 16);
        if (mapped) {
            std::uint64_t t[2];
            std::memcpy(t, mapped, sizeof t);
            if (t[1] > t[0]) {
                g_measured_ns += t[1] - t[0];
                ++g_measured;
            }
        }
        wgpuBufferUnmap(g_query_read);
        g_query_ready = g_query_pending = false;
    }
    const bool measure = g_timestamps && !g_query_pending;
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_device, nullptr);
    WGPUPassTimestampWrites writes = WGPU_PASS_TIMESTAMP_WRITES_INIT;
    writes.querySet = g_queries;
    writes.beginningOfPassWriteIndex = 0;
    writes.endOfPassWriteIndex = 1;
    WGPUComputePassDescriptor cd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    if (measure) cd.timestampWrites = &writes;
    // Native-only submissions should not create an empty compute pass.
    // Timestamp queries are attached to the first actual compute pass.
    WGPUComputePassEncoder pass = nullptr;
    bool first_compute_pass = true;
    Pipeline current = kPipelines;
    Surface *bound_target = nullptr;
    ++g_submits;
    ++g_total_passes;
    for (std::size_t i = 0; i < g_commands.size(); ++i) {
        const Command &c = g_commands[i];
        if (c.pipeline == kNative) {
            if (pass) {
                wgpuComputePassEncoderEnd(pass);
                wgpuComputePassEncoderRelease(pass);
                pass = nullptr;
            }
            i = encode_native(enc, i, true);
            continue;
        }
        if (!pass) {
            pass = wgpuCommandEncoderBeginComputePass(enc, &cd);
            if (first_compute_pass) {
                cd.timestampWrites = nullptr;
                first_compute_pass = false;
            }
            current = kPipelines;
            bound_target = nullptr;
        }
        ++g_dispatches[c.pipeline];
        if (c.pipeline != current) {
            wgpuComputePassEncoderSetPipeline(pass, g_pipelines[c.pipeline]);
            current = c.pipeline;
        }
        const std::uint32_t offset = c.param * kParamStride;
        wgpuComputePassEncoderSetBindGroup(pass, 0, g_group, 1, &offset);
        if (c.target && c.target != bound_target) {
            wgpuComputePassEncoderSetBindGroup(pass, 1, c.target->group, 0, nullptr);
            bound_target = c.target;
        }
        wgpuComputePassEncoderDispatchWorkgroups(pass, c.x, c.y, 1);
    }
    if (pass) {
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }
    if (measure && !first_compute_pass) {
        wgpuCommandEncoderResolveQuerySet(enc, g_queries, 0, 2, g_query_resolve, 0);
        wgpuCommandEncoderCopyBufferToBuffer(enc, g_query_resolve, 0, g_query_read, 0, 16);
    }
    if (present_view) {
        WGPURenderPassColorAttachment color = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
        color.view = present_view;
        color.loadOp = WGPULoadOp_Clear;
        color.storeOp = WGPUStoreOp_Store;
        color.clearValue = WGPUColor{0.0, 0.0, 0.0, 1.0};
        WGPURenderPassDescriptor rd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
        rd.colorAttachmentCount = 1;
        rd.colorAttachments = &color;
        WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &rd);
        const float w = static_cast<float>(g_surface_width), h = static_cast<float>(g_surface_height);
        const float aspect = g_options.widescreen ? 16.0f / 9.0f : 4.0f / 3.0f;
        const float vw = std::min(w, h * aspect), vh = vw / aspect;
        wgpuRenderPassEncoderSetViewport(rp, (w - vw) * 0.5f, (h - vh) * 0.5f, vw, vh, 0.0f, 1.0f);
        wgpuRenderPassEncoderSetPipeline(rp, g_blit);
        wgpuRenderPassEncoderSetBindGroup(rp, 0, g_blit_group, 0, nullptr);
        wgpuRenderPassEncoderDraw(rp, 3, 1, 0, 0);
        wgpuRenderPassEncoderEnd(rp);
        wgpuRenderPassEncoderRelease(rp);
    }
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(g_queue, 1, &cmd);
    // The queue completion callback gates reuse of decoded texture slots.
    // The serial belongs to the commands just submitted, before submitted()
    // advances g_serial. A failed callback never authorizes reclamation.
    track_decode_completion(g_serial);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    if (measure && !first_compute_pass) {
        g_query_pending = true;
        WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        cb.mode = WGPUCallbackMode_AllowProcessEvents;
        cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void *, void *) {
            if (status == WGPUMapAsyncStatus_Success) g_query_ready = true;
            else g_query_pending = false;
        };
        wgpuBufferMapAsync(g_query_read, WGPUMapMode_Read, 0, 16, cb);
    }
#ifdef __EMSCRIPTEN__
    wgpuInstanceProcessEvents(g_instance);
#else
    wgpuDevicePoll(g_device, false, nullptr);
#endif
    submitted();
}

void read_back(WGPUBuffer source, std::uint64_t bytes, void *out, std::uint64_t offset = 0) {
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(enc, source, offset, g_readback, 0, bytes);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(g_queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    bool done = false;
    WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void *flag, void *) {
        if (status != WGPUMapAsyncStatus_Success) std::fprintf(stderr, "[gpu] map failed %d\n", static_cast<int>(status));
        *static_cast<bool *>(flag) = true;
    };
    cb.userdata1 = &done;
    wgpuBufferMapAsync(g_readback, WGPUMapMode_Read, 0, bytes, cb);
    wait_until(done);
    const void *mapped = wgpuBufferGetConstMappedRange(g_readback, 0, bytes);
    if (mapped) std::memcpy(out, mapped, bytes);
    wgpuBufferUnmap(g_readback);
}

std::uint32_t make_setup(const State &st, std::uint64_t attr, std::uint32_t *w) {
    const Context &c = st.ctx[(attr >> 9) & 1u];
    std::uint32_t flags = 0;
    if (attr & 0x8u) flags |= 1u;
    if (attr & 0x10u) flags |= 2u;
    if (attr & 0x20u) flags |= 4u;
    if (attr & 0x40u) flags |= 8u;
    if (attr & 0x100u) flags |= 16u;
    if ((c.zbuf >> 32) & 1u) flags |= 32u;
    if (c.test & 1u) flags |= 64u;
    if ((c.test >> 14) & 1u) flags |= 128u;
    if ((c.test >> 15) & 1u) flags |= 256u;
    // A depth test that always passes and writes nothing has nothing to do with the depth buffer: GT4 draws its
    // colour correction that way, over the memory that was the depth buffer.
    const bool z_unused = ((c.test >> 17) & 3u) == 1 && ((c.zbuf >> 32) & 1u);
    if (((c.test >> 16) & 1u) && !z_unused) flags |= 512u;
    if (st.pabe & 1u) flags |= 1024u;
    if (st.colclamp & 1u) flags |= 2048u;
    if (c.fba & 1u) flags |= 4096u;
    if ((c.tex0 >> 34) & 1u) flags |= 8192u;
    if ((c.tex1 >> 5) & 1u) flags |= 16384u;
    if ((st.texa >> 15) & 1u) flags |= 32768u;
    const std::uint32_t zpsm = 0x30u | (static_cast<std::uint32_t>(c.zbuf >> 24) & 0xFu);
    w[0] = flags;
    w[1] = (static_cast<std::uint32_t>(c.frame) & 0x1FFu) * 32u;
    w[2] = static_cast<std::uint32_t>(c.frame >> 16) & 0x3Fu;
    w[3] = static_cast<std::uint32_t>(c.frame >> 24) & 0x3Fu;
    w[4] = static_cast<std::uint32_t>(c.frame >> 32);
    w[5] = (static_cast<std::uint32_t>(c.zbuf) & 0x1FFu) * 32u;
    w[6] = zpsm;
    w[7] = (static_cast<std::uint32_t>(c.test >> 1) & 7u) | ((static_cast<std::uint32_t>(c.test >> 12) & 3u) << 4) |
           ((static_cast<std::uint32_t>(c.test >> 17) & 3u) << 8) | ((static_cast<std::uint32_t>(c.alpha) & 0xFFu) << 12) |
           ((static_cast<std::uint32_t>(c.tex0 >> 35) & 3u) << 20) | ((static_cast<std::uint32_t>(c.clamp) & 0xFu) << 24) |
           ((static_cast<std::uint32_t>(st.scanmsk) & 3u) << 28);
    w[8] = static_cast<std::uint32_t>(c.test >> 4) & 0xFFu;
    w[9] = static_cast<std::uint32_t>(c.alpha >> 32) & 0xFFu;
    w[10] = (static_cast<std::uint32_t>(c.scissor) & 0x7FFu) | ((static_cast<std::uint32_t>(c.scissor >> 16) & 0x7FFu) << 16);
    w[11] = (static_cast<std::uint32_t>(c.scissor >> 32) & 0x7FFu) | ((static_cast<std::uint32_t>(c.scissor >> 48) & 0x7FFu) << 16);
    w[12] = static_cast<std::uint32_t>(c.tex0) & 0x3FFFu;
    w[13] = static_cast<std::uint32_t>(c.tex0 >> 14) & 0x3Fu;
    w[14] = static_cast<std::uint32_t>(c.tex0 >> 20) & 0x3Fu;
    w[15] = std::min<std::uint32_t>(static_cast<std::uint32_t>(c.tex0 >> 26) & 0xFu, 10u) | (std::min<std::uint32_t>(static_cast<std::uint32_t>(c.tex0 >> 30) & 0xFu, 10u) << 8);
    w[16] = static_cast<std::uint32_t>(c.tex0 >> 51) & 0xFu;
    w[17] = static_cast<std::uint32_t>(c.tex0 >> 56) & 0x1Fu;
    w[18] = (static_cast<std::uint32_t>(c.clamp >> 4) & 0x3FFu) | ((static_cast<std::uint32_t>(c.clamp >> 14) & 0x3FFu) << 16);
    w[19] = (static_cast<std::uint32_t>(c.clamp >> 24) & 0x3FFu) | ((static_cast<std::uint32_t>(c.clamp >> 34) & 0x3FFu) << 16);
    w[20] = (static_cast<std::uint32_t>(st.texa) & 0xFFu) | ((static_cast<std::uint32_t>(st.texa >> 32) & 0xFFu) << 8);
    w[21] = static_cast<std::uint32_t>(st.fogcol) & 0xFFFFFFu;
    w[22] = g_clut_slot;
    w[23] = z_max(zpsm);
    w[24] = 0xFFFFFFFFu;
    return flags;
}

bool texel_rect(const std::uint32_t *setup, const Vertex *v, int count, std::uint32_t prim, bool fst, int rect[4]) {
    const int tw = 1 << (setup[15] & 0xFFu), th = 1 << (setup[15] >> 8);
    float u0 = 1e30f, v0 = 1e30f, u1 = -1e30f, v1 = -1e30f;
    for (int i = 0; i < count; ++i) {
        float u, t;
        if (fst) {
            u = static_cast<float>(v[i].u) / 16.0f;
            t = static_cast<float>(v[i].v) / 16.0f;
        } else {
            const float q = prim == 6 ? v[count - 1].q : v[i].q;
            if (!(q > 0.0f) || !(v[0].q > 0.0f) || !(v[count - 1].q > 0.0f)) return false;
            u = v[i].s / q * static_cast<float>(tw);
            t = v[i].t / q * static_cast<float>(th);
        }
        if (!std::isfinite(u) || !std::isfinite(t)) return false;
        u0 = std::min(u0, u);
        u1 = std::max(u1, u);
        v0 = std::min(v0, t);
        v1 = std::max(v1, t);
    }
    const auto axis = [](std::uint32_t mode, float lo, float hi, int size, std::uint32_t clamp, int &a, int &b) {
        a = static_cast<int>(std::floor(lo)) - 1;
        b = static_cast<int>(std::ceil(hi)) + 1;
        switch (mode) {
        case 0:
            if (a < 0 || b >= size) {
                a = 0;
                b = size - 1;
            }
            return true;
        case 1:
            a = std::clamp(a, 0, size - 1);
            b = std::clamp(b, 0, size - 1);
            return true;
        case 2: {
            const int minv = static_cast<int>(clamp & 0x3FFu), maxv = static_cast<int>(clamp >> 16);
            a = std::clamp(a, minv, std::max(minv, maxv));
            b = std::clamp(b, minv, std::max(minv, maxv));
            return true;
        }
        default: return false;
        }
    };
    int ua, ub, va, vb;
    if (!axis((setup[7] >> 24) & 3u, u0, u1, tw, setup[18], ua, ub) || !axis((setup[7] >> 26) & 3u, v0, v1, th, setup[19], va, vb)) return false;
    rect[0] = std::max(ua, 0);
    rect[1] = std::max(va, 0);
    rect[2] = std::max(ub, 0);
    rect[3] = std::max(vb, 0);
    return true;
}

bool shares_frame(const std::uint32_t *setup, const Batch &b) {
    const auto wide = [](std::uint32_t psm) { return psm == PSMCT32 || psm == PSMCT24 || psm == PSMT8H || psm == PSMT4HL || psm == PSMT4HH; };
    return setup[12] == b.fbp && setup[13] == b.fbw && (setup[14] == b.fpsm || (wide(setup[14]) && wide(b.fpsm)));
}

bool drawn_under(const Batch &b, const int rect[4]) {
    const auto hits = [&](int x0, int y0, int x1, int y1) { return rect[0] <= x1 && x0 <= rect[2] && rect[1] <= y1 && y0 <= rect[3]; };
    const std::size_t prims = b.boxes.size() / 2;
    if (prims > 64) return hits(b.minx, b.miny, b.maxx, b.maxy);
    for (std::size_t i = 0; i < prims; ++i) {
        const std::uint32_t lo = b.boxes[i * 2], hi = b.boxes[i * 2 + 1];
        if (hits(static_cast<int>(lo & 0xFFFFu), static_cast<int>(lo >> 16), static_cast<int>(hi & 0xFFFFu), static_cast<int>(hi >> 16))) return true;
    }
    return false;
}

void attrs(std::uint32_t *rec, int n, const Vertex &v, bool fst, const Vertex *color) {
    const Vertex &c = color ? *color : v;
    rec[12 + n] = c.r | (static_cast<std::uint32_t>(c.g) << 8) | (static_cast<std::uint32_t>(c.b) << 16) | (static_cast<std::uint32_t>(c.a) << 24);
    rec[15 + n * 3] = f2u(fst ? static_cast<float>(v.u) : v.s);
    rec[16 + n * 3] = f2u(fst ? static_cast<float>(v.v) : v.t);
    rec[17 + n * 3] = f2u(fst ? 1.0f : v.q);
    rec[11] |= static_cast<std::uint32_t>(v.f) << (8 * n);
}

}

Mode mode() { return g_mode; }

bool init(Mode m) {
    if (m == Mode::off) return false;
#ifdef __EMSCRIPTEN__
    g_instance = wgpuCreateInstance(nullptr);
#else
    WGPUInstanceExtras extras = {};
    extras.chain.sType = static_cast<WGPUSType>(WGPUSType_InstanceExtras);
    extras.backends = WGPUInstanceBackend_Primary;
    WGPUInstanceDescriptor id = WGPU_INSTANCE_DESCRIPTOR_INIT;
    id.nextInChain = &extras.chain;
    g_instance = wgpuCreateInstance(&id);
#endif
    if (!g_instance) return false;
    WGPUAdapter adapter = nullptr;
    bool adapter_done = false;
    WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    options.powerPreference = std::getenv("PS2_GPU_LOW_POWER") ? WGPUPowerPreference_LowPower : WGPUPowerPreference_HighPerformance;
    options.forceFallbackAdapter = std::getenv("PS2_GPU_FALLBACK") != nullptr;
    WGPURequestAdapterCallbackInfo acb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    acb.mode = WGPUCallbackMode_AllowSpontaneous;
    acb.callback = [](WGPURequestAdapterStatus status, WGPUAdapter a, WGPUStringView, void *out, void *done) {
        if (status == WGPURequestAdapterStatus_Success) *static_cast<WGPUAdapter *>(out) = a;
        *static_cast<bool *>(done) = true;
    };
    acb.userdata1 = &adapter;
    acb.userdata2 = &adapter_done;
    wgpuInstanceRequestAdapter(g_instance, &options, acb);
    while (!adapter_done) pump();
    if (!adapter) {
        std::fprintf(stderr, "[gpu] no adapter\n");
        return false;
    }
    WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
    if (wgpuAdapterGetInfo(adapter, &info) == WGPUStatus_Success) {
        std::fprintf(stderr, "[gpu] %.*s (%.*s)\n", static_cast<int>(info.device.length), info.device.data, static_cast<int>(info.description.length), info.description.data);
        wgpuAdapterInfoFreeMembers(info);
    }
    WGPULimits supported = WGPU_LIMITS_INIT;
    wgpuAdapterGetLimits(adapter, &supported);
    WGPULimits limits = WGPU_LIMITS_INIT;
    limits.maxStorageBufferBindingSize = supported.maxStorageBufferBindingSize;
    limits.maxBufferSize = supported.maxBufferSize;
    WGPUDeviceDescriptor dd = WGPU_DEVICE_DESCRIPTOR_INIT;
    dd.requiredLimits = &limits;
    g_timestamps = wgpuAdapterHasFeature(adapter, WGPUFeatureName_TimestampQuery);
    // PS2_GPU_NATIVE=1 or 0: draw what it can with the GPU's own rasterizer. On by default on OpenGL (the Switch),
    // where the compute rasterizer is too slow to play with; elsewhere it is there to be compared against.
    const char *native_spec = std::getenv("PS2_GPU_NATIVE");
#ifdef PS2_GPU_GL
    const bool want_native = !native_spec || std::atoi(native_spec) != 0;
#else
    const bool want_native = native_spec && std::atoi(native_spec) != 0;
#endif
    g_native = want_native && wgpuAdapterHasFeature(adapter, WGPUFeatureName_DualSourceBlending);
    if (want_native && !g_native) std::fprintf(stderr, "[gpu] no dual source blending: everything goes through the compute rasterizer\n");
    WGPUFeatureName features[2];
    if (g_timestamps) features[dd.requiredFeatureCount++] = WGPUFeatureName_TimestampQuery;
    if (g_native) features[dd.requiredFeatureCount++] = WGPUFeatureName_DualSourceBlending;
    dd.requiredFeatures = features;
    dd.uncapturedErrorCallbackInfo.callback = on_error;
    bool device_done = false;
    WGPURequestDeviceCallbackInfo dcb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    dcb.mode = WGPUCallbackMode_AllowSpontaneous;
    dcb.callback = [](WGPURequestDeviceStatus status, WGPUDevice d, WGPUStringView message, void *out, void *done) {
        if (status == WGPURequestDeviceStatus_Success) *static_cast<WGPUDevice *>(out) = d;
        else std::fprintf(stderr, "[gpu] device: %.*s\n", static_cast<int>(message.length), message.data);
        *static_cast<bool *>(done) = true;
    };
    dcb.userdata1 = &g_device;
    dcb.userdata2 = &device_done;
    wgpuAdapterRequestDevice(adapter, &dd, dcb);
    while (!device_done) pump();
    g_adapter = adapter;
    if (!g_device) return false;
    g_queue = wgpuDeviceGetQueue(g_device);

    if (const char *spec = std::getenv("PS2_GPU_PROFILE")) {
        char path[512] = {};
        unsigned long long a = 0, z = 0;
        if (std::sscanf(spec, "%llu,%llu,%511s", &a, &z, path) == 3) {
            g_profile_out = std::fopen(path, "w");
            g_profile_from = a;
            g_profile_to = z;
        }
    }
    if (const char *scale = std::getenv("PS2_SCALE")) g_scale = std::clamp<std::uint32_t>(static_cast<std::uint32_t>(std::atoi(scale)), 1u, 3u);
    g_copy_words = (1u << 20) * g_scale * g_scale;
    if (const char *tile = std::getenv("PS2_TILE")) {
        const int t = std::atoi(tile);
        if (t == 4 || t == 8 || t == 16) g_tile = t;
    }
    g_tiles_per_row = (2048 * static_cast<int>(g_scale) + g_tile - 1) / g_tile;
    g_tile_count.assign(static_cast<std::size_t>(g_tiles_per_row) * g_tiles_per_row, 0);
    g_tile_fill.assign(static_cast<std::size_t>(g_tiles_per_row) * g_tiles_per_row, 0);
    g_field_words = (1u << 19) * g_scale * g_scale;
    g_vram = make_buffer(static_cast<std::uint64_t>(kMemorySize) * g_scale * g_scale, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst);
    std::vector<std::uint32_t> tables;
    swizzle_tables(tables);
    g_tables = make_buffer(tables.size() * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    wgpuQueueWriteBuffer(g_queue, g_tables, 0, tables.data(), tables.size() * 4);
    g_clut = make_buffer(static_cast<std::uint64_t>(kClutSlots) * kClutEntries * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst);
    g_scratch = make_buffer((static_cast<std::uint64_t>(g_copy_words) + 2ull * g_field_words) * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    g_cache = make_buffer(static_cast<std::uint64_t>(kCacheWords) * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    g_decoded = make_buffer(static_cast<std::uint64_t>(kDecodedWords) * 4, WGPUBufferUsage_Storage);
    g_decoded_cache.clear();
    g_decode_free.clear();
    g_decode_retired.clear();
    g_decoded_used = 0;
    if (const char *decode = std::getenv("PS2_DECODE")) g_decode = std::atoi(decode) != 0;
    g_cache_copy.assign(kCacheWords, 0);
    g_readback = make_buffer(static_cast<std::uint64_t>(g_copy_words) * 4, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    if (g_timestamps) {
        WGPUQuerySetDescriptor qd = WGPU_QUERY_SET_DESCRIPTOR_INIT;
        qd.type = WGPUQueryType_Timestamp;
        qd.count = 2;
        g_queries = wgpuDeviceCreateQuerySet(g_device, &qd);
        g_query_resolve = make_buffer(16, WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc);
        g_query_read = make_buffer(16, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    }

    WGPUBindGroupLayoutEntry le[8] = {};
    const WGPUBufferBindingType types[8] = {WGPUBufferBindingType_Storage, WGPUBufferBindingType_ReadOnlyStorage, WGPUBufferBindingType_Uniform,
                                            WGPUBufferBindingType_ReadOnlyStorage, WGPUBufferBindingType_Storage, WGPUBufferBindingType_Storage,
                                            WGPUBufferBindingType_ReadOnlyStorage, WGPUBufferBindingType_Storage};
    for (int i = 0; i < 8; ++i) {
        le[i] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        le[i].binding = static_cast<std::uint32_t>(i);
        le[i].visibility = WGPUShaderStage_Compute;
        le[i].buffer.type = types[i];
        le[i].buffer.hasDynamicOffset = i == 2;
        le[i].buffer.minBindingSize = i == 2 ? kParamStride : 0;
    }
    WGPUBindGroupLayoutDescriptor ld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    ld.entryCount = 8;
    ld.entries = le;
    g_layout = wgpuDeviceCreateBindGroupLayout(g_device, &ld);
    WGPUPipelineLayoutDescriptor pld = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pld.bindGroupLayoutCount = 1;
    pld.bindGroupLayouts = &g_layout;
    g_pipeline_layout = wgpuDeviceCreatePipelineLayout(g_device, &pld);

    if (g_native) {
        // The compute pipelines that write a target back take its texture as a second group. Unfilterable float
        // fits a colour and a depth texture alike.
        WGPUBindGroupLayoutEntry te = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        te.binding = 0;
        te.visibility = WGPUShaderStage_Compute;
        te.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
        te.texture.viewDimension = WGPUTextureViewDimension_2D;
        WGPUBindGroupLayoutDescriptor tld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        tld.entryCount = 1;
        tld.entries = &te;
        g_target_layout = wgpuDeviceCreateBindGroupLayout(g_device, &tld);
        const WGPUBindGroupLayout both[2] = {g_layout, g_target_layout};
        WGPUPipelineLayoutDescriptor spd = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        spd.bindGroupLayoutCount = 2;
        spd.bindGroupLayouts = both;
        g_store_layout = wgpuDeviceCreatePipelineLayout(g_device, &spd);
        // The render pipelines see the compute pipelines' buffers, but for the upload cache: GS memory, the tables,
        // the CLUTs, the snapshot and the decoded textures in the fragment stage, the parameters and the data in both.
        WGPUBindGroupLayoutEntry ne[7] = {};
        const std::uint32_t bindings[7] = {0, 1, 2, 3, 4, 5, 7};
        for (int i = 0; i < 7; ++i) {
            ne[i] = le[bindings[i]];
            ne[i].visibility = bindings[i] == 2 || bindings[i] == 3 ? WGPUShaderStage_Vertex | WGPUShaderStage_Fragment : WGPUShaderStage_Fragment;
        }
        WGPUBindGroupLayoutDescriptor nld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        nld.entryCount = 7;
        nld.entries = ne;
        g_native_layout = wgpuDeviceCreateBindGroupLayout(g_device, &nld);
        WGPUPipelineLayoutDescriptor npd = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        npd.bindGroupLayoutCount = 1;
        npd.bindGroupLayouts = &g_native_layout;
        g_native_pipeline_layout = wgpuDeviceCreatePipelineLayout(g_device, &npd);
        // And those that read the pixel already there take the target's backdrop as a second group.
        WGPUBindGroupLayoutEntry be = te;
        be.visibility = WGPUShaderStage_Fragment;
        WGPUBindGroupLayoutDescriptor bld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        bld.entryCount = 1;
        bld.entries = &be;
        g_backdrop_layout = wgpuDeviceCreateBindGroupLayout(g_device, &bld);
        const WGPUBindGroupLayout with_backdrop[2] = {g_native_layout, g_backdrop_layout};
        WGPUPipelineLayoutDescriptor fpd = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        fpd.bindGroupLayoutCount = 2;
        fpd.bindGroupLayouts = with_backdrop;
        g_feedback_pipeline_layout = wgpuDeviceCreatePipelineLayout(g_device, &fpd);
        const std::string text = std::string("enable dual_source_blending;\n") + kGsShader + kNativeShader;
        WGPUShaderSourceWGSL native_source = WGPU_SHADER_SOURCE_WGSL_INIT;
        native_source.code = view(text.c_str());
        WGPUShaderModuleDescriptor nmd = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
        nmd.nextInChain = &native_source.chain;
        g_native_module = wgpuDeviceCreateShaderModule(g_device, &nmd);
        WGPUConstantEntry scale = WGPU_CONSTANT_ENTRY_INIT;
        scale.key = view("SCALE");
        scale.value = static_cast<double>(g_scale);
        for (int k = 0; k < 2 && g_native_module; ++k) {
            WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
            target.format = WGPUTextureFormat_RGBA8Unorm;
            WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
            fragment.module = g_native_module;
            fragment.entryPoint = view(k ? "load_depth_fs" : "load_color_fs");
            fragment.constantCount = 1;
            fragment.constants = &scale;
            fragment.targetCount = k ? 0 : 1;
            fragment.targets = k ? nullptr : &target;
            WGPUDepthStencilState depth = WGPU_DEPTH_STENCIL_STATE_INIT;
            depth.format = WGPUTextureFormat_Depth32Float;
            depth.depthWriteEnabled = WGPUOptionalBool_True;
            depth.depthCompare = WGPUCompareFunction_Always;
            WGPURenderPipelineDescriptor rd = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
            rd.layout = g_native_pipeline_layout;
            rd.vertex.module = g_native_module;
            rd.vertex.entryPoint = view("load_vs");
            rd.fragment = &fragment;
            if (k) rd.depthStencil = &depth;
            g_load_pipelines[k] = wgpuDeviceCreateRenderPipeline(g_device, &rd);
        }
        if (!g_native_module || !g_load_pipelines[0] || !g_load_pipelines[1] || !native_pipeline(1u | (2u << 1) | 8u | (0xFu << 4))) {
            std::fprintf(stderr, "[gpu] the native renderer's shaders did not build: everything goes through the compute rasterizer\n");
            g_native = false;
        }
    }

    WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
    source.code = view(kGsShader);
    WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    md.nextInChain = &source.chain;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(g_device, &md);
    for (int i = 0; i < kPipelines; ++i) {
        if (i >= kStoreColor && !g_native) continue;
        WGPUComputePipelineDescriptor cd = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        cd.layout = i >= kStoreColor ? g_store_layout : g_pipeline_layout;
        cd.compute.module = module;
        cd.compute.entryPoint = view(kEntryPoints[i]);
        WGPUConstantEntry constants[2] = {WGPU_CONSTANT_ENTRY_INIT, WGPU_CONSTANT_ENTRY_INIT};
        constants[0].key = view("SCALE");
        constants[0].value = static_cast<double>(g_scale);
        constants[1].key = view("TILE");
        constants[1].value = static_cast<double>(g_tile);
        cd.compute.constantCount = 2;
        cd.compute.constants = constants;
        g_pipelines[i] = wgpuDeviceCreateComputePipeline(g_device, &cd);
        if (!g_pipelines[i] && i >= kStoreColor) g_native = false;
        else if (!g_pipelines[i]) return false;
    }
    wgpuShaderModuleRelease(module);
    ensure_capacity();
    if (g_native) std::fprintf(stderr, "[gpu] native renderer on (PS2_GPU_NATIVE=0 turns it off)\n");
    g_mode = m;
    return true;
}

void draw(const State &st, const Vertex *v, int count) {
    const Timed timed(kCpuDraw);
    if (g_image.active && !g_image.pending.empty()) flush_image();
    std::uint64_t attr = st.prim;
    if (!(st.prmodecont & 1u)) attr = (st.prim & 7u) | (st.prmode & ~7ull);
    const std::uint32_t prim = static_cast<std::uint32_t>(st.prim) & 7u;
    const Context &c = st.ctx[(attr >> 9) & 1u];
    SetupCache &cache = g_setup_cache;
    if (cache.generation != g_setup_generation || cache.attr != attr || cache.clut != g_clut_slot) {
        cache.generation = g_setup_generation;
        cache.attr = attr;
        cache.clut = g_clut_slot;
        const std::uint32_t *w = cache.setup;
        cache.flags = make_setup(st, attr, cache.setup);
        cache.key = static_cast<std::uint64_t>(w[1]) | (static_cast<std::uint64_t>(w[2]) << 14) | (static_cast<std::uint64_t>(w[3]) << 20) |
                    (static_cast<std::uint64_t>(w[5]) << 26) | (static_cast<std::uint64_t>(w[6]) << 40);
        if (cache.flags & 2u) {
            const std::uint32_t tw = w[15] & 0xFFu, th = w[15] >> 8;
            const int tex_w = std::max(1 << tw, static_cast<int>(w[18] >> 16) + 1), tex_h = std::max(1 << th, static_cast<int>(w[19] >> 16) + 1);
            cache.tex = region(w[12], w[13], w[14], 0, 0, tex_w - 1, tex_h - 1);
        }
        NativeState state;
        cache.native = g_native && native_state(w, true, state);
        // The source alpha as a blend factor must not pass 0x80. The vertices' alpha is known here; a texture's is
        // not, and is taken to keep to 0x80, as games' textures do (one that does not blends as if it were 0x80).
        // With a texture's alpha in it: modulate gives at most the vertices' alpha, decal and highlight 2 the
        // texture's own, and highlight the sum of both, which is left to the compute rasterizer.
        const bool texture_alpha = (cache.flags & (2u | 8192u)) == (2u | 8192u);
        const std::uint32_t tfx = (w[7] >> 20) & 3u;
        cache.why = state.why;
        if (state.alpha_factor && texture_alpha && (tfx == 2 || (native_skip() & 128u))) {
            cache.native = false;
            cache.why = tfx == 2 ? 12 : 1;
        }
        cache.alpha_factor = state.alpha_factor && (!texture_alpha || tfx == 0);
    }
    const std::uint32_t *setup = cache.setup;
    const std::uint32_t flags = cache.flags;
    const bool fst = flags & 16u, iip = flags & 1u;
    const int ofx = static_cast<int>(c.xyoffset & 0xFFFFu), ofy = static_cast<int>((c.xyoffset >> 32) & 0xFFFFu);
    const int scax0 = static_cast<int>(setup[10] & 0xFFFFu), scax1 = static_cast<int>(setup[10] >> 16);
    const int scay0 = static_cast<int>(setup[11] & 0xFFFFu), scay1 = static_cast<int>(setup[11] >> 16);
    const auto sx = [&](const Vertex &p) { return (static_cast<float>(p.x) - static_cast<float>(ofx)) / 16.0f; };
    const auto sy = [&](const Vertex &p) { return (static_cast<float>(p.y) - static_cast<float>(ofy)) / 16.0f; };
    const float samples = static_cast<float>(g_scale);
    const auto first = [&](float lo) { return static_cast<int>(std::floor(std::ceil(lo * samples) / samples)); };

    std::uint32_t rec[kPrimWords] = {};
    int minx, miny, maxx, maxy;
    switch (prim) {
    case 3: case 4: case 5: {
        if (count < 3) return;
        const float x0 = sx(v[0]), y0 = sy(v[0]), x1 = sx(v[1]), y1 = sy(v[1]), x2 = sx(v[2]), y2 = sy(v[2]);
        const float area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if (area == 0.0f) return;
        minx = std::max(scax0, first(std::min({x0, x1, x2})));
        maxx = std::min(scax1, static_cast<int>(std::ceil(std::max({x0, x1, x2}))) - 1);
        miny = std::max(scay0, first(std::min({y0, y1, y2})));
        maxy = std::min(scay1, static_cast<int>(std::ceil(std::max({y0, y1, y2}))) - 1);
        rec[1] = 0;
        const float xy[6] = {x0, y0, x1, y1, x2, y2};
        for (int i = 0; i < 6; ++i) rec[4 + i] = f2u(xy[i]);
        rec[10] = f2u(1.0f / area);
        const float sg = area > 0.0f ? 1.0f : -1.0f;
        const auto inclusive = [&](float ax, float ay, float bx, float by) {
            const float gx = (ay - by) * sg, gy = (bx - ax) * sg;
            return gx > 0.0f || (gx == 0.0f && gy > 0.0f);
        };
        rec[27] = (inclusive(x1, y1, x2, y2) ? 1u : 0u) | (inclusive(x2, y2, x0, y0) ? 2u : 0u) | (inclusive(x0, y0, x1, y1) ? 4u : 0u);
        for (int i = 0; i < 3; ++i) attrs(rec, i, v[i], fst, iip ? nullptr : &v[2]);
        rec[24] = v[0].z;
        rec[25] = f2u(static_cast<float>(static_cast<std::int64_t>(v[1].z) - static_cast<std::int64_t>(v[0].z)));
        rec[26] = f2u(static_cast<float>(static_cast<std::int64_t>(v[2].z) - static_cast<std::int64_t>(v[0].z)));
        break;
    }
    case 6: {
        const float x0 = sx(v[0]), y0 = sy(v[0]), x1 = sx(v[1]), y1 = sy(v[1]);
        minx = std::max(scax0, static_cast<int>(std::ceil(std::min(x0, x1))));
        maxx = std::min(scax1, static_cast<int>(std::ceil(std::max(x0, x1))) - 1);
        miny = std::max(scay0, static_cast<int>(std::ceil(std::min(y0, y1))));
        maxy = std::min(scay1, static_cast<int>(std::ceil(std::max(y0, y1))) - 1);
        rec[1] = 1;
        rec[4] = f2u(x0);
        rec[5] = f2u(y0);
        rec[6] = f2u(x1 - x0);
        rec[7] = f2u(y1 - y0);
        attrs(rec, 0, v[0], fst, nullptr);
        attrs(rec, 1, v[1], fst, nullptr);
        rec[24] = v[1].z;
        break;
    }
    case 1: case 2: {
        const float x0 = sx(v[0]), y0 = sy(v[0]), x1 = sx(v[1]), y1 = sy(v[1]);
        const int steps = static_cast<int>(std::max(std::fabs(x1 - x0), std::fabs(y1 - y0)));
        minx = std::max(scax0, static_cast<int>(std::floor(std::min(x0, x1))) - 1);
        maxx = std::min(scax1, static_cast<int>(std::ceil(std::max(x0, x1))) + 1);
        miny = std::max(scay0, static_cast<int>(std::floor(std::min(y0, y1))) - 1);
        maxy = std::min(scay1, static_cast<int>(std::ceil(std::max(y0, y1))) + 1);
        rec[1] = 2;
        rec[4] = f2u(x0);
        rec[5] = f2u(y0);
        rec[6] = f2u(x1);
        rec[7] = f2u(y1);
        rec[8] = static_cast<std::uint32_t>(steps);
        attrs(rec, 0, v[0], fst, iip ? nullptr : &v[1]);
        attrs(rec, 1, v[1], fst, nullptr);
        rec[24] = v[0].z;
        rec[25] = f2u(static_cast<float>(static_cast<std::int64_t>(v[1].z) - static_cast<std::int64_t>(v[0].z)));
        break;
    }
    case 0: {
        const int x = static_cast<int>(sx(v[0])), y = static_cast<int>(sy(v[0]));
        minx = maxx = x;
        miny = maxy = y;
        if (x < scax0 || x > scax1 || y < scay0 || y > scay1) return;
        rec[1] = 3;
        rec[4] = static_cast<std::uint32_t>(x);
        rec[5] = static_cast<std::uint32_t>(y);
        attrs(rec, 0, v[0], fst, nullptr);
        rec[24] = v[0].z;
        break;
    }
    default:
        return;
    }
    minx = std::max(minx, 0);
    miny = std::max(miny, 0);
    maxx = std::min(maxx, 2047);
    maxy = std::min(maxy, 2047);
    if (minx > maxx || miny > maxy) return;

    rec[2] = static_cast<std::uint32_t>(minx) | (static_cast<std::uint32_t>(miny) << 16);
    rec[3] = static_cast<std::uint32_t>(maxx) | (static_cast<std::uint32_t>(maxy) << 16);
    Batch &b = g_batch;
    const bool zte = flags & 512u;
    // Triangles and sprites inside the buffer's width, with depths a Z24 buffer holds without clamping.
    std::uint32_t why = cache.native ? 0 : cache.why;
    if (!why && prim < 3) why = 13;
    if (!why && maxx >= static_cast<int>(setup[2] * 64)) why = 14;
    if (!why && (native_skip() & (prim == 6 ? 64u : 32u))) why = 1;
    // Depths the buffer holds without clamping, and a 32-bit float exactly.
    if (!why && zte && (prim == 6 ? v[1].z : std::max({v[0].z, v[1].z, v[2].z})) > std::min(setup[23], 0xFFFFFFu)) why = 15;
    // A source alpha above 0x80 is a blend factor above one, which only the compute rasterizer has.
    if (!why && cache.alpha_factor && (prim == 6 ? v[1].a : iip ? std::max({v[0].a, v[1].a, v[2].a}) : v[2].a) > 0x80u) why = 16;
    const bool native = cache.native && !why;
    if (g_native && !native) ++g_why_prims[why];
    const std::uint64_t key = cache.key | (native ? 1ull << 63 : 0ull);
    if (b.key != ~0ull && b.key != key) emit_all();
    const Range &tex = cache.tex;
    if (flags & 2u) {
        if (!b.prims.empty()) {
            batch_ranges();
            const auto written = [&](const Range &t) { return t.overlaps(b.fb_range.lo, b.fb_range.hi) || (b.zwrite && t.overlaps(b.z_range.lo, b.z_range.hi)); };
            int rect[4];
            if (written(tex)) {
                if (!texel_rect(setup, v, count, prim, fst, rect)) {
                    emit_all();
                } else {
                    const Range used = region(setup[12], setup[13], setup[14], rect[0], rect[1], rect[2], rect[3]);
                    const bool small = rect[2] - rect[0] < 64 && rect[3] - rect[1] < 64;
                    // A native batch reads its textures from GS memory, which holds the frame as it was before the batch:
                    // what the compute rasterizer's snapshot gives, but for a sprite reading the very pixels it writes.
                    const bool cached = b.texflushes == g_texflushes && small && shares_frame(setup, b) && !(b.native && reads_own_word(rec, setup));
                    const bool frame = !cached && (shares_frame(setup, b) ? drawn_under(b, rect) : used.overlaps(b.fb_range.lo, b.fb_range.hi));
                    if (frame || (b.zwrite && used.overlaps(b.z_range.lo, b.z_range.hi))) emit_all();
                }
            }
        }
    }
    if (b.prims.empty()) {
        b.texflushes = g_texflushes;
        b.key = key;
        b.fbp = setup[1];
        b.fbw = setup[2];
        b.fpsm = setup[3];
        b.zbp = setup[5];
        b.zpsm = setup[6];
        b.zte = false;
        b.zwrite = false;
        b.rx1 = -2;
        b.native = native;
        if (g_native && !native) ++g_why_batches[why];
    }
    b.zte = b.zte || zte;
    b.zwrite = b.zwrite || (zte && !(flags & 32u));
    const std::size_t setups = b.setups.size() / kSetupWords;
    std::uint32_t setup_index;
    if (setups && std::memcmp(b.setups.data() + (setups - 1) * kSetupWords, setup, kSetupWords * 4) == 0) {
        setup_index = static_cast<std::uint32_t>(setups - 1);
    } else {
        setup_index = static_cast<std::uint32_t>(setups);
        b.setups.insert(b.setups.end(), setup, setup + kSetupWords);
    }
    if (flags & 2u) {
        if (g_batch_reads.empty() || g_batch_reads.back().lo != tex.lo || g_batch_reads.back().hi != tex.hi) {
            bool known = false;
            for (const Range &t : g_batch_reads) known = known || (t.lo <= tex.lo && tex.hi <= t.hi);
            if (!known) g_batch_reads.push_back(tex);
        }
    }
    rec[0] = setup_index;
    b.prims.insert(b.prims.end(), rec, rec + kPrimWords);
    b.boxes.push_back(rec[2]);
    b.boxes.push_back(rec[3]);
    b.minx = std::min(b.minx, minx);
    b.miny = std::min(b.miny, miny);
    b.maxx = std::max(b.maxx, maxx);
    b.maxy = std::max(b.maxy, maxy);
    if (b.prims.size() >= kPrimWords * 65536) emit_all();
}

void image_begin(const State &s) {
    const Timed timed(kCpuImage);
    flush_image();
    Image &im = g_image;
    const std::uint64_t buf = s.bitbltbuf, pos = s.trxpos, reg = s.trxreg;
    im = Image{};
    im.dbp = static_cast<std::uint32_t>(buf >> 32) & 0x3FFFu;
    im.dbw = static_cast<std::uint32_t>(buf >> 48) & 0x3Fu;
    im.dpsm = static_cast<std::uint32_t>(buf >> 56) & 0x3Fu;
    im.dsax = static_cast<std::uint32_t>(pos >> 32) & 0x7FFu;
    im.dsay = static_cast<std::uint32_t>(pos >> 48) & 0x7FFu;
    im.rrw = static_cast<std::uint32_t>(reg) & 0xFFFu;
    im.total = im.rrw * (static_cast<std::uint32_t>(reg >> 32) & 0xFFFu);
    im.bpp = bits_per_pixel(im.dpsm);
    im.id = ++g_image_ids ? g_image_ids : ++g_image_ids;
    im.active = im.total != 0;
}

void image_data(const std::uint8_t *data, std::size_t bytes) {
    const Timed timed(kCpuImage);
    Image &im = g_image;
    if (!im.active) return;
    if (im.pending.empty() && !im.lead_bits) {
        const std::uint64_t used = queue_upload(data, bytes, 0);
        const std::size_t whole = static_cast<std::size_t>(used / 8);
        if (im.done < im.total && whole < bytes) im.pending.assign(data + whole, data + bytes);
        im.lead_bits = static_cast<std::uint32_t>(used % 8);
        finish_image_if_done();
        return;
    }
    im.pending.insert(im.pending.end(), data, data + bytes);
}

void local_copy(const State &s) {
    flush_all();
    g_lut.count_ok = false;  // rather than work out whether it was copied over
    const std::uint64_t buf = s.bitbltbuf, pos = s.trxpos, reg = s.trxreg;
    const std::uint32_t w = static_cast<std::uint32_t>(reg) & 0xFFFu, h = static_cast<std::uint32_t>(reg >> 32) & 0xFFFu;
    if (!w || !h) return;
    if (!g_surfaces.empty()) {
        const int sx = static_cast<int>(pos) & 0x7FF, sy = static_cast<int>(pos >> 16) & 0x7FF, dx = static_cast<int>(pos >> 32) & 0x7FF, dy = static_cast<int>(pos >> 48) & 0x7FF;
        sync_memory(pages(static_cast<std::uint32_t>(buf) & 0x3FFFu, static_cast<std::uint32_t>(buf >> 16) & 0x3Fu, static_cast<std::uint32_t>(buf >> 24) & 0x3Fu, sx, sy,
                          sx + static_cast<int>(w) - 1, sy + static_cast<int>(h) - 1), false);
        sync_memory(pages(static_cast<std::uint32_t>(buf >> 32) & 0x3FFFu, static_cast<std::uint32_t>(buf >> 48) & 0x3Fu, static_cast<std::uint32_t>(buf >> 56) & 0x3Fu, dx, dy,
                          dx + static_cast<int>(w) - 1, dy + static_cast<int>(h) - 1), true);
    }
    mark_rendered(pages(static_cast<std::uint32_t>(buf >> 32) & 0x3FFFu, static_cast<std::uint32_t>(buf >> 48) & 0x3Fu, static_cast<std::uint32_t>(buf >> 56) & 0x3Fu,
                        static_cast<int>(pos >> 32) & 0x7FF, static_cast<int>(pos >> 48) & 0x7FF, (static_cast<int>(pos >> 32) & 0x7FF) + static_cast<int>(w) - 1,
                        (static_cast<int>(pos >> 48) & 0x7FF) + static_cast<int>(h) - 1));
    const std::uint32_t read = push_params({static_cast<std::uint32_t>(buf) & 0x3FFFu, static_cast<std::uint32_t>(buf >> 16) & 0x3Fu, static_cast<std::uint32_t>(buf >> 24) & 0x3Fu,
                                            static_cast<std::uint32_t>(pos) & 0x7FFu, static_cast<std::uint32_t>(pos >> 16) & 0x7FFu, w, h});
    const std::uint32_t write = push_params({static_cast<std::uint32_t>(buf >> 32) & 0x3FFFu, static_cast<std::uint32_t>(buf >> 48) & 0x3Fu, static_cast<std::uint32_t>(buf >> 56) & 0x3Fu,
                                             static_cast<std::uint32_t>(pos >> 32) & 0x7FFu, static_cast<std::uint32_t>(pos >> 48) & 0x7FFu, w, h,
                                             static_cast<std::uint32_t>(pos >> 59) & 3u});
    dispatch_linear(kCopyRead, w * h * g_scale * g_scale, read);
    dispatch_linear(kCopyWrite, w * h * g_scale * g_scale, write);
}

void load_clut(std::uint64_t tex0, std::uint64_t texclut) {
    const Timed timed(kCpuClut);
    flush_image();
    if (g_clut_loads + 2 >= kClutSlots) {
        emit_everything();
        submit();
    }
    const std::uint32_t psm = static_cast<std::uint32_t>(tex0 >> 20) & 0x3Fu;
    const std::uint32_t cpsm = (static_cast<std::uint32_t>(tex0 >> 51) & 0xFu) == 0 ? PSMCT32 : static_cast<std::uint32_t>(tex0 >> 51) & 0xFu;
    const std::uint32_t next = (g_clut_slot + 1) % kClutSlots;
    const std::uint32_t tc = static_cast<std::uint32_t>(texclut);
    const std::uint32_t cbp = static_cast<std::uint32_t>(tex0 >> 37) & 0x3FFFu, csm = static_cast<std::uint32_t>(tex0 >> 55) & 1u;
    const std::uint32_t cbw = tc & 0x3Fu, cou = (tc >> 6) & 0x3Fu, cov = (tc >> 12) & 0x3FFu;
    PreOp op;
    op.clut = true;
    const std::uint32_t rec[10] = {g_clut_slot, next, cbp, cpsm, csm, static_cast<std::uint32_t>(tex0 >> 56) & 0x1Fu, (psm == PSMT8 || psm == PSMT8H) ? 1u : 0u, cbw, cou, cov};
    std::memcpy(op.rec, rec, sizeof rec);
    op.range = csm ? region(cbp, cbw, cpsm, static_cast<int>(cou * 16), static_cast<int>(cov), static_cast<int>(cou * 16 + 255), static_cast<int>(cov))
                   : region(cbp, 1, cpsm, 0, 0, 15, 15);
    queue_pre(op);
    std::uint64_t content = g_clut_content[g_clut_slot];
    for (int k = 2; k < 10; ++k) content = mix(content, rec[k]);
    g_clut_content[next] = mix(content, last_written(op.range));
    g_clut_source[next] = {cbp, cpsm == PSMCT32 && !csm && rec[6] != 0};
    g_clut_slot = next;
    ++g_clut_loads;
}

void texflush() { ++g_texflushes; }

void flush() {
    if (g_mode == Mode::off) return;
    flush_all();
    submit();
}

void wait_idle() {
    flush_all();
    submit();
    bool done = false;
    WGPUQueueWorkDoneCallbackInfo cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.callback = [](WGPUQueueWorkDoneStatus, WGPUStringView, void *flag, void *) { *static_cast<bool *>(flag) = true; };
    cb.userdata1 = &done;
    wgpuQueueOnSubmittedWorkDone(g_queue, cb);
    wait_until(done);
}

void sync_bytes(std::uint32_t offset, std::uint32_t bytes) {
    Range r;
    r.add(offset, static_cast<std::uint64_t>(offset) + bytes);
    sync_memory(r, false);
}

void sync_display(const DisplaySource &src) {
    if (g_surfaces.empty()) return;
    sync_memory(pages(src.fbp, src.fbw, src.psm, static_cast<int>(src.dbx), static_cast<int>(src.dby), static_cast<int>(src.dbx + src.width) - 1,
                      static_cast<int>(src.dby + src.height) - 1), false);
}

struct PendingDownload {
    WGPUBuffer buffer = nullptr;
    std::uint64_t capacity = 0;
    std::uint8_t *memory = nullptr;
    std::uint32_t offset = 0, bytes = 0, serial = 0;
    bool busy = false, ready = false;
} g_pending;

void collect_download() {
    if (!g_pending.busy || !g_pending.ready) return;
    const void *mapped = wgpuBufferGetConstMappedRange(g_pending.buffer, 0, g_pending.bytes);
    if (mapped) std::memcpy(g_pending.memory + g_pending.offset, mapped, g_pending.bytes);
    wgpuBufferUnmap(g_pending.buffer);
    g_pending.busy = g_pending.ready = false;
}

bool download_async(std::uint8_t *memory, std::uint32_t offset, std::uint32_t bytes, std::uint32_t serial) {
    collect_download();
    if (g_pending.busy) return false;
    if (g_pending.capacity < bytes) {
        if (g_pending.buffer) wgpuBufferRelease(g_pending.buffer);
        g_pending.capacity = bytes;
        g_pending.buffer = make_buffer(bytes, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    }
    flush_all();
    sync_bytes(offset, bytes);
    if (g_scale != 1) dispatch_linear(kResolve, bytes / 4, push_params({offset / 4}));
    submit();
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(enc, g_scale == 1 ? g_vram : g_scratch, offset, g_pending.buffer, 0, bytes);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(g_queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    g_pending.memory = memory;
    g_pending.offset = offset;
    g_pending.bytes = bytes;
    g_pending.serial = serial;
    g_pending.busy = true;
    WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void *, void *) {
        if (status == WGPUMapAsyncStatus_Success) g_pending.ready = true;
        else g_pending.busy = false;
    };
    wgpuBufferMapAsync(g_pending.buffer, WGPUMapMode_Read, 0, bytes, cb);
    return true;
}

bool finish_download(std::uint32_t serial) {
    if (!g_pending.busy || g_pending.serial != serial) return false;
    while (g_pending.busy && !g_pending.ready) pump();
    const bool done = g_pending.ready;
    collect_download();
    return done;
}

void download(std::uint8_t *memory, std::uint32_t offset, std::uint32_t bytes) {
    flush_all();
    sync_bytes(offset, bytes);
    if (g_scale == 1) {
        submit();
        read_back(g_vram, bytes, memory + offset, offset);
        return;
    }
    dispatch_linear(kResolve, bytes / 4, push_params({offset / 4}));
    submit();
    read_back(g_scratch, bytes, memory + offset, offset);
}

void read_state(std::vector<std::uint32_t> &vram, std::uint32_t &scale, std::vector<std::uint32_t> &clut) {
    flush_all();
    sync_all();
    submit();
    scale = g_scale;
    vram.resize(static_cast<std::size_t>(kMemorySize / 4) * g_scale * g_scale);
    read_back(g_vram, vram.size() * 4, vram.data());
    clut.resize(kClutEntries);
    read_back(g_clut, kClutEntries * 4, clut.data(), static_cast<std::uint64_t>(g_clut_slot) * kClutEntries * 4);
}

void write_state(const std::vector<std::uint32_t> &vram, std::uint32_t scale, const std::vector<std::uint32_t> &clut) {
    wait_idle();
    // GS memory is replaced whole: the native targets hold nothing worth keeping.
    for (const auto &s : g_surfaces) s->dirty = false;
    while (!g_surfaces.empty()) retire_surface(0);
    const std::size_t words = kMemorySize / 4;
    const std::uint32_t from = scale * scale, to = g_scale * g_scale;
    std::fill(std::begin(g_rendered), std::end(g_rendered), from == to && to > 1);
    wrote(Range{0, kMemorySize});
    g_decoded_cache.clear();
    g_decode_free.clear();
    g_decode_retired.clear();
    g_decoded_used = 0;
    g_clut_content[g_clut_slot] = mix(g_clut_content[g_clut_slot], g_write_clock);
    if (from == to && vram.size() == words * to) {
        wgpuQueueWriteBuffer(g_queue, g_vram, 0, vram.data(), vram.size() * 4);
    } else {
        std::vector<std::uint32_t> layers(words * to);
        for (std::size_t w = 0; w < words; ++w)
            for (std::uint32_t k = 0; k < to; ++k) layers[w * to + k] = vram[w * from];
        wgpuQueueWriteBuffer(g_queue, g_vram, 0, layers.data(), layers.size() * 4);
    }
    if (clut.size() >= kClutEntries) wgpuQueueWriteBuffer(g_queue, g_clut, static_cast<std::uint64_t>(g_clut_slot) * kClutEntries * 4, clut.data(), kClutEntries * 4);
    wait_idle();
}

std::uint64_t corrected_tiles() { return g_lut_tiles; }

void stats() {
    if (g_mode == Mode::off) return;
    std::fprintf(stderr, "[gpu] %llu submits, dispatches:", static_cast<unsigned long long>(g_submits));
    for (int i = 0; i < kPipelines; ++i) std::fprintf(stderr, " %s %llu", kEntryPoints[i], static_cast<unsigned long long>(g_dispatches[i]));
    std::fprintf(stderr, "\n");
    if (g_native)
        std::fprintf(stderr, "[gpu] native: %llu batches (%llu prims, %llu draws), compute: %llu batches (%llu prims); %llu stores, %llu loads, %zu targets, %zu pipelines\n",
                     static_cast<unsigned long long>(g_native_batches), static_cast<unsigned long long>(g_native_prims), static_cast<unsigned long long>(g_native_draws),
                     static_cast<unsigned long long>(g_compute_batches), static_cast<unsigned long long>(g_compute_prims), static_cast<unsigned long long>(g_stores),
                     static_cast<unsigned long long>(g_loads), g_surfaces.size(), g_native_pipelines.size());
    if (g_lut_tiles) std::fprintf(stderr, "[gpu] colour correction: %llu tiles as table lookups in %llu passes\n", static_cast<unsigned long long>(g_lut_tiles), static_cast<unsigned long long>(g_lut_passes));
}

void configure_surface() {
    WGPUSurfaceConfiguration config = WGPU_SURFACE_CONFIGURATION_INIT;
    config.device = g_device;
    config.format = g_surface_format;
    config.usage = WGPUTextureUsage_RenderAttachment;
    config.width = std::max(g_surface_width, 1u);
    config.height = std::max(g_surface_height, 1u);
    config.presentMode = g_present_mode;
    config.alphaMode = WGPUCompositeAlphaMode_Auto;
    wgpuSurfaceConfigure(g_surface, &config);
}

bool attach_surface(const SurfaceTarget &target) {
    if (g_mode != Mode::primary || !g_device) return false;
    WGPUSurfaceDescriptor sd = WGPU_SURFACE_DESCRIPTOR_INIT;
#ifdef __EMSCRIPTEN__
    WGPUEmscriptenSurfaceSourceCanvasHTMLSelector canvas = WGPU_EMSCRIPTEN_SURFACE_SOURCE_CANVAS_HTML_SELECTOR_INIT;
    canvas.selector = view(target.selector ? target.selector : "#canvas");
    sd.nextInChain = &canvas.chain;
#else
    WGPUSurfaceSourceXlibWindow xlib = WGPU_SURFACE_SOURCE_XLIB_WINDOW_INIT;
    WGPUSurfaceSourceWaylandSurface wayland = WGPU_SURFACE_SOURCE_WAYLAND_SURFACE_INIT;
    WGPUSurfaceSourceMetalLayer metal = WGPU_SURFACE_SOURCE_METAL_LAYER_INIT;
    if (target.kind == SurfaceTarget::Metal) {
        metal.layer = target.surface;
        sd.nextInChain = &metal.chain;
    } else if (target.kind == SurfaceTarget::X11) {
        xlib.display = target.display;
        xlib.window = target.window;
        sd.nextInChain = &xlib.chain;
    } else if (target.kind == SurfaceTarget::Wayland) {
        wayland.display = target.display;
        wayland.surface = target.surface;
        sd.nextInChain = &wayland.chain;
    } else {
        return false;
    }
#endif
    g_surface = wgpuInstanceCreateSurface(g_instance, &sd);
    if (!g_surface) return false;
    WGPUSurfaceCapabilities caps = WGPU_SURFACE_CAPABILITIES_INIT;
    if (wgpuSurfaceGetCapabilities(g_surface, g_adapter, &caps) != WGPUStatus_Success || !caps.formatCount) {
        wgpuSurfaceRelease(g_surface);
        g_surface = nullptr;
        return false;
    }
    g_surface_format = caps.formats[0];
    for (std::size_t i = 0; i < caps.formatCount; ++i)
        if (caps.formats[i] == WGPUTextureFormat_BGRA8Unorm || caps.formats[i] == WGPUTextureFormat_RGBA8Unorm) {
            g_surface_format = caps.formats[i];
            break;
        }
    g_present_mode = WGPUPresentMode_Fifo;
    for (const WGPUPresentMode wanted : {WGPUPresentMode_Immediate, WGPUPresentMode_Mailbox})
        for (std::size_t i = 0; i < caps.presentModeCount; ++i)
            if (caps.presentModes[i] == wanted) g_present_mode = wanted;
    wgpuSurfaceCapabilitiesFreeMembers(caps);
    g_surface_width = target.width;
    g_surface_height = target.height;
    configure_surface();

    WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
    source.code = view(kBlitShader);
    WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    md.nextInChain = &source.chain;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(g_device, &md);
    WGPUBindGroupLayoutEntry le[2] = {WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT};
    le[0].binding = 0;
    le[0].visibility = WGPUShaderStage_Fragment;
    le[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    le[1].binding = 1;
    le[1].visibility = WGPUShaderStage_Fragment;
    le[1].buffer.type = WGPUBufferBindingType_Uniform;
    WGPUBindGroupLayoutDescriptor ld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    ld.entryCount = 2;
    ld.entries = le;
    WGPUBindGroupLayout layout = wgpuDeviceCreateBindGroupLayout(g_device, &ld);
    WGPUPipelineLayoutDescriptor pld = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pld.bindGroupLayoutCount = 1;
    pld.bindGroupLayouts = &layout;
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(g_device, &pld);
    WGPUColorTargetState color = WGPU_COLOR_TARGET_STATE_INIT;
    color.format = g_surface_format;
    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    fragment.module = module;
    fragment.entryPoint = view("fs");
    fragment.targetCount = 1;
    fragment.targets = &color;
    WGPURenderPipelineDescriptor rd = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    rd.layout = pipeline_layout;
    rd.vertex.module = module;
    rd.vertex.entryPoint = view("vs");
    rd.fragment = &fragment;
    g_blit = wgpuDeviceCreateRenderPipeline(g_device, &rd);
    wgpuShaderModuleRelease(module);
    wgpuPipelineLayoutRelease(pipeline_layout);
    g_blit_params = make_buffer(32, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBindGroupEntry be[2] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT};
    be[0].binding = 0;
    be[0].buffer = g_scratch;
    be[0].size = WGPU_WHOLE_SIZE;
    be[1].binding = 1;
    be[1].buffer = g_blit_params;
    be[1].size = 32;
    WGPUBindGroupDescriptor bd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bd.layout = layout;
    bd.entryCount = 2;
    bd.entries = be;
    g_blit_group = wgpuDeviceCreateBindGroup(g_device, &bd);
    wgpuBindGroupLayoutRelease(layout);
    return g_blit != nullptr;
}

void resize_surface(std::uint32_t width, std::uint32_t height) {
    if (!g_surface || (width == g_surface_width && height == g_surface_height)) return;
    g_surface_width = width;
    g_surface_height = height;
    configure_surface();
}

bool has_surface() { return g_surface != nullptr; }

bool present(const DisplaySource &src) {
    const Timed timed(kCpuSubmit);
    if (!g_surface) return false;
    flush_all();
    sync_display(src);
    const std::uint32_t current = field_base(g_field), previous = field_base(g_field + 1);
    static const bool weave_off = std::getenv("PS2_NO_DEINTERLACE") != nullptr;
    const bool blend = src.halved && g_have_previous && g_last_width == src.width && g_last_height == src.height && !weave_off;
    const std::uint32_t param = push_params({src.fbp, src.fbw, src.psm, src.dbx, src.dby, src.width, src.height, src.halved ? 1u : 0u, current});
    g_commands.push_back({kDisplay, (src.width * g_scale + 7) / 8, (src.height * g_scale + 7) / 8, param});
    const float sw = static_cast<float>(g_surface_width), sh = static_cast<float>(g_surface_height);
    const float aspect = g_options.widescreen ? 16.0f / 9.0f : 4.0f / 3.0f;
    const float vw = std::min(sw, sh * aspect), vh = vw / aspect;
    const std::uint32_t blit[8] = {src.width * g_scale, src.height * g_scale, current, previous, blend ? 1u : 0u, static_cast<std::uint32_t>(vw), static_cast<std::uint32_t>(vh), ps2::video_options()};
    wgpuQueueWriteBuffer(g_queue, g_blit_params, 0, blit, sizeof blit);
    ++g_field;
    g_have_previous = true;
    g_last_width = src.width;
    g_last_height = src.height;
    WGPUSurfaceTexture st = WGPU_SURFACE_TEXTURE_INIT;
    wgpuSurfaceGetCurrentTexture(g_surface, &st);
    if (st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal && st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
        if (st.texture) wgpuTextureRelease(st.texture);
        submit();
        configure_surface();
        return true;
    }
    WGPUTextureView target = wgpuTextureCreateView(st.texture, nullptr);
    submit(target);
#ifndef __EMSCRIPTEN__
    wgpuSurfacePresent(g_surface);
#endif
    wgpuTextureViewRelease(target);
    wgpuTextureRelease(st.texture);
    return true;
}

double busy_ms() {
    if (!g_measured) return 0.0;
    return static_cast<double>(g_measured_ns) / static_cast<double>(g_measured) * static_cast<double>(g_total_passes) / 1e6;
}

struct Copy {
    WGPUBuffer buffer = nullptr;
    std::uint64_t size = 0, order = 0;
    std::uint32_t width = 0, height = 0;
    bool busy = false, ready = false;
};
Copy g_copies[3];
std::uint64_t g_copy_order = 0;

bool display_async(const DisplaySource &src, std::vector<std::uint32_t> &rgba, std::uint32_t &width, std::uint32_t &height) {
#ifdef __EMSCRIPTEN__
    wgpuInstanceProcessEvents(g_instance);
#else
    wgpuDevicePoll(g_device, false, nullptr);
#endif
    Copy *newest = nullptr;
    for (Copy &c : g_copies)
        if (c.ready && (!newest || c.order > newest->order)) newest = &c;
    if (newest) {
        const std::uint64_t bytes = static_cast<std::uint64_t>(newest->width) * newest->height * 4;
        const void *mapped = wgpuBufferGetConstMappedRange(newest->buffer, 0, bytes);
        if (mapped) {
            rgba.resize(static_cast<std::size_t>(newest->width) * newest->height);
            std::memcpy(rgba.data(), mapped, bytes);
            width = newest->width;
            height = newest->height;
        }
        for (Copy &c : g_copies)
            if (c.ready && c.order <= newest->order) {
                wgpuBufferUnmap(c.buffer);
                c.ready = c.busy = false;
            }
        if (!mapped) newest = nullptr;
    }
    Copy *slot = nullptr;
    for (Copy &c : g_copies)
        if (!c.busy) slot = &c;
    if (slot) {
        const std::uint64_t bytes = static_cast<std::uint64_t>(src.width) * src.height * g_scale * g_scale * 4;
        if (slot->size < bytes) {
            if (slot->buffer) wgpuBufferRelease(slot->buffer);
            slot->size = bytes;
            slot->buffer = make_buffer(bytes, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
        }
        flush_all();
        sync_display(src);
        const std::uint32_t param = push_params({src.fbp, src.fbw, src.psm, src.dbx, src.dby, src.width, src.height, src.halved ? 1u : 0u});
        g_commands.push_back({kDisplay, (src.width * g_scale + 7) / 8, (src.height * g_scale + 7) / 8, param});
        submit();
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_device, nullptr);
        wgpuCommandEncoderCopyBufferToBuffer(enc, g_scratch, 0, slot->buffer, 0, bytes);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
        wgpuQueueSubmit(g_queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
        slot->busy = true;
        slot->order = ++g_copy_order;
        slot->width = src.width * g_scale;
        slot->height = src.height * g_scale;
        WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        cb.mode = WGPUCallbackMode_AllowProcessEvents;
        cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void *copy, void *) {
            Copy &c = *static_cast<Copy *>(copy);
            if (status == WGPUMapAsyncStatus_Success) c.ready = true;
            else c.busy = false;
        };
        cb.userdata1 = slot;
        wgpuBufferMapAsync(slot->buffer, WGPUMapMode_Read, 0, bytes, cb);
    }
    return newest != nullptr;
}

bool display(const DisplaySource &src, std::vector<std::uint32_t> &rgba, std::uint32_t &width, std::uint32_t &height) {
    flush_all();
    sync_display(src);
    const std::uint32_t param = push_params({src.fbp, src.fbw, src.psm, src.dbx, src.dby, src.width, src.height, src.halved ? 1u : 0u});
    g_commands.push_back({kDisplay, (src.width * g_scale + 7) / 8, (src.height * g_scale + 7) / 8, param});
    submit();
    width = src.width * g_scale;
    height = src.height * g_scale;
    rgba.resize(static_cast<std::size_t>(width) * height);
    read_back(g_scratch, rgba.size() * 4, rgba.data());
    return true;
}

}
