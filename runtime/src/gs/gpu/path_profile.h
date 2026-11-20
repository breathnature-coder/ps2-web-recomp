#pragma once

#include <array>
#include <cstdint>
#include <map>

namespace ps2::gs {
struct SurfaceTransfer {
    std::uint64_t operations = 0, source_pixels = 0, native_rect_bytes = 0;
    void add(std::uint32_t width, std::uint32_t height, std::uint32_t scale) {
        ++operations;
        const std::uint64_t pixels = std::uint64_t(width) * height;
        source_pixels += pixels;
        // RGBA8 and Depth32Float native surfaces both occupy four bytes/texel.
        // Rectangle footprint, not measured memory-controller traffic.
        native_rect_bytes += pixels * scale * scale * 4;
    }
};
struct SurfaceTraffic {
    // Store: sync causes 0..5, eviction=6, explicit sync-all=7.
    std::array<SurfaceTransfer, 8> stores{};
    // Load: accumulated invalidation causes 0..5 as bits, initial allocation=bit 6, debug reload=bit 7.
    std::array<SurfaceTransfer, 256> loads{};
    void clear() { stores = {}; loads = {}; }
};
// A diagnostic family, NOT a rendering equivalence key. Addresses, dimensions,
// CLUT contents, fog and vertex values deliberately do not identify a family.
using PathKey = std::array<std::uint32_t, 10>;
inline PathKey path_key(const std::uint32_t *w, std::uint32_t kind) {
    return {w[0], w[3], w[4], w[6], w[7], w[8], w[9],
            (w[0] & 2u) ? w[14] : 0u, kind, 0u};
}
struct PathCost {
    std::uint64_t batches = 0, prims = 0, native = 0, mixed = 0;
    std::uint64_t stores = 0, loads = 0, copies = 0, copy_pixels = 0;
    double cpu_ms = 0;
    PathCost &operator+=(const PathCost &b) {
        batches += b.batches; prims += b.prims; native += b.native; mixed += b.mixed;
        stores += b.stores; loads += b.loads; copies += b.copies; copy_pixels += b.copy_pixels;
        cpu_ms += b.cpu_ms;
        return *this;
    }
};
struct PathProfile {
    static constexpr std::size_t capacity = 256;
    std::map<PathKey, PathCost> rows;
    PathCost overflow, total;
    void add(const PathKey &key, const PathCost &cost) {
        total += cost;
        auto it = rows.find(key);
        if (it != rows.end()) it->second += cost;
        else if (rows.size() < capacity) rows.emplace(key, cost);
        else overflow += cost;  // Never silently lose work or grow without bound.
    }
    void clear() { rows.clear(); overflow = {}; total = {}; }
};
} // namespace ps2::gs
