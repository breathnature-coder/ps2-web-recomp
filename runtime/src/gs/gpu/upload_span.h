#pragma once
#include <cstdint>

namespace ps2::gs {
struct UploadSpan { std::uint64_t lo = 0, hi = 0; };
// Exact enclosing page range for a restricted, non-wrapping CT32/CT24 rectangle.
// Other formats/address layouts must keep the existing conservative dependency.
inline bool upload_span32(std::uint32_t bp, std::uint32_t bw, std::uint32_t psm,
                          std::uint32_t x0, std::uint32_t y0, std::uint32_t x1,
                          std::uint32_t y1, UploadSpan &out) {
    if (psm > 1 || !bw || bw > 63 || bp % 32 || bp >= 16384 ||
        x0 > x1 || y0 > y1 || x1 >= 2048 || y1 >= 2048 || x1 >= bw * 64) return false;
    const std::uint64_t base = std::uint64_t(bp) * 256;
    const auto lo = base + (std::uint64_t(y0 / 32) * bw + x0 / 64) * 8192;
    const auto hi = base + (std::uint64_t(y1 / 32) * bw + x1 / 64 + 1) * 8192;
    if (hi > 4u * 1024 * 1024) return false;
    out = {lo, hi};
    return true;
}
} // namespace ps2::gs
