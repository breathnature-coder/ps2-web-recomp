// Prints a hash of the VU vector helpers' results over many inputs, including
// the edge cases the PS2 float rules care about. Build it with two compilers
// (clang and GCC, or x86-64 and AArch64) and compare the printed hashes: the
// runtime relies on these producing bit-identical results everywhere.
#include "ps2/vu_ops.hpp"

#include <cstdint>
#include <cstdio>

using namespace ps2::vuops;

namespace {

std::uint64_t state = 0x9E3779B97F4A7C15ull;
std::uint32_t next() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return static_cast<std::uint32_t>(state >> 16);
}

const std::uint32_t kEdges[] = {
    0x00000000u, 0x80000000u, 0x00000001u, 0x007FFFFFu, 0x00800000u, 0x80800000u, 0x3F800000u, 0xBF800000u,
    0x7F7FFFFFu, 0xFF7FFFFFu, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0x7FFFFFFFu, 0xFFFFFFFFu, 0x4F000000u,
    0xCF000000u, 0x4EFFFFFFu, 0x01000000u, 0xFF000000u, 0x00FFFFFFu, 0x7FFFFFFEu, 0x4B000000u, 0x5F000000u,
};

std::uint32_t pick() {
    const std::uint32_t r = next();
    if ((r & 3u) == 0u) return kEdges[(r >> 2) % (sizeof(kEdges) / sizeof(kEdges[0]))];
    return next();
}

v4u rnd() { return v4u{pick(), pick(), pick(), pick()}; }

std::uint64_t hash = 1469598103934665603ull;
void mix(v4u v) {
    for (int i = 0; i < 4; ++i) {
        hash ^= v[i];
        hash *= 1099511628211ull;
    }
}

}  // namespace

int main() {
    for (int n = 0; n < 2000000; ++n) {
        const v4u a = rnd(), b = rnd(), c = rnd();
        mix(vfix(a));
        mix(vadd(a, b));
        mix(vsub(a, b));
        mix(vmul(a, b));
        mix(vmadd(c, a, b));
        mix(vmsub(c, a, b));
        mix(vmax(a, b));
        mix(vmin(a, b));
        mix(vabs(a));
        const int frac = static_cast<int>(next() % 4u) * 4;  // FTOI0/4/12/15 use 0, 4, 12, 15
        mix(vftoi(a, frac == 8 ? 15 : frac));
        mix(vitof(a, frac == 8 ? 15 : frac));
    }
    std::printf("%016llx\n", static_cast<unsigned long long>(hash));
    return 0;
}
