#include "gs/gpu/path_profile.h"
#include <cstdio>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)
int main() {
    using namespace ps2::gs;
    std::uint32_t w[25] = {};
    w[0] = 2; w[3] = 0; w[14] = 19;
    auto base = path_key(w, 0);
    // Resource relocation and decoded-cache addresses must not fragment a family.
    for (unsigned index : {1u, 5u, 12u, 22u, 24u}) w[index] = 1234;
    CHECK(path_key(w, 0) == base);
    // State differences that affect the intended opaque-path candidate stay visible.
    for (unsigned index : {0u, 3u, 4u, 6u, 7u, 8u, 9u, 14u}) {
        ++w[index]; CHECK(path_key(w, 0) != base); --w[index];
    }
    CHECK(path_key(w, 1) != base);
    w[0] = 0; base = path_key(w, 0); w[14] = 17;
    CHECK(path_key(w, 0) == base); // Untextured draws ignore stale texture format.

    PathProfile p;
    PathCost c; c.batches = 1; c.prims = 9; c.native = 1; c.stores = 2;
    c.loads = 3; c.copies = 4; c.copy_pixels = 500; c.cpu_ms = .25;
    for (unsigned i = 0; i < 300; ++i) { auto key = base; key[4] = i; p.add(key, c); }
    CHECK(p.rows.size() == 256);
    CHECK(p.overflow.batches == 44);
    // Existing families still accumulate after reaching capacity.
    auto known = base; known[4] = 0;
    c.native = 0; p.add(known, c);
    CHECK(p.rows.at(known).batches == 2);
    CHECK(p.rows.at(known).native == 1);
    PathCost sum = p.overflow;
    for (const auto &[key, value] : p.rows) sum += value;
    CHECK(sum.batches == 301 && sum.prims == 2709 && sum.native == 300);
    CHECK(sum.stores == p.total.stores && sum.loads == p.total.loads);
    CHECK(sum.copy_pixels == 150500 && sum.copies == 1204 && sum.cpu_ms == 75.25);
    p.clear();
    CHECK(p.rows.empty() && p.overflow.batches == 0 && p.total.cpu_ms == 0);
    auto mixed = base; mixed[9] = 1; c.mixed = 1;
    p.add(mixed, c); c.mixed = 0; p.add(base, c);
    CHECK(p.rows.size() == 2 && p.total.mixed == 1);
    return 0;
}
