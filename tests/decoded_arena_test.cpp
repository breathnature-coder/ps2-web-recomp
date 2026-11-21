#include "gs/gpu/decoded_arena.h"

#include <cstdio>
#include <cstdint>
#include <limits>

int main() {
    using ps2::gs::decoded_arena_can_append;
    constexpr std::uint32_t capacity = 1024;
    static_assert(decoded_arena_can_append(0, 1, capacity));
    static_assert(decoded_arena_can_append(0, capacity, capacity));
    static_assert(decoded_arena_can_append(1023, 1, capacity));
    static_assert(!decoded_arena_can_append(1023, 2, capacity));
    static_assert(!decoded_arena_can_append(capacity, 1, capacity));
    static_assert(!decoded_arena_can_append(capacity + 1, 1, capacity));
    static_assert(!decoded_arena_can_append(0, 0, capacity));
    static_assert(!decoded_arena_can_append(0, capacity + 1, capacity));
    constexpr auto max = std::numeric_limits<std::uint32_t>::max();
    static_assert(!decoded_arena_can_append(max, max, capacity));
    static_assert(decoded_arena_can_append(max - 1, 1, max));
    static_assert(!decoded_arena_can_append(max - 1, 2, max));
    // Regression: after a completed tail retirement rewinds the cursor,
    // append allocation becomes possible again.
    std::uint32_t used = capacity;
    if (decoded_arena_can_append(used, 64, capacity)) return 1;
    used -= 64;
    if (!decoded_arena_can_append(used, 64, capacity)) return 1;

    // Exhaustively compare small arenas with 64-bit reference arithmetic.
    // Include cursors beyond capacity and requests beyond the remaining
    // space; the test must still run when NDEBUG disables assert().
    for (std::uint32_t cap = 0; cap <= 128; ++cap) {
        for (std::uint32_t cursor = 0; cursor <= cap + 2; ++cursor) {
            for (std::uint32_t request = 0; request <= cap + 2; ++request) {
                const bool expected = request != 0 &&
                    static_cast<std::uint64_t>(cursor) + request <= cap;
                if (decoded_arena_can_append(cursor, request, cap) != expected) {
                    std::fprintf(stderr, "decoded arena mismatch: used=%u words=%u capacity=%u\n",
                                 cursor, request, cap);
                    return 1;
                }
            }
        }
    }
    // Exercise the upper 32-bit boundary, where adding cursor and
    // request in uint32_t would wrap. Use uint64_t as an independent oracle.
    constexpr std::uint32_t limits[] = {max - 2, max - 1, max};
    constexpr std::uint32_t requests[] = {0, 1, 2, 3, 64, 1024, max - 1, max};
    for (std::uint32_t cap : limits) {
        const std::uint32_t cursors[] = {0, 1, cap - 2, cap - 1, cap, max};
        for (std::uint32_t cursor : cursors) {
            for (std::uint32_t request : requests) {
                const bool expected = request != 0 &&
                    static_cast<std::uint64_t>(cursor) + request <= cap;
                if (decoded_arena_can_append(cursor, request, cap) != expected) {
                    std::fprintf(stderr, "decoded arena high-bound mismatch: used=%u words=%u capacity=%u\n",
                                 cursor, request, cap);
                    return 1;
                }
            }
        }
    }
    return 0;
}
