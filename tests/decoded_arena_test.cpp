#include "gs/gpu/decoded_arena.h"

#include <cassert>
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
    assert(!decoded_arena_can_append(used, 64, capacity));
    used -= 64;
    assert(decoded_arena_can_append(used, 64, capacity));
    return 0;
}
