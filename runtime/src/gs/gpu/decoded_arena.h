#pragma once

#include <cstdint>

namespace ps2::gs {

// Check capacity without adding the requested size to the cursor: adding
// first can overflow and incorrectly admit an out-of-bounds allocation.
constexpr bool decoded_arena_can_append(std::uint32_t used, std::uint32_t words,
                                        std::uint32_t capacity) {
    return words != 0 && used <= capacity && words <= capacity - used;
}

}  // namespace ps2::gs
