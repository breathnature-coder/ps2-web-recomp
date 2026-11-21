#pragma once
#include <algorithm>
#include <array>

namespace ps2::gs {
// Inclusive row intervals. Bounded fragmentation keeps a reload to <=4 draws.
// On overflow, the caller must preserve dirty pixels in the enlarged interval.
struct StaleRows {
    struct Span { int first, last; };
    std::array<Span, 5> spans{};
    unsigned count = 0;
    void clear() { count = 0; }
    void add(int first, int last) {
        if (first > last) return;
        spans[count++] = {first, last};
        std::sort(spans.begin(), spans.begin() + count,
                  [](Span a, Span b) { return a.first < b.first; });
        unsigned n = 0;
        for (unsigned i = 0; i < count; ++i) {
            if (n && spans[i].first <= spans[n - 1].last + 1)
                spans[n - 1].last = std::max(spans[n - 1].last, spans[i].last);
            else spans[n++] = spans[i];
        }
        count = n;
        if (count > 4) { spans[0].last = spans[count - 1].last; count = 1; }
    }
    bool overlaps(int first, int last) const {
        for (unsigned i = 0; i < count; ++i)
            if (first <= spans[i].last && spans[i].first <= last) return true;
        return false;
    }
};
} // namespace ps2::gs
