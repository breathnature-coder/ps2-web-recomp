#include <algorithm>
#include "gs/gpu/upload_span.h"
#include "gs/gs.hpp"
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); return 1; } } while (0)
int main() {
    using namespace ps2::gs;
    setenv("PS2_GS", "cpu", 1); init();
    UploadSpan span;
    CHECK(upload_span32(0,10,0,0,256,639,271,span));
    CHECK(span.lo==655360 && span.hi==737280);
    CHECK(!upload_span32(1,10,0,0,256,639,271,span));
    CHECK(!upload_span32(0,0,0,0,0,0,0,span));
    CHECK(!upload_span32(0,64,0,0,0,0,0,span));
    CHECK(!upload_span32(0,1,2,0,0,0,0,span));
    CHECK(!upload_span32(0,1,19,0,0,0,0,span));
    CHECK(!upload_span32(0,1,0,0,0,64,0,span));
    CHECK(!upload_span32(0,1,0,1,0,0,0,span));
    CHECK(!upload_span32(0,1,0,0,2,0,1,span));
    CHECK(!upload_span32(0,1,0,0,0,0,2048,span));
    CHECK(!upload_span32(16352,1,0,0,32,63,63,span));
    CHECK(upload_span32(16352,1,1,0,0,63,31,span));
    CHECK(span.hi==kMemorySize);
    unsigned checked=0; std::uint32_t rng=12345;
    const auto next=[&]() { rng=rng*1664525+1013904223; return rng; };
    for(unsigned i=0;i<10000;++i) {
        const auto bp=(next()%512)*32, bw=1+next()%32;
        const auto x0=next()%(bw*64), x1=std::min(bw*64-1,x0+next()%128);
        const auto y0=next()%2048, y1=std::min(2047u,y0+next()%64);
        if(!upload_span32(bp,bw,i%2,x0,y0,x1,y1,span)) continue;
        ++checked;
        // Independent existing GS swizzle implementation: every addressed word
        // must be inside the new sync span, including CT24's retained alpha byte.
        for(auto y=y0;y<=y1;++y) for(auto x=x0;x<=x1;++x) {
            const auto address=address32(bp,bw,x,y);
            CHECK(address>=span.lo && std::uint64_t(address)+4<=span.hi);
        }
    }
    CHECK(checked>100);
    return 0;
}
