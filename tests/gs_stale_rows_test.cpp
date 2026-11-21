#include "gs/gpu/stale_rows.h"
#include <cstdint>
#include <cstdio>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)
int main() {
    using ps2::gs::StaleRows;
    StaleRows rows;
    rows.add(0,31); rows.add(192,223);
    CHECK(rows.count == 2 && !rows.overlaps(64,127));
    // Dirty native pixels in the gap survive two disjoint GS-memory updates.
    int memory[256]{}, native[256]{};
    for (int i=64;i<128;++i) native[i]=7;
    for (int i=0;i<32;++i) memory[i]=1;
    for (int i=192;i<224;++i) memory[i]=2;
    for (unsigned i=0;i<rows.count;++i)
        for (int y=rows.spans[i].first;y<=rows.spans[i].last;++y) native[y]=memory[y];
    CHECK(native[0]==1 && native[100]==7 && native[200]==2);
    rows.add(40,47); rows.add(144,151); rows.add(240,247);
    CHECK(rows.count == 1 && rows.overlaps(64,127)); // collapse requires dirty writeback
    CHECK(rows.spans[0].first==0 && rows.spans[0].last==247);
    rows.clear(); rows.add(2000,2047); rows.add(0,1999);
    CHECK(rows.count==1 && rows.spans[0].last==2047);
    std::uint32_t rng=12345;
    for (int trial=0;trial<1000;++trial) {
        rows.clear(); bool required[2048]{};
        for (int step=0;step<100;++step) {
            rng=rng*1664525+1013904223; int a=rng%2048;
            rng=rng*1664525+1013904223; int b=std::min(2047,a+int(rng%64));
            rows.add(a,b);
            for(int y=a;y<=b;++y) required[y]=true;
            CHECK(rows.count<=4);
            for(unsigned i=1;i<rows.count;++i) CHECK(rows.spans[i-1].last+1<rows.spans[i].first);
            for(int y=0;y<2048;++y) CHECK(!required[y] || rows.overlaps(y,y));
        }
    }
    return 0;
}
