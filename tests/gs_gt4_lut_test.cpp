// GT4's colour correction, as its race draws it (taken from a trace on hardware), against what it comes to: every
// pixel's red, green and blue each looked up in a 256-entry table, the alpha left alone.
//
// The game does it a 64x32 tile (one page of the frame) at a time:
//   1. 48 sprites read the tile's page as an 8-bit texture, so one byte of a pixel at a time, through the table as
//      a CLUT, into a scratch page: 16 for red, then 16 adding blue and 16 adding green.
//   2. 8 sprites copy the scratch page back over the tile, each through a quarter... an eighth of it as a CLUT and
//      a texture that just counts, which undoes the order the first step left the pixels in.
// The GPU renderer recognizes the pair and applies the table to the page directly (gpu.cpp, lut_*). This test
// draws the real sequence through the software renderer, checks that it is the lookup, and then that the GPU
// renderer, with the shortcut and without, leaves the same bytes.
#include "gs/gpu/gpu.hpp"
#include "gs/gs.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace ps2::gs;

namespace {

std::uint64_t g_rng = 0x9E3779B97F4A7C15ull;
std::uint32_t rnd() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return static_cast<std::uint32_t>(g_rng >> 11);
}
std::uint64_t bits(std::uint64_t v, int shift) { return v << shift; }

enum Reg : std::uint32_t {
    PRIM = 0x00, RGBAQ = 0x01, UV = 0x03, XYZ2 = 0x05, TEX0_1 = 0x06, CLAMP_1 = 0x08, TEX1_1 = 0x14, XYOFFSET_1 = 0x18, PRMODECONT = 0x1A,
    TEXCLUT = 0x1C, TEXA = 0x3B, TEXFLUSH = 0x3F, SCISSOR_1 = 0x40, ALPHA_1 = 0x42, DTHE = 0x45, COLCLAMP = 0x46, TEST_1 = 0x47, PABE = 0x49,
    FBA_1 = 0x4A, FRAME_1 = 0x4C, ZBUF_1 = 0x4E, BITBLTBUF = 0x50, TRXPOS = 0x51, TRXREG = 0x52, TRXDIR = 0x53,
};

// Where GT4 had them in the traced frame.
constexpr std::uint32_t kFrame = 0, kScratch = 0x3E20, kTable = 0x3E40, kCount = 0x3E44, kWidth = 640, kHeight = 224;

void upload(std::uint32_t bp, std::uint32_t bw, std::uint32_t psm, std::uint32_t w, std::uint32_t h, const std::vector<std::uint8_t> &bytes) {
    write(BITBLTBUF, bits(bp, 32) | bits(bw, 48) | bits(psm, 56));
    write(TRXPOS, 0);
    write(TRXREG, bits(w, 0) | bits(h, 32));
    write(TRXDIR, 0);
    std::vector<std::uint8_t> data((bytes.size() + 15) / 16 * 16);
    std::memcpy(data.data(), bytes.data(), bytes.size());
    write_image(data.data(), static_cast<std::uint32_t>(data.size() / 16));
}

void sprite(std::uint32_t x, std::uint32_t y, std::uint32_t w, std::uint32_t h, std::uint32_t u0, std::uint32_t v0, std::uint32_t u1, std::uint32_t v1) {
    write(UV, u0 | (v0 << 16));
    write(XYZ2, (x * 16) | ((y * 16) << 16));
    write(UV, u1 | (v1 << 16));
    write(XYZ2, ((x + w) * 16) | (((y + h) * 16) << 16));
}

void common() {
    write(PRMODECONT, 1);
    write(XYOFFSET_1, 0);
    write(ZBUF_1, bits(0, 0) | bits(0, 24) | bits(1, 32));  // Z32 at the frame's own address, masked
    write(TEST_1, bits(1, 16) | bits(1, 17));               // depth test on, always
    write(TEX1_1, 0);
    write(TEXA, bits(0x80, 32));
    write(COLCLAMP, 1);
    write(PABE, 0);
    write(FBA_1, 0);
    write(DTHE, 0);
}

// Step 1 for the tile whose page is at block pointer `page`.
void to_scratch(std::uint32_t page) {
    common();
    write(FRAME_1, bits(kScratch / 32, 0) | bits(10, 16) | bits(PSMCT32, 24) | bits(0xFF000000u, 32));
    write(SCISSOR_1, bits(0, 0) | bits(0x7FF, 16) | bits(0, 32) | bits(0x7FF, 48));
    write(ALPHA_1, bits(0, 0) | bits(2, 2) | bits(2, 4) | bits(1, 6) | bits(0x80, 32));  // Cs * 1.0 + Cd
    write(TEXFLUSH, 0);
    write(TEX0_1, bits(page, 0) | bits(2, 14) | bits(PSMT8, 20) | bits(7, 26) | bits(6, 30) | bits(1, 34) | bits(0, 35) | bits(kTable, 37) | bits(1, 61));
    static const struct { std::uint32_t color, u, v; bool blend, region; } kChannel[3] = {{0x00000080u, 0, 0, false, false}, {0x00800000u, 128, 0, true, false}, {0x00008000u, 64, 32, true, true}};
    for (const auto &c : kChannel) {
        write(CLAMP_1, c.region ? bits(3, 0) | bits(0x3F7, 4) : 0);
        write(PRIM, 6 | bits(1, 4) | bits(c.blend ? 1 : 0, 6) | bits(1, 8));
        write(RGBAQ, c.color);
        for (std::uint32_t i = 0; i < 16; ++i) {
            const std::uint32_t u = c.u + (i / 2) * 256, v = c.v + (i % 2) * 16;
            sprite((i % 8) * 8, (i / 8) * 16, 8, 16, u, v, u + 128, v + 1024);
        }
    }
}

// Step 2 for the tile at x, y.
void from_scratch(std::uint32_t x, std::uint32_t y) {
    common();
    write(FRAME_1, bits(kFrame / 32, 0) | bits(10, 16) | bits(PSMCT24, 24) | bits(0xFF000000u, 32));
    write(SCISSOR_1, bits(0, 0) | bits(kWidth - 1, 16) | bits(0, 32) | bits(kHeight - 1, 48));
    write(CLAMP_1, 0);
    write(PRIM, 6 | bits(1, 4) | bits(1, 8));
    write(RGBAQ, 0x00008000u);
    static const std::uint32_t kClut[8] = {0x00, 0x04, 0x10, 0x14, 0x08, 0x0C, 0x18, 0x1C};
    for (std::uint32_t k = 0; k < 8; ++k) {
        write(TEX0_1, bits(kCount, 0) | bits(1, 14) | bits(PSMT8, 20) | bits(3, 26) | bits(5, 30) | bits(1, 34) | bits(1, 35) | bits(kScratch + kClut[k], 37) | bits(1, 61));
        sprite(x + k * 8, y, 8, 32, 0, 0, 128, 512);
    }
}

// A CLUT's entry, CSM1: where in its 16 by 16 pixels index i is.
std::uint32_t clut_entry(const std::uint8_t *mem, std::uint32_t cbp, std::uint32_t i) {
    const std::uint32_t at = (i & 0xE7u) | ((i & 8u) << 1) | ((i & 16u) >> 1);
    std::uint32_t v;
    std::memcpy(&v, mem + address32(cbp, 1, at & 15u, at >> 4) , 4);
    return v;
}

}  // namespace

int main() {
    setenv("PS2_GS", "both", 0);
    render_reset();
    if (gpu::mode() != gpu::Mode::mirror) {
        std::fprintf(stderr, "no GPU renderer available\n");
        return 77;
    }
    // A frame of noise, a table of noise.
    std::vector<std::uint8_t> frame(kWidth * kHeight * 4), table(16 * 16 * 3);
    for (auto &b : frame) b = static_cast<std::uint8_t>(rnd());
    for (auto &b : table) b = static_cast<std::uint8_t>(rnd());
    upload(kFrame, 10, PSMCT32, kWidth, kHeight, frame);
    upload(kTable, 1, PSMCT24, 16, 16, table);
    flush_and_wait();
    const std::uint8_t *mem = memory();
    const auto pixel = [&](std::uint32_t x, std::uint32_t y) {
        std::uint32_t v;
        std::memcpy(&v, mem + address32(kFrame, 10, x, y), 4);
        return v;
    };
    std::vector<std::uint32_t> lut(256);
    for (std::uint32_t i = 0; i < 256; ++i) lut[i] = clut_entry(mem, kTable, i);
    const auto looked_up = [&](std::uint32_t p) { return (p & 0xFF000000u) | (lut[p & 0xFFu] & 0xFFu) | (lut[(p >> 8) & 0xFFu] & 0xFF00u) | (lut[(p >> 16) & 0xFFu] & 0xFF0000u); };

    // The first step alone on one tile: is every looked-up pixel of the tile in the CLUT its strip will use, and at
    // which index? That index, by position in the strip, is the counting texture.
    const std::uint32_t tx = 3 * 64, ty = 2 * 32, page = (3 + 2 * 10) * 32;
    std::vector<std::uint32_t> before(64 * 32);
    for (std::uint32_t y = 0; y < 32; ++y)
        for (std::uint32_t x = 0; x < 64; ++x) before[y * 64 + x] = pixel(tx + x, ty + y);
    to_scratch(page);
    flush_and_wait();
    static const std::uint32_t kClut[8] = {0x00, 0x04, 0x10, 0x14, 0x08, 0x0C, 0x18, 0x1C};
    std::vector<std::uint8_t> count(8 * 32);
    int missing = 0, disagree = 0;
    for (std::uint32_t k = 0; k < 8; ++k)
        for (std::uint32_t y = 0; y < 32; ++y)
            for (std::uint32_t x = 0; x < 8; ++x) {
                const std::uint32_t want = looked_up(before[y * 64 + k * 8 + x]) & 0xFFFFFFu;
                int found = -1;
                for (std::uint32_t i = 0; i < 256 && found < 0; ++i)
                    if ((clut_entry(mem, kScratch + kClut[k], i) & 0xFFFFFFu) == want) found = static_cast<int>(i);
                if (found < 0) ++missing;
                else if (k == 0) count[y * 8 + x] = static_cast<std::uint8_t>(found);
                else if (count[y * 8 + x] != found) ++disagree;
            }
    std::printf("step 1: %d of 2048 looked-up pixels missing from their CLUT, %d at another index than in the first strip\n", missing, disagree);
    if (missing || disagree) return 1;
    // The whole correction, 70 tiles as the game draws them, and then things that interrupt it: a step 1 without
    // its step 2, a tile done twice, a draw into the frame between two tiles.
    upload(kCount, 1, PSMT8, 8, 32, count);
    for (std::uint32_t t = 0; t < 70; ++t) {
        to_scratch(kFrame + t * 32);
        from_scratch((t % 10) * 64, (t / 10) * 32);
    }
    flush_and_wait();
    int wrong = 0;
    for (std::uint32_t y = 0; y < kHeight; ++y)
        for (std::uint32_t x = 0; x < kWidth; ++x) {
            std::uint32_t original;
            std::memcpy(&original, frame.data() + (y * kWidth + x) * 4, 4);
            if (pixel(x, y) != looked_up(original)) ++wrong;
        }
    std::printf("70 tiles: %d pixels of the software renderer's frame are not the lookup of what they were\n", wrong);
    std::vector<std::uint8_t> gpu_memory(kMemorySize);
    const auto differing = [&](const char *what) {
        flush_and_wait();
        gpu::download(gpu_memory.data());
        int bytes = 0;
        std::size_t first = 0, last = 0;
        for (std::size_t i = 0; i < kMemorySize; ++i)
            if (gpu_memory[i] != mem[i]) {
                if (!bytes++) first = i;
                last = i;
            }
        if (bytes) std::printf("  from %zX to %zX; at the first the GPU has %02X, the software renderer %02X\n", first, last, gpu_memory[first], mem[first]);
        std::printf("%s: %d bytes of GS memory differ between the GPU and the software renderer\n", what, bytes);
        return bytes;
    };
    int bad = wrong + differing("70 tiles");
    const std::uint64_t taken = gpu::corrected_tiles();
    to_scratch(kFrame + 5 * 32);
    bad += differing("a step 1 alone");
    to_scratch(kFrame + 7 * 32);
    from_scratch(7 * 64, 0);
    to_scratch(kFrame + 7 * 32);
    from_scratch(7 * 64, 0);
    bad += differing("a tile twice");
    to_scratch(kFrame + 12 * 32);
    from_scratch(2 * 64, 32);
    common();
    write(FRAME_1, bits(kFrame / 32, 0) | bits(10, 16) | bits(PSMCT32, 24));
    write(SCISSOR_1, bits(0, 0) | bits(kWidth - 1, 16) | bits(0, 32) | bits(kHeight - 1, 48));
    write(PRIM, 6);
    write(RGBAQ, 0x40302010u);
    sprite(100, 20, 150, 60, 0, 0, 0, 0);
    to_scratch(kFrame + 13 * 32);
    from_scratch(3 * 64, 32);
    to_scratch(kFrame + 14 * 32);
    from_scratch(4 * 64, 32 + 32);  // the wrong tile: not the correction, drawn as it is
    bad += differing("a draw between tiles, and a step 2 elsewhere");
    std::printf("%llu tiles taken as lookups\n", static_cast<unsigned long long>(gpu::corrected_tiles()));
    gpu::stats();
    const char *off = std::getenv("PS2_GT4_LUT");
    if (!(off && std::atoi(off) == 0) && taken != 70) {
        std::printf("the 70 tiles were not all recognized\n");
        return 1;
    }
    return bad ? 1 : 0;
}
