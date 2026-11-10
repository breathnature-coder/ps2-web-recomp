// Drives the GS with synthetic scenes in mirror mode (PS2_GS=both), where the
// software rasterizer and the GPU renderer both draw everything, and compares
// the 4 MiB of GS memory they end up with. No game data is involved.
//
//   gs_gpu_compare_test [scenes] [--dump file]
//
// It prints, per scene, how many bytes differ and where the first differences
// are. --dump writes the GPU's final memory, so two GPU backends can be
// compared with each other (cmp a.bin b.bin) even where both differ from the
// software reference; every scene's memory goes into the file, one after the
// other. The exit status is 0 when the GPU matched the software
// renderer in at least PS2_COMPARE_MIN_PERCENT (default 100) percent of scenes.
#include "gs/gpu/gpu.hpp"
#include "gs/gs.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ps2::gs;

namespace {

std::uint64_t g_rng = 0x2545F4914F6CDD1Dull;
std::uint32_t rnd() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return static_cast<std::uint32_t>(g_rng >> 11);
}
std::uint32_t pick(std::uint32_t n) { return rnd() % n; }
bool chance(std::uint32_t percent) { return pick(100) < percent; }

std::uint64_t bits(std::uint64_t v, int shift) { return v << shift; }
std::uint32_t f2u(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

enum Reg : std::uint32_t {
    PRIM = 0x00, RGBAQ = 0x01, ST = 0x02, UV = 0x03, XYZF2 = 0x04, XYZ2 = 0x05, TEX0_1 = 0x06, CLAMP_1 = 0x08, FOG = 0x0A,
    TEX1_1 = 0x14, XYOFFSET_1 = 0x18, PRMODECONT = 0x1A, TEXCLUT = 0x1C, TEXA = 0x3B, FOGCOL = 0x3D, TEXFLUSH = 0x3F,
    SCISSOR_1 = 0x40, ALPHA_1 = 0x42, DIMX = 0x44, DTHE = 0x45, COLCLAMP = 0x46, TEST_1 = 0x47, PABE = 0x49, FBA_1 = 0x4A,
    FRAME_1 = 0x4C, ZBUF_1 = 0x4E, BITBLTBUF = 0x50, TRXPOS = 0x51, TRXREG = 0x52, TRXDIR = 0x53,
};

constexpr std::uint32_t kOff = 2048 * 16;  // XYOFFSET of 2048 pixels, in 12.4

// Frame at block 0 (640x448 32-bit = 0x8C000 bytes), Z after it, textures from block 0x2000 (1 MiB) on.
constexpr std::uint32_t kFrameBp = 0, kZBp = 0x1180, kTexBp = 0x2000, kTex2Bp = 0x2800, kClutBp = 0x3F00;
constexpr int kW = 640, kH = 448;

void upload(std::uint32_t dbp, std::uint32_t dbw, std::uint32_t psm, std::uint32_t x, std::uint32_t y, std::uint32_t w, std::uint32_t h) {
    write(BITBLTBUF, bits(dbp, 32) | bits(dbw, 48) | bits(psm, 56));
    write(TRXPOS, bits(x, 32) | bits(y, 48));
    write(TRXREG, bits(w, 0) | bits(h, 32));
    write(TRXDIR, 0);
    std::uint32_t bpp = 32;
    if (psm == PSMCT24) bpp = 24;
    if (psm == PSMCT16 || psm == PSMCT16S) bpp = 16;
    if (psm == PSMT8 || psm == PSMT8H) bpp = 8;
    if (psm == PSMT4 || psm == PSMT4HL || psm == PSMT4HH) bpp = 4;
    const std::size_t bytes = static_cast<std::size_t>(w) * h * bpp / 8;
    std::vector<std::uint8_t> data((bytes + 15) / 16 * 16);
    // Smooth gradients with some noise, so bilinear filtering and blending have something to chew on.
    const std::uint32_t seed = rnd();
    for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<std::uint8_t>((i * 7 + (i / 64) * 13 + seed) ^ (chance(10) ? rnd() : 0u));
    write_image(data.data(), static_cast<std::uint32_t>(data.size() / 16));
}

void local_copy(std::uint32_t sbp, std::uint32_t dbp, std::uint32_t bw, std::uint32_t psm, std::uint32_t sx, std::uint32_t sy, std::uint32_t dx,
                std::uint32_t dy, std::uint32_t w, std::uint32_t h) {
    write(BITBLTBUF, bits(sbp, 0) | bits(bw, 16) | bits(psm, 24) | bits(dbp, 32) | bits(bw, 48) | bits(psm, 56));
    write(TRXPOS, bits(sx, 0) | bits(sy, 16) | bits(dx, 32) | bits(dy, 48));
    write(TRXREG, bits(w, 0) | bits(h, 32));
    write(TRXDIR, 2);
}

struct Scene {
    std::uint32_t fpsm, zpsm;
    bool texture;
    std::uint32_t tpsm;
};

std::uint32_t frame_psm() {
    static const std::uint32_t psms[] = {PSMCT32, PSMCT32, PSMCT24, PSMCT16, PSMCT16S};
    return psms[pick(5)];
}
std::uint32_t z_psm() {
    static const std::uint32_t psms[] = {PSMZ32, PSMZ24, PSMZ16, PSMZ16S};
    return psms[pick(4)];
}
std::uint32_t tex_psm() {
    static const std::uint32_t psms[] = {PSMCT32, PSMCT24, PSMCT16, PSMT8, PSMT4, PSMT8H, PSMT4HL, PSMT4HH};
    return psms[pick(8)];
}

// Every random draw is its own statement: the order C++ evaluates the operands
// of | or of a call's arguments in is unspecified, and GCC and clang differ,
// which would make the same seed build different scenes.
void setup_context(const Scene &sc) {
    write(PRMODECONT, 1);
    std::uint64_t frame = bits(kFrameBp / 32, 0) | bits(kW / 64, 16) | bits(sc.fpsm, 24);
    if (chance(15)) frame |= bits(rnd() & 0x00FF00FFu, 32);
    write(FRAME_1, frame);
    write(ZBUF_1, bits(kZBp / 32, 0) | bits(sc.zpsm & 0xF, 24) | (chance(10) ? bits(1, 32) : 0));
    write(XYOFFSET_1, bits(kOff, 0) | bits(kOff, 32));
    const std::uint32_t x0 = chance(20) ? pick(100) : 0;
    const std::uint32_t y0 = chance(20) ? pick(100) : 0;
    const std::uint32_t x1 = kW - 1 - (chance(20) ? pick(100) : 0);
    const std::uint32_t y1 = kH - 1 - (chance(20) ? pick(100) : 0);
    write(SCISSOR_1, bits(x0, 0) | bits(x1, 16) | bits(y0, 32) | bits(y1, 48));
    const bool ate = chance(30);
    const bool date = chance(10);
    const bool zte = chance(70);
    const std::uint32_t atst = pick(8), aref = pick(256), afail = pick(4), datm = pick(2), ztst = pick(4);
    write(TEST_1, (ate ? 1u : 0u) | bits(atst, 1) | bits(aref, 4) | bits(afail, 12) | (date ? bits(1, 14) : 0) | bits(datm, 15) | (zte ? bits(1, 16) : 0) |
                      bits(ztst, 17));
    const std::uint32_t a = pick(3), b = pick(3), c = pick(3), d = pick(3), fix = pick(256);
    write(ALPHA_1, bits(a, 0) | bits(b, 2) | bits(c, 4) | bits(d, 6) | bits(fix, 32));
    write(FBA_1, chance(15) ? 1 : 0);
    write(COLCLAMP, chance(85) ? 1 : 0);
    write(PABE, chance(10) ? 1 : 0);
    write(DTHE, chance(10) ? 1 : 0);
    const std::uint64_t dimx_lo = rnd();
    const std::uint64_t dimx_hi = rnd();
    write(DIMX, dimx_lo | (dimx_hi << 32));
    write(FOGCOL, rnd() & 0xFFFFFFu);
    const std::uint32_t ta0 = pick(256);
    const bool aem = chance(30);
    const std::uint32_t ta1 = pick(256);
    write(TEXA, bits(ta0, 0) | (aem ? bits(1, 15) : 0) | bits(ta1, 32));
    if (sc.texture) {
        const std::uint32_t tw = 4 + pick(5);  // 16..256
        const std::uint32_t th = 4 + pick(5);
        const std::uint32_t tbw = std::max<std::uint32_t>(1, (1u << tw) / 64);
        const bool clut = sc.tpsm == PSMT8 || sc.tpsm == PSMT4 || sc.tpsm == PSMT8H || sc.tpsm == PSMT4HL || sc.tpsm == PSMT4HH;
        const bool four = sc.tpsm == PSMT4 || sc.tpsm == PSMT4HL || sc.tpsm == PSMT4HH;
        const bool high = sc.tpsm == PSMT8H || sc.tpsm == PSMT4HL || sc.tpsm == PSMT4HH;  // their texels live in a 32-bit buffer
        const std::uint32_t tbw_reg = std::max<std::uint32_t>(tbw, (sc.tpsm == PSMT8 || sc.tpsm == PSMT4) ? 2u : 1u);
        if (high) upload(kTexBp, tbw, PSMCT32, 0, 0, 1u << tw, 1u << th);
        else upload(kTexBp, tbw_reg, sc.tpsm, 0, 0, 1u << tw, 1u << th);
        std::uint32_t cpsm = PSMCT32, csm = 0;
        if (clut) {
            cpsm = chance(50) ? PSMCT32 : PSMCT16;
            csm = chance(20) ? 1u : 0u;
            if (csm) upload(kClutBp, 4, cpsm, 0, 0, four ? 16 : 256, 1);
            else upload(kClutBp, 1, cpsm, 0, 0, four ? 8 : 16, four ? 2 : 16);
            if (csm) write(TEXCLUT, bits(4, 0) | bits(0, 6) | bits(0, 12));
        }
        const std::uint32_t tcc = pick(2), tfx = pick(4);
        const std::uint32_t csa = csm ? 0 : pick(2) * 16 / (four ? 1 : 16);
        const std::uint64_t tex0 = bits(kTexBp, 0) | bits(tbw_reg, 14) | bits(sc.tpsm, 20) | bits(tw, 26) | bits(th, 30) | bits(tcc, 34) | bits(tfx, 35) |
                                   bits(kClutBp, 37) | bits(cpsm, 51) | bits(csm, 55) | bits(csa, 56) | bits(clut ? 1 : 0, 61);
        write(TEXFLUSH, 0);
        write(TEX0_1, tex0);
        const std::uint32_t mmag = pick(2);
        const std::uint32_t mmin = pick(2);
        const std::uint32_t lcm = chance(20) ? 1 : 0;
        const std::uint32_t k = pick(64);
        write(TEX1_1, bits(lcm, 0) | bits(mmag, 5) | bits(mmin, 6) | bits(k, 32));
        const std::uint32_t wms = pick(4);
        const std::uint32_t wmt = pick(4);
        const std::uint32_t maxu = (1u << tw) - 1, maxv = (1u << th) - 1;
        const std::uint32_t minu_r = wms == 3 ? maxu >> 1 : pick(maxu + 1) / 2;
        const std::uint32_t maxu_r = wms == 3 ? 3 : maxu - pick(maxu / 2 + 1);
        const std::uint32_t minv_r = wmt == 3 ? maxv >> 1 : pick(maxv + 1) / 2;
        const std::uint32_t maxv_r = wmt == 3 ? 3 : maxv - pick(maxv / 2 + 1);
        write(CLAMP_1, bits(wms, 0) | bits(wmt, 2) | bits(minu_r, 4) | bits(maxu_r, 14) | bits(minv_r, 24) | bits(maxv_r, 34));
    }
}

void vertex(bool fst, bool fog, bool texture, int x, int y) {
    const std::uint32_t r = pick(256);
    const std::uint32_t g = pick(256);
    const std::uint32_t b = pick(256);
    const std::uint32_t a = pick(256);
    const float q = texture && !fst ? 0.5f + static_cast<float>(pick(1000)) / 500.0f : 1.0f;
    write(RGBAQ, r | (g << 8) | (b << 16) | (static_cast<std::uint64_t>(a) << 24) | (static_cast<std::uint64_t>(f2u(q)) << 32));
    if (texture) {
        if (fst) {
            const std::uint32_t u = pick(300 * 16);
            const std::uint32_t v = pick(300 * 16);
            write(UV, bits(u, 0) | bits(v, 16));
        } else {
            const float s = static_cast<float>(static_cast<int>(pick(3000)) - 500) / 1000.0f * q;
            const float t = static_cast<float>(static_cast<int>(pick(3000)) - 500) / 1000.0f * q;
            write(ST, f2u(s) | (static_cast<std::uint64_t>(f2u(t)) << 32));
        }
    }
    const std::uint32_t z = rnd();
    const std::uint32_t xx = static_cast<std::uint32_t>(x + 2048 * 16) & 0xFFFFu, yy = static_cast<std::uint32_t>(y + 2048 * 16) & 0xFFFFu;
    if (fog) {
        const std::uint64_t f = pick(256);
        write(XYZF2, xx | (yy << 16) | (static_cast<std::uint64_t>(z & 0xFFFFFFu) << 32) | (f << 56));
    } else {
        write(XYZ2, xx | (yy << 16) | (static_cast<std::uint64_t>(z) << 32));
    }
}

void draw_prims(const Scene &sc, int count) {
    for (int n = 0; n < count; ++n) {
        const std::uint32_t type = pick(10) < 2 ? 6u : pick(7);  // sprites are common in PS2 games
        const bool iip = chance(60);
        const bool fge = chance(15);
        const bool abe = chance(40);
        const bool fst = chance(50);
        const bool tme = sc.texture && chance(85);
        write(PRIM, type | (iip ? 8u : 0u) | (tme ? 16u : 0u) | (fge ? 32u : 0u) | (abe ? 64u : 0u) | (fst ? 256u : 0u));
        const int verts = type == 0 ? 1 : type <= 2 || type == 6 ? 2 : type == 3 ? 3 : 3 + static_cast<int>(pick(4));
        const int cx = static_cast<int>(pick(kW));
        const int cy = static_cast<int>(pick(kH));
        const int span = 8 + static_cast<int>(pick(chance(20) ? 400 : 120));
        for (int i = 0; i < verts; ++i) {
            int x = (cx + static_cast<int>(pick(static_cast<std::uint32_t>(span))) - span / 2) * 16;
            x += static_cast<int>(pick(16));
            int y = (cy + static_cast<int>(pick(static_cast<std::uint32_t>(span))) - span / 2) * 16;
            y += static_cast<int>(pick(16));
            vertex(fst, fge, tme, x, y);
        }
    }
}

int differences(const std::uint8_t *a, const std::uint8_t *b, std::size_t n, std::size_t &first) {
    int count = 0;
    first = n;
    for (std::size_t i = 0; i < n; ++i)
        if (a[i] != b[i]) {
            if (first == n) first = i;
            ++count;
        }
    return count;
}

}  // namespace

int main(int argc, char **argv) {
    int scenes = 40;
    const char *dump = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) dump = argv[++i];
        else scenes = std::atoi(argv[i]);
    }
    setenv("PS2_GS", "both", 0);
    render_reset();
    if (gpu::mode() != gpu::Mode::mirror) {
        std::fprintf(stderr, "no GPU renderer available\n");
        return 77;
    }
    std::vector<std::uint8_t> gpu_memory(kMemorySize);
    std::FILE *dump_file = dump ? std::fopen(dump, "wb") : nullptr;
    int matched = 0;
    for (int s = 0; s < scenes; ++s) {
        Scene sc{};
        sc.fpsm = frame_psm();
        sc.zpsm = z_psm();
        sc.texture = chance(70);
        sc.tpsm = tex_psm();
        setup_context(sc);
        // A clear, like games do at the start of a frame.
        write(PRIM, 6);
        write(RGBAQ, rnd());
        write(XYZ2, (kOff & 0xFFFFu) | (kOff << 16));
        write(XYZ2, ((kOff + kW * 16) & 0xFFFFu) | ((kOff + kH * 16) << 16));
        draw_prims(sc, 30 + static_cast<int>(pick(120)));
        if (chance(30)) {
            const std::uint32_t psm = chance(50) ? PSMCT32 : PSMCT16;
            const std::uint32_t sx = pick(200), sy = pick(200), dx = pick(200), dy = pick(100), w = 16 + pick(200), h = 16 + pick(150);
            local_copy(kFrameBp, kTex2Bp, kW / 64, psm, sx, sy, dx, dy, w, h);
        }
        if (chance(20)) {
            // Render to a texture, then draw with it: a feedback path games use for effects.
            Scene next{sc.fpsm, sc.zpsm, true, sc.fpsm == PSMCT24 ? PSMCT24 : sc.fpsm == PSMCT32 ? PSMCT32 : PSMCT16};
            write(TEXFLUSH, 0);
            write(TEX0_1, bits(kFrameBp, 0) | bits(kW / 64, 14) | bits(next.tpsm, 20) | bits(8, 26) | bits(8, 30) | bits(1, 34));
            draw_prims(next, 20);
        }
        gpu::download(gpu_memory.data());
        if (dump_file) std::fwrite(gpu_memory.data(), 1, gpu_memory.size(), dump_file);
        // The picture the display circuit would show, as the renderer converts it for the screen.
        std::vector<std::uint32_t> rgba;
        std::uint32_t width = 0, height = 0;
        gpu::display(gpu::DisplaySource{kFrameBp, kW / 64, sc.fpsm, 0, 0, kW, kH, false}, rgba, width, height);
        if (dump_file) std::fwrite(rgba.data(), 4, rgba.size(), dump_file);
        std::size_t first = 0;
        const int diff = differences(memory(), gpu_memory.data(), kMemorySize, first);
        if (!diff) ++matched;
        std::printf("scene %2d fpsm %02X zpsm %02X tex %d tpsm %02X: %s", s, sc.fpsm, sc.zpsm, sc.texture ? 1 : 0, sc.tpsm, diff ? "DIFF" : "ok");
        if (diff) std::printf(" %d bytes, first at 0x%06zX", diff, first);
        std::printf("\n");
    }
    if (dump_file) std::fclose(dump_file);
    const char *min = std::getenv("PS2_COMPARE_MIN_PERCENT");
    const int need = min ? std::atoi(min) : 100;
    std::printf("%d of %d scenes identical to the software renderer\n", matched, scenes);
    return matched * 100 >= need * scenes ? 0 : 1;
}
