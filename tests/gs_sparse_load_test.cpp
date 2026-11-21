// Reuse deterministic GS register/upload helpers from the comparison harness.
#define main gs_compare_main
#include "gs_gpu_compare_test.cpp"
#undef main

int main() {
    setenv("PS2_GS", "both", 1);
    setenv("PS2_NATIVE_SMALL", "0", 1);
    render_reset();
    if (gpu::mode() != gpu::Mode::mirror) return 77;
    write(PRMODECONT, 1); write(XYOFFSET_1, 0);
    write(FRAME_1, bits(kW / 64, 16));
    write(ZBUF_1, bits(1, 32)); write(TEST_1, 0);
    write(SCISSOR_1, bits(kW - 1, 16) | bits(319, 48));
    write(COLCLAMP, 1); write(PABE, 0); write(DTHE, 0); write(FBA_1, 0);
    const auto rect = [](int top, int bottom, unsigned color) {
        write(PRIM, 6); write(RGBAQ, color);
        write(XYZ2, bits(top * 16, 16));
        write(XYZ2, bits(kW * 16, 0) | bits(bottom * 16, 16));
        gpu::flush();
    };
    std::vector<std::uint8_t> actual(kMemorySize);
    for (unsigned format : {PSMCT32, PSMCT24})
    for (int coordinates = 0; coordinates < 2; ++coordinates)
    for (int fragmented = 0; fragmented < 2; ++fragmented) {
        rect(0, 320, 0x80402010);
        gpu::download(actual.data()); // Establish a clean native target.
        rect(40, 56, 0x80224466); // Newer native pixels in a gap between written pages.
        const auto update = [&](unsigned y) {
            // Exercise both a shifted base and ordinary destination-Y uploads.
            upload(coordinates ? 0 : (y / 32) * (kW / 64) * 32, kW / 64, format,
                   0, coordinates ? y : 0, kW, 16);
        };
        update(0);
        if (fragmented) for (unsigned y : {64u, 128u, 192u}) update(y);
        update(256);
        gpu::flush();
        rect(300, 304, 0x80335577); // Reload stale rows without covering the dirty gap.
        gpu::download(actual.data());
        std::size_t first = 0;
        if (differences(memory(), actual.data(), kMemorySize, first)) {
            std::fprintf(stderr, "sparse fixture format=%u coordinates=%d fragmented=%d differs at %zu\n", format, coordinates, fragmented, first);
            return 1;
        }
    }
    gpu::stats();
    return 0;
}
