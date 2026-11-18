// Standalone host regression test for the GL submission barrier gate.
// Build from the repository root:
// g++ -std=c++20 -Iruntime/include -Iruntime/src -Ithird_party/wgpu-native/include
//   tests/gs_gl_queue_reuse_gate_test.cpp runtime/src/gs/gpu/gl/glapi.cpp
//   runtime/src/gs/gpu/gl/shaders.cpp -Wl,-l:libEGL.so.1 -o /tmp/gs-gl-reuse-gate
// This includes the backend to exercise its actual command recorder/submitter.
// No GL context is needed: only the barrier entry point is replaced with a spy.
#include "gs/gpu/gl/webgpu_gl.cpp"

namespace {
std::vector<unsigned> events;
void barrier_spy(GLbitfield bits) {
    if (bits != GL_SHADER_STORAGE_BARRIER_BIT) std::abort();
    events.push_back(1);
}
void require_events(std::initializer_list<unsigned> expected) {
    if (events != std::vector<unsigned>(expected)) {
        std::fprintf(stderr, "decode reuse barrier ordering mismatch\n");
        std::exit(1);
    }
    events.clear();
}
}

int main() {
    unsetenv("PS2_GL_STATS");
    glMemoryBarrier = barrier_spy;
    WGPUCommandBufferImpl first, second;
    first.ops.emplace_back([] { events.push_back(2); });
    second.ops.emplace_back([] { events.push_back(3); });
    WGPUCommandBuffer commands[] = {&first, &second};

    // Default and explicit fallback must leave the old submission sequence intact.
    wgpuQueueSubmit(nullptr, 2, commands);
    require_events({2, 3});
    ps2GlSetDecodeQueueReuse(false);
    wgpuQueueSubmit(nullptr, 2, commands);
    require_events({2, 3});

    // The experiment needs exactly one barrier, before every command buffer.
    ps2GlSetDecodeQueueReuse(true);
    wgpuQueueSubmit(nullptr, 2, commands);
    require_events({1, 2, 3});
    wgpuQueueSubmit(nullptr, 0, nullptr);
    require_events({});

    // Reconfiguration must remove the barrier again, not leave stale backend state.
    ps2GlSetDecodeQueueReuse(false);
    wgpuQueueSubmit(nullptr, 1, commands);
    require_events({2});
    std::puts("PASS: default/off has no added barrier; on orders one barrier before commands; empty and reset paths pass");
}
