// The subset of the WebGPU C API that runtime/src/gs/gpu/gpu.cpp uses,
// implemented on OpenGL 4.3 core through EGL. With it the GS renderer runs
// unchanged where there is no WebGPU implementation, such as Switch homebrew
// (Mesa's nouveau driver through devkitPro's switch-mesa).
//
// The compute shaders are not translated at run time: scripts/gen_glsl.py has
// naga turn gs.wgsl into GLSL ahead of time (gl/shaders.cpp), and a pipeline
// is looked up by its entry point and SCALE constant.
//
// Semantics follow WebGPU where gpu.cpp relies on them: commands are recorded
// into encoders and run in order at wgpuQueueSubmit, buffers start zeroed,
// every dispatch sees the storage writes of the ones before it, and map and
// work-done callbacks fire from wgpuDevicePoll or wgpuInstanceProcessEvents
// once a fence says the GPU got there. Timestamp queries are not offered.
//
// Presenting: a surface is created from a WGPUSurfaceSourceXlibWindow whose
// display is null, and its window field is then taken as the platform's EGL
// native window (the Switch's NWindow *). Desktop X11 and Wayland windows are
// not supported; those builds use WebGPU itself.
#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>

#include "glapi.hpp"
#include "shaders.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace ps2::gs::gpu::gl;

struct WGPUInstanceImpl {
    int unused;
};
struct WGPUAdapterImpl {
    int unused;
};
struct WGPUDeviceImpl {
    int unused;
};
struct WGPUQueueImpl {
    int unused;
};
struct WGPUQuerySetImpl {
    int unused;
};
struct WGPUPipelineLayoutImpl {
    int unused;
};
struct WGPUTextureImpl {
    int unused;
};
struct WGPUTextureViewImpl {
    int unused;
};

struct WGPUBufferImpl {
    GLuint id = 0;
    std::uint64_t size = 0;
    WGPUBufferUsage usage = 0;
    std::vector<std::uint8_t> shadow;  // what a mapped (MapRead) buffer exposes
    bool mapped = false;
};

struct WGPUBindGroupLayoutImpl {
    struct Entry {
        std::uint32_t binding;
        bool uniform, dynamic;
    };
    std::vector<Entry> entries;
};

struct WGPUBindGroupImpl {
    struct Entry {
        std::uint32_t binding;
        GLuint buffer;
        std::uint64_t offset, size;
        bool uniform, dynamic;
    };
    std::vector<Entry> entries;  // in binding order, which is the order of dynamic offsets
};

struct WGPUShaderModuleImpl {
    bool blit;
};

struct WGPUComputePipelineImpl {
    GLuint program;
};

struct WGPURenderPipelineImpl {
    GLuint program, vao;
};

using Op = std::function<void()>;

struct WGPUCommandEncoderImpl {
    std::vector<Op> ops;
};
struct WGPUCommandBufferImpl {
    std::vector<Op> ops;
};
struct WGPUComputePassEncoderImpl {
    WGPUCommandEncoder encoder;
};
struct WGPURenderPassEncoderImpl {
    WGPUCommandEncoder encoder;
};

struct WGPUSurfaceImpl {
    void *window = nullptr;
    EGLSurface surface = nullptr;
    std::uint32_t width = 0, height = 0;
};

namespace {

struct Context {
    EGLDisplay display = nullptr;
    EGLConfig config = nullptr;
    EGLContext context = nullptr;
    EGLSurface draw = nullptr;  // the window surface once presenting, else a pbuffer or none
    std::string renderer, version;
    GLint max_storage = 0;
} g;

WGPUInstanceImpl g_instance;
WGPUAdapterImpl g_adapter;
WGPUDeviceImpl g_device;
WGPUQueueImpl g_queue;
WGPUQuerySetImpl g_query_set;
WGPUPipelineLayoutImpl g_pipeline_layout;
WGPUTextureImpl g_texture;
WGPUTextureViewImpl g_view;
std::uint64_t g_futures = 0;

WGPUStringView text(const char *s) { return WGPUStringView{s, s ? std::strlen(s) : 0}; }
std::string str(WGPUStringView v) {
    if (!v.data) return {};
    return v.length == WGPU_STRLEN ? std::string(v.data) : std::string(v.data, v.length);
}
WGPUFuture future() { return WGPUFuture{++g_futures}; }

void current() {
    if (g.context && eglGetCurrentContext() != g.context && !eglMakeCurrent(g.display, g.draw, g.draw, g.context))
        std::fprintf(stderr, "[gl] cannot make the context current here (EGL error 0x%X)\n", static_cast<unsigned>(eglGetError()));
}

bool create_context() {
    if (g.context) return true;
    // Desktop Mesa can run without any window system; elsewhere (the Switch) the default display is the screen.
#ifndef __SWITCH__
    using GetPlatformDisplay = EGLDisplay (*)(EGLenum, void *, const EGLint *);
    if (auto platform = reinterpret_cast<GetPlatformDisplay>(eglGetProcAddress("eglGetPlatformDisplayEXT"))) {
        const char *ext = eglQueryString(nullptr, EGL_EXTENSIONS);
        if (ext && std::strstr(ext, "EGL_MESA_platform_surfaceless") && !std::getenv("PS2_GL_DEFAULT_DISPLAY"))
            g.display = platform(EGL_PLATFORM_SURFACELESS_MESA, nullptr, nullptr);
    }
#endif
    if (!g.display) g.display = eglGetDisplay(nullptr);
    EGLint major = 0, minor = 0;
    if (!g.display || !eglInitialize(g.display, &major, &minor)) {
        std::fprintf(stderr, "[gl] no EGL display\n");
        return false;
    }
    if (!eglBindAPI(EGL_OPENGL_API)) {
        std::fprintf(stderr, "[gl] EGL has no desktop OpenGL\n");
        return false;
    }
    // Prefer a config that can also draw to a window, for presenting later.
    const EGLint surface_types[] = {EGL_WINDOW_BIT | EGL_PBUFFER_BIT, EGL_PBUFFER_BIT, 0};
    for (const EGLint surface : surface_types) {
        const EGLint attrs[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_SURFACE_TYPE, surface, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE};
        EGLint count = 0;
        if (eglChooseConfig(g.display, attrs, &g.config, 1, &count) && count == 1) break;
        g.config = nullptr;
    }
    const EGLint context_attrs[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 3, EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
    g.context = eglCreateContext(g.display, g.config, nullptr, context_attrs);
    if (!g.context) {
        std::fprintf(stderr, "[gl] cannot create an OpenGL 4.3 core context (EGL error 0x%X)\n", static_cast<unsigned>(eglGetError()));
        return false;
    }
    if (!eglMakeCurrent(g.display, nullptr, nullptr, g.context)) {
        // Without EGL_KHR_surfaceless_context a context needs some surface.
        const EGLint pbuffer_attrs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
        g.draw = g.config ? eglCreatePbufferSurface(g.display, g.config, pbuffer_attrs) : nullptr;
        if (!g.draw || !eglMakeCurrent(g.display, g.draw, g.draw, g.context)) {
            std::fprintf(stderr, "[gl] cannot make the context current\n");
            return false;
        }
    }
    if (!load_functions()) return false;
    g.renderer = reinterpret_cast<const char *>(glGetString(GL_RENDERER));
    g.version = reinterpret_cast<const char *>(glGetString(GL_VERSION));
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &g.max_storage);
    std::fprintf(stderr, "[gl] %s, OpenGL %s\n", g.renderer.c_str(), g.version.c_str());
    return true;
}

// Callbacks waiting for the GPU, oldest first.
struct Pending {
    GLsync fence;
    std::function<void()> done;
};
std::deque<Pending> g_pending;

void fence(std::function<void()> done) {
    current();
    g_pending.push_back({glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0), std::move(done)});
    glFlush();
}

void process(bool wait) {
    current();
    while (!g_pending.empty()) {
        Pending &p = g_pending.front();
        GLenum r = glClientWaitSync(p.fence, GL_SYNC_FLUSH_COMMANDS_BIT, 0);
        while (wait && r == GL_TIMEOUT_EXPIRED) r = glClientWaitSync(p.fence, GL_SYNC_FLUSH_COMMANDS_BIT, 100'000'000);
        if (r == GL_TIMEOUT_EXPIRED) return;  // later fences cannot be done either
        if (r == GL_WAIT_FAILED) std::fprintf(stderr, "[gl] fence wait failed\n");
        glDeleteSync(p.fence);
        std::function<void()> done = std::move(p.done);
        g_pending.pop_front();
        done();  // may queue more work; the loop picks it up
    }
}

GLuint compile(GLenum type, const char *source, const char *name) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        std::fprintf(stderr, "[gl] %s does not compile:\n%s\n", name, log.c_str());
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

GLuint link(std::initializer_list<GLuint> shaders, const char *name) {
    const GLuint program = glCreateProgram();
    for (GLuint s : shaders) glAttachShader(program, s);
    glLinkProgram(program);
    for (GLuint s : shaders) glDeleteShader(s);
    GLint ok = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
        glGetProgramInfoLog(program, length, nullptr, log.data());
        std::fprintf(stderr, "[gl] %s does not link:\n%s\n", name, log.c_str());
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

const char *find_shader(const std::string &entry, std::uint32_t scale) {
    for (std::size_t i = 0; i < kShaderCount; ++i)
        if (kShaders[i].scale == scale && entry == kShaders[i].entry) return kShaders[i].source;
    return nullptr;
}

std::map<std::pair<std::string, std::uint32_t>, GLuint> g_programs;

void bind(const WGPUBindGroupImpl &group, std::vector<std::uint32_t> dynamic) {
    std::size_t next = 0;
    for (const auto &e : group.entries) {
        std::uint64_t offset = e.offset;
        if (e.dynamic && next < dynamic.size()) offset += dynamic[next++];
        glBindBufferRange(e.uniform ? GL_UNIFORM_BUFFER : GL_SHADER_STORAGE_BUFFER, e.binding, e.buffer, static_cast<GLintptr>(offset),
                          static_cast<GLsizeiptr>(e.size));
    }
}

}  // namespace

extern "C" {

WGPUInstance wgpuCreateInstance(WGPUInstanceDescriptor const *) { return &g_instance; }

void wgpuInstanceProcessEvents(WGPUInstance) {
    if (g.context) process(false);
}

WGPUFuture wgpuInstanceRequestAdapter(WGPUInstance, WGPURequestAdapterOptions const *, WGPURequestAdapterCallbackInfo cb) {
    if (create_context()) cb.callback(WGPURequestAdapterStatus_Success, &g_adapter, text(""), cb.userdata1, cb.userdata2);
    else cb.callback(WGPURequestAdapterStatus_Unavailable, nullptr, text("no OpenGL 4.3"), cb.userdata1, cb.userdata2);
    return future();
}

WGPUStatus wgpuAdapterGetInfo(WGPUAdapter, WGPUAdapterInfo *info) {
    info->device = text(g.renderer.c_str());
    info->description = text(g.version.c_str());
    info->backendType = WGPUBackendType_OpenGL;
    return WGPUStatus_Success;
}

void wgpuAdapterInfoFreeMembers(WGPUAdapterInfo) {}

WGPUStatus wgpuAdapterGetLimits(WGPUAdapter, WGPULimits *limits) {
    limits->maxStorageBufferBindingSize = static_cast<std::uint64_t>(std::max(g.max_storage, 0));
    limits->maxBufferSize = static_cast<std::uint64_t>(std::max(g.max_storage, 0));
    return WGPUStatus_Success;
}

WGPUBool wgpuAdapterHasFeature(WGPUAdapter, WGPUFeatureName) { return false; }

WGPUFuture wgpuAdapterRequestDevice(WGPUAdapter, WGPUDeviceDescriptor const *, WGPURequestDeviceCallbackInfo cb) {
    cb.callback(WGPURequestDeviceStatus_Success, &g_device, text(""), cb.userdata1, cb.userdata2);
    return future();
}

WGPUQueue wgpuDeviceGetQueue(WGPUDevice) { return &g_queue; }

WGPUBool wgpuDevicePoll(WGPUDevice, WGPUBool wait, WGPUSubmissionIndex const *) {
    process(wait);
    return g_pending.empty();
}

WGPUBuffer wgpuDeviceCreateBuffer(WGPUDevice, WGPUBufferDescriptor const *d) {
    current();
    auto *b = new WGPUBufferImpl;
    b->size = d->size;
    b->usage = d->usage;
    glGenBuffers(1, &b->id);
    glBindBuffer(GL_COPY_WRITE_BUFFER, b->id);
    // WebGPU buffers start zeroed, and the renderer counts on it for GS memory.
    const std::vector<std::uint8_t> zeros(static_cast<std::size_t>(d->size));
    glBufferData(GL_COPY_WRITE_BUFFER, static_cast<GLsizeiptr>(d->size), zeros.data(), (d->usage & WGPUBufferUsage_MapRead) ? GL_STREAM_READ : GL_DYNAMIC_DRAW);
    return b;
}

void wgpuBufferRelease(WGPUBuffer b) {
    if (!b) return;
    current();
    glDeleteBuffers(1, &b->id);
    delete b;
}

WGPUFuture wgpuBufferMapAsync(WGPUBuffer b, WGPUMapMode, size_t offset, size_t size, WGPUBufferMapCallbackInfo cb) {
    fence([b, offset, size, cb] {
        b->shadow.resize(static_cast<std::size_t>(b->size));
        glBindBuffer(GL_COPY_READ_BUFFER, b->id);
        glGetBufferSubData(GL_COPY_READ_BUFFER, static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(size), b->shadow.data() + offset);
        b->mapped = true;
        cb.callback(WGPUMapAsyncStatus_Success, text(""), cb.userdata1, cb.userdata2);
    });
    return future();
}

void const *wgpuBufferGetConstMappedRange(WGPUBuffer b, size_t offset, size_t) { return b->mapped ? b->shadow.data() + offset : nullptr; }

void wgpuBufferUnmap(WGPUBuffer b) { b->mapped = false; }

WGPUFuture wgpuQueueOnSubmittedWorkDone(WGPUQueue, WGPUQueueWorkDoneCallbackInfo cb) {
    fence([cb] { cb.callback(WGPUQueueWorkDoneStatus_Success, text(""), cb.userdata1, cb.userdata2); });
    return future();
}

void wgpuQueueWriteBuffer(WGPUQueue, WGPUBuffer b, uint64_t offset, void const *data, size_t size) {
    if (!size) return;
    current();
    glBindBuffer(GL_COPY_WRITE_BUFFER, b->id);
    glBufferSubData(GL_COPY_WRITE_BUFFER, static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(size), data);
}

void wgpuQueueSubmit(WGPUQueue, size_t count, WGPUCommandBuffer const *commands) {
    current();
    for (std::size_t i = 0; i < count; ++i)
        for (const Op &op : commands[i]->ops) op();
}

WGPUQuerySet wgpuDeviceCreateQuerySet(WGPUDevice, WGPUQuerySetDescriptor const *) { return &g_query_set; }

WGPUBindGroupLayout wgpuDeviceCreateBindGroupLayout(WGPUDevice, WGPUBindGroupLayoutDescriptor const *d) {
    auto *l = new WGPUBindGroupLayoutImpl;
    for (std::size_t i = 0; i < d->entryCount; ++i) {
        const WGPUBindGroupLayoutEntry &e = d->entries[i];
        l->entries.push_back({e.binding, e.buffer.type == WGPUBufferBindingType_Uniform, e.buffer.hasDynamicOffset != 0});
    }
    return l;
}

void wgpuBindGroupLayoutRelease(WGPUBindGroupLayout l) { delete l; }

WGPUPipelineLayout wgpuDeviceCreatePipelineLayout(WGPUDevice, WGPUPipelineLayoutDescriptor const *) { return &g_pipeline_layout; }

void wgpuPipelineLayoutRelease(WGPUPipelineLayout) {}

WGPUBindGroup wgpuDeviceCreateBindGroup(WGPUDevice, WGPUBindGroupDescriptor const *d) {
    auto *group = new WGPUBindGroupImpl;
    for (std::size_t i = 0; i < d->entryCount; ++i) {
        const WGPUBindGroupEntry &e = d->entries[i];
        bool uniform = false, dynamic = false;
        for (const auto &le : d->layout->entries)
            if (le.binding == e.binding) uniform = le.uniform, dynamic = le.dynamic;
        const std::uint64_t size = e.size == WGPU_WHOLE_SIZE ? e.buffer->size - e.offset : e.size;
        group->entries.push_back({e.binding, e.buffer->id, e.offset, size, uniform, dynamic});
    }
    std::sort(group->entries.begin(), group->entries.end(), [](const auto &a, const auto &b) { return a.binding < b.binding; });
    return group;
}

void wgpuBindGroupRelease(WGPUBindGroup group) { delete group; }

WGPUShaderModule wgpuDeviceCreateShaderModule(WGPUDevice, WGPUShaderModuleDescriptor const *d) {
    bool blit = false;
    for (const WGPUChainedStruct *c = d->nextInChain; c; c = c->next)
        if (c->sType == WGPUSType_ShaderSourceWGSL) blit = str(reinterpret_cast<const WGPUShaderSourceWGSL *>(c)->code).find("fn vs(") != std::string::npos;
    return new WGPUShaderModuleImpl{blit};
}

void wgpuShaderModuleRelease(WGPUShaderModule m) { delete m; }

WGPUComputePipeline wgpuDeviceCreateComputePipeline(WGPUDevice, WGPUComputePipelineDescriptor const *d) {
    current();
    const std::string entry = str(d->compute.entryPoint);
    std::uint32_t scale = 1;
    for (std::size_t i = 0; i < d->compute.constantCount; ++i)
        if (str(d->compute.constants[i].key) == "SCALE") scale = static_cast<std::uint32_t>(d->compute.constants[i].value);
    const auto key = std::make_pair(entry, scale);
    auto it = g_programs.find(key);
    if (it == g_programs.end()) {
        const char *source = find_shader(entry, scale);
        if (!source) {
            std::fprintf(stderr, "[gl] no GLSL for %s at scale %u; run scripts/gen_glsl.py\n", entry.c_str(), scale);
            return nullptr;
        }
        const GLuint shader = compile(GL_COMPUTE_SHADER, source, entry.c_str());
        const GLuint program = shader ? link({shader}, entry.c_str()) : 0;
        if (!program) return nullptr;
        it = g_programs.emplace(key, program).first;
    }
    return new WGPUComputePipelineImpl{it->second};
}

WGPURenderPipeline wgpuDeviceCreateRenderPipeline(WGPUDevice, WGPURenderPipelineDescriptor const *) {
    current();
    const GLuint vs = compile(GL_VERTEX_SHADER, find_shader("blit_vs", 0), "blit vertex shader");
    const GLuint fs = compile(GL_FRAGMENT_SHADER, find_shader("blit_fs", 0), "blit fragment shader");
    const GLuint program = vs && fs ? link({vs, fs}, "blit") : 0;
    if (!program) return nullptr;
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);  // a core profile draws nothing without one
    return new WGPURenderPipelineImpl{program, vao};
}

WGPUCommandEncoder wgpuDeviceCreateCommandEncoder(WGPUDevice, WGPUCommandEncoderDescriptor const *) { return new WGPUCommandEncoderImpl; }

void wgpuCommandEncoderRelease(WGPUCommandEncoder e) { delete e; }

WGPUCommandBuffer wgpuCommandEncoderFinish(WGPUCommandEncoder e, WGPUCommandBufferDescriptor const *) {
    auto *cmd = new WGPUCommandBufferImpl;
    cmd->ops = std::move(e->ops);
    return cmd;
}

void wgpuCommandBufferRelease(WGPUCommandBuffer c) { delete c; }

void wgpuCommandEncoderCopyBufferToBuffer(WGPUCommandEncoder e, WGPUBuffer src, uint64_t src_offset, WGPUBuffer dst, uint64_t dst_offset, uint64_t size) {
    const GLuint s = src->id, d = dst->id;
    e->ops.push_back([s, d, src_offset, dst_offset, size] {
        glBindBuffer(GL_COPY_READ_BUFFER, s);
        glBindBuffer(GL_COPY_WRITE_BUFFER, d);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, static_cast<GLintptr>(src_offset), static_cast<GLintptr>(dst_offset),
                            static_cast<GLsizeiptr>(size));
    });
}

void wgpuCommandEncoderResolveQuerySet(WGPUCommandEncoder, WGPUQuerySet, uint32_t, uint32_t, WGPUBuffer, uint64_t) {}

WGPUComputePassEncoder wgpuCommandEncoderBeginComputePass(WGPUCommandEncoder e, WGPUComputePassDescriptor const *) {
    return new WGPUComputePassEncoderImpl{e};
}

void wgpuComputePassEncoderSetPipeline(WGPUComputePassEncoder p, WGPUComputePipeline pipeline) {
    const GLuint program = pipeline->program;
    p->encoder->ops.push_back([program] { glUseProgram(program); });
}

void wgpuComputePassEncoderSetBindGroup(WGPUComputePassEncoder p, uint32_t, WGPUBindGroup group, size_t count, uint32_t const *offsets) {
    std::vector<std::uint32_t> dynamic(offsets, offsets + count);
    p->encoder->ops.push_back([g = *group, dynamic = std::move(dynamic)] { bind(g, dynamic); });
}

void wgpuComputePassEncoderDispatchWorkgroups(WGPUComputePassEncoder p, uint32_t x, uint32_t y, uint32_t z) {
    p->encoder->ops.push_back([x, y, z] {
        glDispatchCompute(x, y, z);
        // WebGPU orders the storage accesses of successive dispatches; OpenGL needs telling.
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    });
}

void wgpuComputePassEncoderEnd(WGPUComputePassEncoder) {}

void wgpuComputePassEncoderRelease(WGPUComputePassEncoder p) { delete p; }

WGPURenderPassEncoder wgpuCommandEncoderBeginRenderPass(WGPUCommandEncoder e, WGPURenderPassDescriptor const *d) {
    const WGPUColor c = d->colorAttachmentCount ? d->colorAttachments[0].clearValue : WGPUColor{0, 0, 0, 1};
    e->ops.push_back([c] {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glClearColor(static_cast<float>(c.r), static_cast<float>(c.g), static_cast<float>(c.b), static_cast<float>(c.a));
        glClear(GL_COLOR_BUFFER_BIT);
    });
    return new WGPURenderPassEncoderImpl{e};
}

}  // extern "C"

namespace {
std::uint32_t g_surface_height = 0;
}

extern "C" {

void wgpuRenderPassEncoderSetViewport(WGPURenderPassEncoder p, float x, float y, float w, float h, float, float) {
    // WebGPU puts the origin at the top left, OpenGL at the bottom left.
    p->encoder->ops.push_back([x, y, w, h] {
        glViewport(static_cast<GLint>(x), static_cast<GLint>(static_cast<float>(g_surface_height) - y - h), static_cast<GLsizei>(w), static_cast<GLsizei>(h));
    });
}

void wgpuRenderPassEncoderSetPipeline(WGPURenderPassEncoder p, WGPURenderPipeline pipeline) {
    const GLuint program = pipeline->program, vao = pipeline->vao;
    p->encoder->ops.push_back([program, vao] {
        glUseProgram(program);
        glBindVertexArray(vao);
    });
}

void wgpuRenderPassEncoderSetBindGroup(WGPURenderPassEncoder p, uint32_t, WGPUBindGroup group, size_t count, uint32_t const *offsets) {
    std::vector<std::uint32_t> dynamic(offsets, offsets + count);
    p->encoder->ops.push_back([g = *group, dynamic = std::move(dynamic)] { bind(g, dynamic); });
}

void wgpuRenderPassEncoderDraw(WGPURenderPassEncoder p, uint32_t vertices, uint32_t, uint32_t first, uint32_t) {
    p->encoder->ops.push_back([vertices, first] { glDrawArrays(GL_TRIANGLES, static_cast<GLint>(first), static_cast<GLsizei>(vertices)); });
}

void wgpuRenderPassEncoderEnd(WGPURenderPassEncoder) {}

void wgpuRenderPassEncoderRelease(WGPURenderPassEncoder p) { delete p; }

WGPUSurface wgpuInstanceCreateSurface(WGPUInstance, WGPUSurfaceDescriptor const *d) {
    for (const WGPUChainedStruct *c = d->nextInChain; c; c = c->next)
        if (c->sType == WGPUSType_SurfaceSourceXlibWindow) {
            const auto *x = reinterpret_cast<const WGPUSurfaceSourceXlibWindow *>(c);
            if (!x->display && x->window) {
                auto *s = new WGPUSurfaceImpl;
                s->window = reinterpret_cast<void *>(static_cast<std::uintptr_t>(x->window));
                return s;
            }
        }
    std::fprintf(stderr, "[gl] only a native EGL window can be presented to\n");
    return nullptr;
}

WGPUStatus wgpuSurfaceGetCapabilities(WGPUSurface, WGPUAdapter, WGPUSurfaceCapabilities *caps) {
    static const WGPUTextureFormat formats[] = {WGPUTextureFormat_RGBA8Unorm};
    static const WGPUPresentMode modes[] = {WGPUPresentMode_Fifo};
    caps->formatCount = 1;
    caps->formats = formats;
    caps->presentModeCount = 1;
    caps->presentModes = modes;
    return WGPUStatus_Success;
}

void wgpuSurfaceCapabilitiesFreeMembers(WGPUSurfaceCapabilities) {}

void wgpuSurfaceConfigure(WGPUSurface s, WGPUSurfaceConfiguration const *config) {
    s->width = config->width;
    s->height = config->height;
    g_surface_height = config->height;
    if (s->surface) return;
    s->surface = eglCreateWindowSurface(g.display, g.config, s->window, nullptr);
    if (!s->surface) {
        std::fprintf(stderr, "[gl] cannot create a window surface (EGL error 0x%X)\n", static_cast<unsigned>(eglGetError()));
        return;
    }
    g.draw = s->surface;
    eglMakeCurrent(g.display, g.draw, g.draw, g.context);
    eglSwapInterval(g.display, 1);
}

void wgpuSurfaceGetCurrentTexture(WGPUSurface s, WGPUSurfaceTexture *out) {
    out->texture = s->surface ? &g_texture : nullptr;
    out->status = s->surface ? WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal : WGPUSurfaceGetCurrentTextureStatus_Error;
}

WGPUStatus wgpuSurfacePresent(WGPUSurface s) {
    current();
    return s->surface && eglSwapBuffers(g.display, s->surface) ? WGPUStatus_Success : WGPUStatus_Error;
}

void wgpuSurfaceRelease(WGPUSurface s) {
    if (!s) return;
    if (s->surface) {
        if (g.draw == s->surface) {
            g.draw = nullptr;
            eglMakeCurrent(g.display, nullptr, nullptr, g.context);
        }
        eglDestroySurface(g.display, s->surface);
    }
    delete s;
}

WGPUTextureView wgpuTextureCreateView(WGPUTexture, WGPUTextureViewDescriptor const *) { return &g_view; }
void wgpuTextureRelease(WGPUTexture) {}
void wgpuTextureViewRelease(WGPUTextureView) {}

}  // extern "C"
