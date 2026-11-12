// Prints the GLSL that naga, inside wgpu-native, makes from a WGSL shader when
// wgpu runs on its OpenGL backend. Used by scripts/gen_glsl.py to translate the
// GS shaders for the OpenGL renderer; needs a working EGL/OpenGL (Mesa's
// llvmpipe is enough). Not part of any build.
//
//   wgsl_to_glsl <file.wgsl> <SCALE> <entry point>...   compute pipelines
//   wgsl_to_glsl <file.wgsl> 1 --blit                   the vs/fs render pipeline
//   wgsl_to_glsl <file.wgsl> <SCALE> --store            the compute pipelines that read a native target's texture
//   wgsl_to_glsl <file.wgsl> <SCALE> --native           the native renderer's render pipelines (gs.wgsl and
//                                                       native.wgsl in one file, after "enable dual_source_blending;")
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>
static WGPUStringView sv(const char *s) { return {s, WGPU_STRLEN}; }
int main(int argc, char **argv) {
    std::ifstream f(argv[1]);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string src = ss.str();
    double scale = argc > 2 ? atof(argv[2]) : 1;
    wgpuSetLogCallback([](WGPULogLevel l, WGPUStringView m, void *) { fprintf(stdout, "@@LOG %d\n%.*s\n@@END\n", (int)l, (int)m.length, m.data); },
                       nullptr);
    wgpuSetLogLevel(WGPULogLevel_Trace);
    WGPUInstanceExtras ex = {};
    ex.chain.sType = (WGPUSType)WGPUSType_InstanceExtras;
    ex.backends = WGPUInstanceBackend_GL;
    WGPUInstanceDescriptor id = WGPU_INSTANCE_DESCRIPTOR_INIT;
    id.nextInChain = &ex.chain;
    WGPUInstance inst = wgpuCreateInstance(&id);
    WGPUAdapter ad = nullptr;
    WGPURequestAdapterCallbackInfo acb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    acb.mode = WGPUCallbackMode_AllowSpontaneous;
    acb.callback = [](WGPURequestAdapterStatus s, WGPUAdapter a, WGPUStringView m, void *o, void *) {
        if (s == WGPURequestAdapterStatus_Success)
            *(WGPUAdapter *)o = a;
        else
            fprintf(stderr, "adapter: %.*s\n", (int)m.length, m.data);
    };
    acb.userdata1 = &ad;
    wgpuInstanceRequestAdapter(inst, nullptr, acb);
    if (!ad) {
        fprintf(stderr, "no adapter\n");
        return 1;
    }
    WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
    wgpuAdapterGetInfo(ad, &info);
    fprintf(stderr, "adapter %.*s backend %d\n", (int)info.device.length, info.device.data, (int)info.backendType);
    WGPULimits sup = WGPU_LIMITS_INIT;
    wgpuAdapterGetLimits(ad, &sup);
    WGPULimits lim = WGPU_LIMITS_INIT;
    lim.maxStorageBufferBindingSize = sup.maxStorageBufferBindingSize;
    lim.maxBufferSize = sup.maxBufferSize;
    lim.maxStorageBuffersPerShaderStage = sup.maxStorageBuffersPerShaderStage;
    lim.maxComputeWorkgroupStorageSize = sup.maxComputeWorkgroupStorageSize;
    lim.maxComputeInvocationsPerWorkgroup = sup.maxComputeInvocationsPerWorkgroup;
    lim.maxComputeWorkgroupSizeX = sup.maxComputeWorkgroupSizeX;
    lim.maxComputeWorkgroupSizeY = sup.maxComputeWorkgroupSizeY;
    fprintf(stderr, "storage buffers/stage %u, max binding %llu\n", sup.maxStorageBuffersPerShaderStage,
            (unsigned long long)sup.maxStorageBufferBindingSize);
    WGPUDeviceDescriptor dd = WGPU_DEVICE_DESCRIPTOR_INIT;
    dd.requiredLimits = &lim;
    const WGPUFeatureName dual = WGPUFeatureName_DualSourceBlending;
    if (wgpuAdapterHasFeature(ad, dual)) {
        dd.requiredFeatureCount = 1;
        dd.requiredFeatures = &dual;
    }
    WGPUDevice dev = nullptr;
    WGPURequestDeviceCallbackInfo dcb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    dcb.mode = WGPUCallbackMode_AllowSpontaneous;
    dcb.callback = [](WGPURequestDeviceStatus s, WGPUDevice d, WGPUStringView m, void *o, void *) {
        if (s == WGPURequestDeviceStatus_Success)
            *(WGPUDevice *)o = d;
        else
            fprintf(stderr, "device: %.*s\n", (int)m.length, m.data);
    };
    dcb.userdata1 = &dev;
    wgpuAdapterRequestDevice(ad, &dd, dcb);
    if (!dev)
        return 1;
    WGPUShaderSourceWGSL s = WGPU_SHADER_SOURCE_WGSL_INIT;
    s.code = sv(src.c_str());
    WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    md.nextInChain = &s.chain;
    WGPUShaderModule mod = wgpuDeviceCreateShaderModule(dev, &md);
    WGPUBindGroupLayoutEntry le[8] = {};
    const WGPUBufferBindingType types[8] = {WGPUBufferBindingType_Storage,         WGPUBufferBindingType_ReadOnlyStorage,
                                            WGPUBufferBindingType_Uniform,         WGPUBufferBindingType_ReadOnlyStorage,
                                            WGPUBufferBindingType_Storage,         WGPUBufferBindingType_Storage,
                                            WGPUBufferBindingType_ReadOnlyStorage, WGPUBufferBindingType_Storage};
    for (int i = 0; i < 8; i++) {
        le[i] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        le[i].binding = i;
        le[i].visibility = WGPUShaderStage_Compute;
        le[i].buffer.type = types[i];
        le[i].buffer.hasDynamicOffset = i == 2;
        le[i].buffer.minBindingSize = i == 2 ? 256 : 0;
    }
    WGPUBindGroupLayoutDescriptor ld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    ld.entryCount = 8;
    ld.entries = le;
    WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(dev, &ld);
    WGPUPipelineLayoutDescriptor pld = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pld.bindGroupLayoutCount = 1;
    pld.bindGroupLayouts = &bgl;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(dev, &pld);
    if (argc > 3 && std::string(argv[3]) == "--blit") {
        WGPUBindGroupLayoutEntry be[2] = {WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT};
        be[0].binding = 0;
        be[0].visibility = WGPUShaderStage_Fragment;
        be[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
        be[1].binding = 1;
        be[1].visibility = WGPUShaderStage_Fragment;
        be[1].buffer.type = WGPUBufferBindingType_Uniform;
        WGPUBindGroupLayoutDescriptor bld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        bld.entryCount = 2;
        bld.entries = be;
        WGPUBindGroupLayout bl = wgpuDeviceCreateBindGroupLayout(dev, &bld);
        WGPUPipelineLayoutDescriptor bpd = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        bpd.bindGroupLayoutCount = 1;
        bpd.bindGroupLayouts = &bl;
        WGPUPipelineLayout bp = wgpuDeviceCreatePipelineLayout(dev, &bpd);
        WGPUColorTargetState color = WGPU_COLOR_TARGET_STATE_INIT;
        color.format = WGPUTextureFormat_RGBA8Unorm;
        WGPUFragmentState fs = WGPU_FRAGMENT_STATE_INIT;
        fs.module = mod;
        fs.entryPoint = sv("fs");
        fs.targetCount = 1;
        fs.targets = &color;
        WGPURenderPipelineDescriptor rd = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
        rd.layout = bp;
        rd.vertex.module = mod;
        rd.vertex.entryPoint = sv("vs");
        rd.fragment = &fs;
        fprintf(stdout, "@@ENTRY blit\n");
        WGPURenderPipeline p = wgpuDeviceCreateRenderPipeline(dev, &rd);
        fprintf(stderr, "blit: %p\n", (void *)p);
        return 0;
    }
    WGPUConstantEntry scale_constant = WGPU_CONSTANT_ENTRY_INIT;
    scale_constant.key = sv("SCALE");
    scale_constant.value = scale;
    if (argc > 3 && std::string(argv[3]) == "--store") {
        const char *names[2] = {"store_color", "store_depth"};
        for (int k = 0; k < 2; k++) {
            WGPUBindGroupLayoutEntry te = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
            te.binding = 0;
            te.visibility = WGPUShaderStage_Compute;
            te.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
            te.texture.viewDimension = WGPUTextureViewDimension_2D;
            WGPUBindGroupLayoutDescriptor tld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
            tld.entryCount = 1;
            tld.entries = &te;
            const WGPUBindGroupLayout both[2] = {bgl, wgpuDeviceCreateBindGroupLayout(dev, &tld)};
            WGPUPipelineLayoutDescriptor spd = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
            spd.bindGroupLayoutCount = 2;
            spd.bindGroupLayouts = both;
            WGPUComputePipelineDescriptor cd = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
            cd.layout = wgpuDeviceCreatePipelineLayout(dev, &spd);
            cd.compute.module = mod;
            cd.compute.entryPoint = sv(names[k]);
            cd.compute.constantCount = 1;
            cd.compute.constants = &scale_constant;
            fprintf(stdout, "@@ENTRY %s\n", names[k]);
            WGPUComputePipeline p = wgpuDeviceCreateComputePipeline(dev, &cd);
            fprintf(stderr, "%s: %p\n", names[k], (void *)p);
        }
        return 0;
    }
    if (argc > 3 && std::string(argv[3]) == "--native") {
        WGPUBindGroupLayoutEntry ne[7] = {};
        const int bindings[7] = {0, 1, 2, 3, 4, 5, 7};
        for (int i = 0; i < 7; i++) {
            ne[i] = le[bindings[i]];
            ne[i].visibility = bindings[i] == 2 || bindings[i] == 3 ? WGPUShaderStage_Vertex | WGPUShaderStage_Fragment : WGPUShaderStage_Fragment;
        }
        WGPUBindGroupLayoutDescriptor nld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        nld.entryCount = 7;
        nld.entries = ne;
        WGPUBindGroupLayout nl = wgpuDeviceCreateBindGroupLayout(dev, &nld);
        WGPUPipelineLayoutDescriptor npd = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        npd.bindGroupLayoutCount = 1;
        npd.bindGroupLayouts = &nl;
        WGPUPipelineLayout np = wgpuDeviceCreatePipelineLayout(dev, &npd);
        // The pipelines that read the pixel already there take the target's backdrop as a second group.
        WGPUBindGroupLayoutEntry be = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        be.binding = 0;
        be.visibility = WGPUShaderStage_Fragment;
        be.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
        be.texture.viewDimension = WGPUTextureViewDimension_2D;
        WGPUBindGroupLayoutDescriptor bld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        bld.entryCount = 1;
        bld.entries = &be;
        const WGPUBindGroupLayout with_backdrop[2] = {nl, wgpuDeviceCreateBindGroupLayout(dev, &bld)};
        WGPUPipelineLayoutDescriptor fpd = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        fpd.bindGroupLayoutCount = 2;
        fpd.bindGroupLayouts = with_backdrop;
        WGPUPipelineLayout fp = wgpuDeviceCreatePipelineLayout(dev, &fpd);
        // vertex entry, fragment entry, a colour target, a depth target, the backdrop
        const struct { const char *vs, *fs; bool color, depth, backdrop; } pipelines[6] = {
            {"native_vs", "native_fs", true, false, false},          {"native_vs", "native_depth_fs", true, true, false},
            {"native_vs", "native_feedback_fs", true, false, true},  {"native_vs", "native_feedback_depth_fs", true, true, true},
            {"load_vs", "load_color_fs", true, false, false},        {"load_vs", "load_depth_fs", false, true, false}};
        for (const auto &d : pipelines) {
            WGPUColorTargetState color = WGPU_COLOR_TARGET_STATE_INIT;
            color.format = WGPUTextureFormat_RGBA8Unorm;
            WGPUFragmentState fs = WGPU_FRAGMENT_STATE_INIT;
            fs.module = mod;
            fs.entryPoint = sv(d.fs);
            fs.constantCount = 1;
            fs.constants = &scale_constant;
            fs.targetCount = d.color ? 1 : 0;
            fs.targets = d.color ? &color : nullptr;
            WGPUDepthStencilState depth = WGPU_DEPTH_STENCIL_STATE_INIT;
            depth.format = WGPUTextureFormat_Depth32Float;
            depth.depthWriteEnabled = WGPUOptionalBool_True;
            depth.depthCompare = WGPUCompareFunction_Always;
            WGPURenderPipelineDescriptor rd = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
            rd.layout = d.backdrop ? fp : np;
            rd.vertex.module = mod;
            rd.vertex.entryPoint = sv(d.vs);
            rd.vertex.constantCount = 1;
            rd.vertex.constants = &scale_constant;
            rd.fragment = &fs;
            if (d.depth) rd.depthStencil = &depth;
            fprintf(stdout, "@@ENTRY %s\n", d.fs);
            WGPURenderPipeline p = wgpuDeviceCreateRenderPipeline(dev, &rd);
            fprintf(stderr, "%s: %p\n", d.fs, (void *)p);
        }
        return 0;
    }
    for (int i = 3; i < argc; i++) {
        WGPUComputePipelineDescriptor cd = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        cd.layout = pl;
        cd.compute.module = mod;
        cd.compute.entryPoint = sv(argv[i]);
        WGPUConstantEntry c[2] = {WGPU_CONSTANT_ENTRY_INIT, WGPU_CONSTANT_ENTRY_INIT};
        c[0].key = sv("SCALE");
        c[0].value = scale;
        c[1].key = sv("TILE");
        c[1].value = getenv("WGSL_TILE") ? atof(getenv("WGSL_TILE")) : 16;
        cd.compute.constantCount = 2;
        cd.compute.constants = c;
        fprintf(stdout, "@@ENTRY %s\n", argv[i]);
        WGPUComputePipeline p = wgpuDeviceCreateComputePipeline(dev, &cd);
        fprintf(stderr, "%s: %p\n", argv[i], (void *)p);
    }
}
