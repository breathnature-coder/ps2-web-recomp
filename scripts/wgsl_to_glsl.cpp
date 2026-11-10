// Prints the GLSL that naga, inside wgpu-native, makes from a WGSL shader when
// wgpu runs on its OpenGL backend. Used by scripts/gen_glsl.py to translate the
// GS shaders for the OpenGL renderer; needs a working EGL/OpenGL (Mesa's
// llvmpipe is enough). Not part of any build.
//
//   wgsl_to_glsl <file.wgsl> <SCALE> <entry point>...   compute pipelines
//   wgsl_to_glsl <file.wgsl> 1 --blit                   the vs/fs render pipeline
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
    for (int i = 3; i < argc; i++) {
        WGPUComputePipelineDescriptor cd = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        cd.layout = pl;
        cd.compute.module = mod;
        cd.compute.entryPoint = sv(argv[i]);
        WGPUConstantEntry c = WGPU_CONSTANT_ENTRY_INIT;
        c.key = sv("SCALE");
        c.value = scale;
        cd.compute.constantCount = 1;
        cd.compute.constants = &c;
        fprintf(stdout, "@@ENTRY %s\n", argv[i]);
        WGPUComputePipeline p = wgpuDeviceCreateComputePipeline(dev, &cd);
        fprintf(stderr, "%s: %p\n", argv[i], (void *)p);
    }
}
