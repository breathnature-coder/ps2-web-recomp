cmake_minimum_required(VERSION 3.20)
project(ps2-web-recomp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PS2_GENERATED "" CACHE PATH "Folder with the C++ generated for a game")

if(NINTENDO_SWITCH)
    # Set by devkitPro's Switch toolchain file. The generated code needs
    # guaranteed tail calls ([[gnu::musttail]]), which GCC has from version 15.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15)
        message(FATAL_ERROR "devkitA64 with GCC 15 or newer is needed (found ${CMAKE_CXX_COMPILER_VERSION}); update with: dkp-pacman -Syu")
    endif()
endif()

if(EMSCRIPTEN)
    add_compile_options(-msimd128 -mtail-call -fwasm-exceptions -sWASM_LEGACY_EXCEPTIONS=0 -pthread)
    add_link_options(-msimd128 -mtail-call -fwasm-exceptions -sWASM_LEGACY_EXCEPTIONS=0 -pthread)
endif()

if(NOT EMSCRIPTEN AND NOT NINTENDO_SWITCH)
    add_executable(eerecomp
        recomp/main.cpp
        recomp/r5900.cpp
        recomp/elf.cpp
        recomp/analysis.cpp
        recomp/codegen.cpp
        recomp/snr2.cpp)
    target_compile_options(eerecomp PRIVATE -Wall -Wextra)

    add_executable(vujit_dump recomp/vujit_dump.cpp runtime/src/vu_jit.cpp)
    target_include_directories(vujit_dump PRIVATE runtime/include runtime/src)
    add_executable(vurecomp recomp/vurecomp.cpp)
    target_include_directories(vurecomp PRIVATE runtime/include)
    target_compile_options(vurecomp PRIVATE -Wall -Wextra)
endif()

add_library(ps2runtime STATIC
    runtime/src/memory.cpp
    runtime/src/dispatch.cpp
    runtime/src/system.cpp
    runtime/src/kernel.cpp
    runtime/src/hw.cpp
    runtime/src/cop.cpp
    runtime/src/disc.cpp
    runtime/src/dmac.cpp
    runtime/src/vif.cpp
    runtime/src/vu_micro.cpp
    runtime/src/vu_jit.cpp
    runtime/src/vu_jit_web.cpp
    runtime/src/ipu/ipu.cpp
    runtime/src/gs/gs.cpp
    runtime/src/gs/gif.cpp
    runtime/src/gs/queue.cpp
    runtime/src/gs/raster.cpp
    runtime/src/iop/cpu.cpp
    runtime/src/iop/kernel.cpp
    runtime/src/iop/libs.cpp
    runtime/src/iop/hw.cpp
    runtime/src/iop/spu2.cpp
    runtime/src/iop/sio2.cpp
    runtime/src/iop/io.cpp
    runtime/src/iop/cdvd.cpp
    runtime/src/iop/sif.cpp
    runtime/src/iop/cdvdfsv.cpp)
target_include_directories(ps2runtime PUBLIC runtime/include)
target_compile_options(ps2runtime PRIVATE -Wall -Wextra)

set(PS2_WGPU_DIR "${CMAKE_SOURCE_DIR}/third_party/wgpu-native" CACHE PATH "wgpu-native release folder")
# The GS GPU renderer: WebGPU (browser, or wgpu-native on the desktop), or the
# same renderer on OpenGL 4.3 through runtime/src/gs/gpu/gl (the Switch).
set(PS2_GPU_BACKEND "auto" CACHE STRING "GS GPU renderer: auto, webgpu, gl or none")
set(PS2_GPU_RESOLVED "${PS2_GPU_BACKEND}")
if(PS2_GPU_RESOLVED STREQUAL "auto")
    if(NINTENDO_SWITCH)
        set(PS2_GPU_RESOLVED gl)
    elseif(EMSCRIPTEN OR EXISTS "${PS2_WGPU_DIR}/lib/libwgpu_native.a")
        set(PS2_GPU_RESOLVED webgpu)
    else()
        set(PS2_GPU_RESOLVED none)
    endif()
endif()
if(NOT PS2_GPU_RESOLVED STREQUAL "none")
    set(PS2_GS_WGSL "${CMAKE_BINARY_DIR}/gs_wgsl.cpp")
    add_custom_command(OUTPUT "${PS2_GS_WGSL}"
        COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_SOURCE_DIR}/runtime/src/gs/gpu/gs.wgsl -DOUT=${PS2_GS_WGSL} -DNAME=kGsShader -P ${CMAKE_SOURCE_DIR}/cmake/embed.cmake
        DEPENDS runtime/src/gs/gpu/gs.wgsl cmake/embed.cmake)
    set(PS2_BLIT_WGSL "${CMAKE_BINARY_DIR}/blit_wgsl.cpp")
    add_custom_command(OUTPUT "${PS2_BLIT_WGSL}"
        COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_SOURCE_DIR}/runtime/src/gs/gpu/blit.wgsl -DOUT=${PS2_BLIT_WGSL} -DNAME=kBlitShader -P ${CMAKE_SOURCE_DIR}/cmake/embed.cmake
        DEPENDS runtime/src/gs/gpu/blit.wgsl cmake/embed.cmake)
    set(PS2_NATIVE_WGSL "${CMAKE_BINARY_DIR}/native_wgsl.cpp")
    add_custom_command(OUTPUT "${PS2_NATIVE_WGSL}"
        COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_SOURCE_DIR}/runtime/src/gs/gpu/native.wgsl -DOUT=${PS2_NATIVE_WGSL} -DNAME=kNativeShader -P ${CMAKE_SOURCE_DIR}/cmake/embed.cmake
        DEPENDS runtime/src/gs/gpu/native.wgsl cmake/embed.cmake)
    target_sources(ps2runtime PRIVATE runtime/src/gs/gpu/gpu.cpp "${PS2_GS_WGSL}" "${PS2_BLIT_WGSL}" "${PS2_NATIVE_WGSL}")
    target_compile_definitions(ps2runtime PUBLIC PS2_GPU)
    if(PS2_GPU_RESOLVED STREQUAL "gl")
        # Only the WebGPU headers are used; wgpu-native itself is not linked.
        if(NOT EXISTS "${PS2_WGPU_DIR}/include/webgpu/webgpu.h")
            message(FATAL_ERROR "The OpenGL renderer needs the WebGPU headers in ${PS2_WGPU_DIR}/include: run scripts/fetch_wgpu.sh")
        endif()
        target_sources(ps2runtime PRIVATE runtime/src/gs/gpu/gl/webgpu_gl.cpp runtime/src/gs/gpu/gl/glapi.cpp runtime/src/gs/gpu/gl/shaders.cpp)
        target_include_directories(ps2runtime PRIVATE "${PS2_WGPU_DIR}/include")
        target_compile_definitions(ps2runtime PUBLIC PS2_GPU_GL)
        if(NINTENDO_SWITCH)
            # devkitPro's switch-mesa (Mesa 20.1) by default. PS2_SWITCH_GL=mesa-switch
            # links a newer Mesa port instead (danfromtico/mesa-switch, Mesa 26: nvc0,
            # or Zink on NVK Vulkan), through the CMake package it installs. Its
            # pkg-config files are not used: egl.pc requires a libelf.pc that the
            # switch-dev image does not carry.
            set(PS2_SWITCH_GL "legacy" CACHE STRING "Switch OpenGL libraries: legacy or mesa-switch")
            if(PS2_SWITCH_GL STREQUAL "legacy")
                target_link_libraries(ps2runtime PUBLIC EGL glapi drm_nouveau)
            else()
                # The archives by name, linked as one group on the executable since
                # they refer to each other (and SDL2 refers to EGL and GLES).
                set(PS2_MESA_ARCHIVES EGL GLESv2 GL glapi vulkan vulkan_util nvk nak nak_rs nil libnil_format_table nir vtn compiler compiler_c_helpers
                    nouveau_mme nouveau_ws nouveau_horizon nvidia_headers_c xmlconfig mesa_util mesa_util_c11 mesa_util_simd blake3 expat zstd z)
                set(PS2_MESA_LINK "")
                foreach(lib IN LISTS PS2_MESA_ARCHIVES)
                    find_library(PS2_MESA_${lib} NAMES ${lib})
                    if(PS2_MESA_${lib})
                        list(APPEND PS2_MESA_LINK "${PS2_MESA_${lib}}")
                    endif()
                endforeach()
                message(STATUS "Switch OpenGL: mesa-switch archives: ${PS2_MESA_LINK}")
            endif()
        else()
            find_library(PS2_EGL_LIBRARY NAMES EGL libEGL.so.1 REQUIRED)
            target_link_libraries(ps2runtime PUBLIC "${PS2_EGL_LIBRARY}" pthread)
        endif()
    elseif(EMSCRIPTEN)
        target_compile_options(ps2runtime PUBLIC --use-port=emdawnwebgpu)
        target_link_options(ps2runtime PUBLIC --use-port=emdawnwebgpu)
    else()
        target_include_directories(ps2runtime PRIVATE "${PS2_WGPU_DIR}/include")
        target_link_libraries(ps2runtime PUBLIC "${PS2_WGPU_DIR}/lib/libwgpu_native.a" dl pthread m)
        if(APPLE)
            target_link_libraries(ps2runtime PUBLIC "-framework Metal" "-framework QuartzCore" "-framework Foundation" "-framework CoreFoundation" "-framework IOKit" "-framework IOSurface" objc)
        endif()
    endif()
    message(STATUS "GS GPU renderer: ${PS2_GPU_RESOLVED}")
endif()

option(PS2_BUILD_TESTS "Build native synthetic runtime tests (no game data required)" OFF)
if(PS2_BUILD_TESTS AND NOT EMSCRIPTEN AND NOT NINTENDO_SWITCH)
    enable_testing()
    add_executable(decoded_arena_test tests/decoded_arena_test.cpp)
    target_include_directories(decoded_arena_test PRIVATE runtime/src)
    add_test(NAME decoded_arena COMMAND decoded_arena_test)
    foreach(name IN ITEMS sif_scratchpad kernel_exit_delete_thread gt3_initialization gt3_state)
        add_executable(${name}_test tests/${name}_test.cpp)
        target_include_directories(${name}_test PRIVATE runtime/src)
        target_link_libraries(${name}_test PRIVATE ps2runtime)
        add_test(NAME ${name} COMMAND ${name}_test)
    endforeach()
    # The VU helpers must give bit-identical results on every compiler and CPU
    # (GT4's replays depend on it); this hash comes from clang and GCC on x86-64.
    if(TARGET ps2runtime AND PS2_GS_WGSL)
        # Software rasterizer against the GPU renderer, on synthetic scenes.
        add_executable(gs_gpu_compare_test tests/gs_gpu_compare_test.cpp)
        target_include_directories(gs_gpu_compare_test PRIVATE runtime/src)
        target_link_libraries(gs_gpu_compare_test PRIVATE ps2runtime)
        # The GPU renderer is not bit-exact with the software one (a few hundred
        # bytes per scene), so this only checks that it runs; compare two GPU
        # renderers with each other through scripts/compare_gpu_backends.sh.
        add_test(NAME gs_gpu_compare COMMAND gs_gpu_compare_test 40)
        # GT4's colour correction as it draws it, against the table lookup it is.
        add_executable(gs_gt4_lut_test tests/gs_gt4_lut_test.cpp)
        target_include_directories(gs_gt4_lut_test PRIVATE runtime/src)
        target_link_libraries(gs_gt4_lut_test PRIVATE ps2runtime)
        add_test(NAME gs_gt4_lut COMMAND gs_gt4_lut_test)
        set_tests_properties(gs_gt4_lut PROPERTIES SKIP_RETURN_CODE 77)
        set_tests_properties(gs_gpu_compare PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "PS2_COMPARE_MIN_PERCENT=0")
    endif()
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND AND EXISTS "${PS2_WGPU_DIR}/lib/libwgpu_native.a")
        # The OpenGL renderer's GLSL must match gs.wgsl and blit.wgsl.
        add_test(NAME gl_shaders_current COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/gen_glsl.py" --check)
    endif()
    add_executable(vu_ops_diff tests/vu_ops_diff.cpp)
    target_include_directories(vu_ops_diff PRIVATE runtime/include)
    add_test(NAME vu_ops_bits COMMAND vu_ops_diff)
    set_tests_properties(vu_ops_bits PROPERTIES PASS_REGULAR_EXPRESSION "^faeca02bf648188b")
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME gt3_core COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/gt3_core_test.py")
    endif()
endif()

if(PS2_GENERATED)
    file(GLOB PS2_UNITS CONFIGURE_DEPENDS "${PS2_GENERATED}/*.cpp")
    add_library(ps2game STATIC ${PS2_UNITS})
    target_include_directories(ps2game PRIVATE "${PS2_GENERATED}")
    target_link_libraries(ps2game PUBLIC ps2runtime)
    target_compile_options(ps2game PRIVATE -Wno-unused-label -Wno-unused-variable)

    if(EMSCRIPTEN)
        set(PS2_WEB_ELF "" CACHE FILEPATH "Game executable preloaded as /game/core.elf")
        set(PS2_WEB_BOOT_FILE "" CACHE FILEPATH "Text file holding the argv[0] the game sees, preloaded as /game/boot.txt")
        set(PS2_WEB_PATCHES "" CACHE FILEPATH "Patch list the runtime writes into memory, preloaded as /game/patches.txt")
        add_executable(ps2web host/web.cpp)
        if(PS2_WEB_BOOT_FILE)
            target_link_options(ps2web PRIVATE "SHELL:--preload-file \"${PS2_WEB_BOOT_FILE}@/game/boot.txt\"")
        endif()
        if(PS2_WEB_PATCHES)
            target_link_options(ps2web PRIVATE "SHELL:--preload-file \"${PS2_WEB_PATCHES}@/game/patches.txt\"")
        endif()
        target_link_libraries(ps2web PRIVATE ps2game ps2runtime)
        target_link_options(ps2web PRIVATE --profiling-funcs -sJSPI -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=268435456 -sPTHREAD_POOL_SIZE=2 -sALLOW_TABLE_GROWTH=1 -lidbfs.js -sSTACK_SIZE=16777216 -sENVIRONMENT=web
            -sEXPORTED_FUNCTIONS=_main,_ps2web_pad,_ps2web_pad_connected,_ps2web_resize,_ps2web_gpu_ms,_ps2web_frames,_ps2web_primitives,_ps2web_audio_ring,_ps2web_audio_queued_ms,_ps2web_idle_ms,_ps2web_vu_interpreted,_ps2web_quick_state,_ps2web_video_options,_ps2web_stream
            -sEXPORTED_RUNTIME_METHODS=callMain,addRunDependency,removeRunDependency,HEAPU8,ENV
            --shell-file ${CMAKE_SOURCE_DIR}/web/shell.html --preload-file ${PS2_WEB_ELF}@/game/core.elf)
        set_target_properties(ps2web PROPERTIES SUFFIX ".html" LINK_DEPENDS ${CMAKE_SOURCE_DIR}/web/shell.html)
        return()
    endif()

    if(NINTENDO_SWITCH)
        # Homebrew .nro: the desktop runner's main and SDL window, started by
        # host/switch.cpp with arguments read from the SD card.
        find_package(PkgConfig REQUIRED)
        add_executable(ps2nx host/switch.cpp host/main.cpp host/window.cpp)
        target_compile_definitions(ps2nx PRIVATE PS2_WINDOW PS2_MAIN_NAME=ps2_main)
        if(PS2_SWITCH_GL STREQUAL "legacy")
            pkg_check_modules(SDL2 REQUIRED IMPORTED_TARGET sdl2)
            target_link_libraries(ps2nx PRIVATE ps2game ps2runtime PkgConfig::SDL2)
        else()
            # sdl2.pc still names devkitPro's Mesa 20.1 libraries (drm_nouveau is
            # gone in mesa-switch): take only SDL2 itself from it, and put it in
            # one link group with the mesa-switch archives.
            pkg_check_modules(SDL2 REQUIRED sdl2)
            find_library(PS2_SDL2_ARCHIVE NAMES SDL2 REQUIRED)
            target_include_directories(ps2nx PRIVATE ${SDL2_INCLUDE_DIRS})
            target_compile_options(ps2nx PRIVATE ${SDL2_CFLAGS_OTHER})
            target_link_libraries(ps2nx PRIVATE ps2game ps2runtime -Wl,--start-group "${PS2_SDL2_ARCHIVE}" ${PS2_MESA_LINK} -Wl,--end-group nx stdc++ pthread m)
        endif()
        set(PS2_NX_TITLE "PS2 Recomp" CACHE STRING "Title shown in the homebrew menu")
        nx_generate_nacp(ps2nx.nacp NAME "${PS2_NX_TITLE}" AUTHOR "ps2-web-recomp" VERSION "0.1")
        nx_create_nro(ps2nx NACP ps2nx.nacp)
        return()
    endif()

    add_executable(ps2run host/main.cpp)
    target_link_libraries(ps2run PRIVATE ps2game ps2runtime)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_sources(ps2run PRIVATE host/profile.cpp)
        target_compile_definitions(ps2run PRIVATE PS2_PROFILE)
    endif()
    find_package(SDL2 QUIET)
    if(SDL2_FOUND AND NOT EMSCRIPTEN)
        target_sources(ps2run PRIVATE host/window.cpp)
        target_compile_definitions(ps2run PRIVATE PS2_WINDOW)
        target_link_libraries(ps2run PRIVATE SDL2::SDL2)
    endif()
endif()
