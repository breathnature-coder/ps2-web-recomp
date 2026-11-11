# Switch 移植（GT4 优先）

目标：把 ps2-web-recomp 为 Gran Turismo 4 生成的 C++ 编译成 Switch 自制程序（.nro），用你自己的光盘镜像在自己的机器上运行。生成的代码和游戏数据都属于游戏版权方，只能留在自己的设备上，不要发布。

## 架构：哪些能直接用，哪些要换

| 部分 | 代码 | Switch 上的情况 |
|---|---|---|
| EE 重编译器 `eerecomp`、`vurecomp` | `recomp/` | 在 PC 上运行，输出可移植的 C++，不用改 |
| 运行时：内核 HLE、IOP、DMAC、VIF、VU、SPU2、IPU、软件 GS | `runtime/` | 纯 C++20，已能用 GCC 编译，也能交叉编译成 AArch64 |
| 窗口、输入、音频 | `host/window.cpp`（SDL2） | 复用 devkitPro 的 switch-sdl2，已加 `__SWITCH__` 分支 |
| GPU 渲染器 | `runtime/src/gs/gpu/`（WebGPU + WGSL，约 3000 行） | 原样复用：`gpu/gl/` 用 OpenGL 4.3 实现了它用到的 WebGPU 函数，见第 2 阶段 |
| VU 即时编译 | `vu_jit*.cpp`（生成 WebAssembly） | 只有网页版用。Switch 用预编译的 VU 程序，没见过的程序走解释器 |

## 阶段

### 第 0 阶段：可移植性（已完成，在 `switch-port` 分支）
- `vu_ops.hpp` 里两个 clang 专用内建函数加了 GCC 回退（devkitA64 用的是 GCC）。clang 和 GCC、-O0 和 -O2 的结果逐位一致，新测试 `vu_ops_bits` 会持续检查这一点。
- 生成代码里的 `[[clang::musttail]]` 改成 `PS2_MUSTTAIL` 宏。clang 用 `clang::musttail`，GCC 15+ 用 `gnu::musttail`；更旧的编译器只在生成代码里报错，不会悄悄生成可能爆栈的程序。
- 运行时全部 27 个源文件都能交叉编译成 AArch64 目标文件。用的是借来的 x86 头文件，所以这只能说明代码里没有平台相关的写法，不等于在 Switch 上编译通过。
- 新增 `host/switch.cpp`（libnx 入口：从 SD 卡读启动参数，检查是否是应用模式，可选 CPU 超频和日志）、CMake 里的 `ps2nx` 目标、`scripts/build_switch.sh`。
- 所有合成测试在 clang 和 GCC 下都通过。

### 第 1 阶段：在 Switch 上启动 GT4
目标是验证正确性：能开机，进菜单，有声音，能读写存档。第 2 阶段做完后默认就走 GPU 渲染；如果画面有问题，可以在 `args.txt` 里加一行 `@env PS2_GS=cpu` 换成软件渲染对照（会非常慢：原作者在 PC 上用软件 GS 加 VU 解释器只有大约 3 FPS）。
- 要验证：线程栈（16 MiB）、内存（必须用应用模式启动）、光盘流式读取、手柄映射、音频节奏。
- 调试：在 `args.txt` 里加 `@log`，或者用 nxlink。

### 第 2 阶段：GPU 渲染器（已完成，待真机验证）
没有重写渲染器，而是让原来的渲染器在 OpenGL 上运行：
- `runtime/src/gs/gpu/gl/webgpu_gl.cpp` 用 OpenGL 4.3（EGL）实现了 `gpu.cpp` 用到的那组 WebGPU 函数。上游的 `gpu.cpp` 一行没改，以后同步上游几乎没有冲突。
- 着色器由 wgpu 内置的 naga 预先从 WGSL 翻译成 GLSL（`scripts/gen_glsl.py` 生成 `gl/shaders.cpp`）。WGSL 改了之后重新运行它；`gl_shaders_current` 测试会在没同步时报错。
- 语义和 WebGPU 一致：命令先录制、提交时执行；缓冲区初始为零；每次 dispatch 之后加内存屏障；回读用 fence 异步完成。
- **验证结果：** 合成测试覆盖纹理上传、CLUT、各种图元、混合、深度和 Alpha 测试、纹理过滤、本地拷贝和渲染到纹理。在 1、2、3 倍内部分辨率下各跑 100 到 200 个场景，OpenGL 版的 GS 显存和显示输出与原版 WebGPU（Vulkan）**逐字节一致**，clang 和 GCC 编译的结果也一样。复查命令：`scripts/compare_gpu_backends.sh`。
- Switch 上由 OpenGL 后端直接画到主机的原生窗口（NWindow），不经过 SDL 的渲染器；GPU 初始化失败时自动退回 SDL 软件显示（也可以在 `args.txt` 里写 `@env PS2_SDL_RENDERER=1` 强制）。
- 构建：Switch 默认使用这个后端（CMake 选项 `PS2_GPU_BACKEND=gl`，链接 `EGL glapi drm_nouveau`）。只需要 WebGPU 的头文件，不需要 wgpu-native 库本身。
- **还没验证的：** 真机上 nouveau 驱动的着色器编译和速度、显示路径（这边没有窗口系统，测不了画到屏幕这一步）、HOME 菜单的挂起和恢复。

### 第 3 阶段：性能（进行中）
第一次真机测试（CPU 1963 MHz、GPU 998 MHz、内存 2133 MHz）的结果：
- 菜单 20 到 30 帧，比赛 3 到 5 帧。超频插件显示 CPU 各核都没跑满，GPU 和内存带宽负载都在 90% 以上。
- 菜单里瓶颈是 GPU 渲染：日志显示 `gs` 线程空闲 0%，模拟线程有 88% 在等。
- 比赛里主要是 VU 程序没被预编译（`vu1 interpreted` 每秒 3000 次）：PC 上的探索运行没进过比赛，比赛用到的 VU 微程序没录下来。

对应的工具：
- **录制没见过的代码：** `scripts/build_switch.sh gt4 --record --log` 打包的版本会在 SD 卡上记下没预编译的 VU 程序（`vu1/`）和没翻译到的 EE 代码地址（`misses.txt`）。玩一圈之后，`scripts/import_switch_run.sh gt4 <镜像> <卡上的 switch/ps2recomp 文件夹>` 会把它们导回来、重新翻译编译、重新打包。多录几次，覆盖不同的赛道和车辆。
- **GPU 时间分析：** `--stats` 让 `log.txt` 每秒多一行 `[gl]`，列出渲染线程花在发命令、上传、等待、回读和显示上的时间，以及 GPU 在每个着色器入口上花的时间。
- OpenGL 层现在只重新绑定发生变化的缓冲区，合成测试里省掉了约 88% 的绑定调用，输出仍与 WebGPU 逐字节一致。
- 之后：根据 `[gl]` 的数据优化最耗时的着色器。线程绑核和降低内部分辨率也在候选里，不过默认已经是 1 倍。

### 第 3b 阶段：原生光栅化渲染器（方案 B，进行中）
依据（都是真机数据，CPU 1963 MHz）：
- 现有的计算着色器渲染器：Mesa 26 的 nvc0 比 Mesa 20.1 快约 2.2 倍，比赛仍只有 5 到 8 帧，GPU 占满。zink（Vulkan）更慢。
- NetherSX2_nx（PCSX2 系，原生三角形光栅化）跑同一个游戏约 28 帧，GPU 在 768 MHz 下负载只有 22% 到 27%，一个 CPU 核跑满。说明这块 GPU 用原生光栅化画 GT4 绰绰有余，它卡在 CPU。
- 我们的模拟线程在比赛里每帧约 20 ms，所以配上够快的渲染器，上限在 50 帧左右。

设计：
- GS 显存仍是一块 GPU 缓冲区，纹理直接在片段着色器里从这块缓冲区取（沿用现有的地址换算和 CLUT 代码），所以不需要纹理缓存。
- 正在绘制的帧缓冲和深度缓冲用真正的 GPU 纹理（按 FBP、FBW、格式缓存）。有人要读对应的显存区域时（当纹理采样、本地拷贝、回读、显示），再把纹理写回显存缓冲区；显存被别的途径改写后，下次绘制前从显存重新载入。
- 混合、深度测试、Alpha 测试尽量映射到 GPU 的固定功能。映射不了的状态整批退回现有的计算着色器光栅化，所以任何时候画面都应当是对的，只是那一批慢。
- 分步扩大原生路径覆盖的状态。先做哪些由 `--census` 的统计决定：它每十秒列出游戏用到的绘制状态和各自覆盖的屏幕面积。
- 桌面上仍走 WebGPU，Switch 上走 `gl/` 那层，两边共用同一套 WGSL，继续用合成场景和计算着色器版本对照（原生光栅化不可能逐位一致，按容差比较）。

### 第 4 阶段：体验
- ZL/ZR 只有开和关，要映射成满压力的 L2/R2（GT4 用压力键控制油门和刹车）。
- 掌机和底座模式的分辨率、HOME 菜单挂起和恢复、存档备份。

## 怎么构建

1. **在电脑上生成代码**（Linux、WSL2 或 macOS；需要 clang、CMake、Ninja、Python 3）：
   ```bash
   git checkout switch-port
   scripts/port.sh gt4 "Gran Turismo 4.iso" --native-only
   ```
   先在电脑上用 `build/native-gt4/ps2run ... --window` 确认能玩，再上 Switch。原生窗口需要 SDL2 开发包。
2. **装 devkitPro**：`dkp-pacman -S switch-dev switch-sdl2 switch-mesa`。devkitA64 的 GCC 必须是 15 以上（`aarch64-none-elf-gcc --version`）。
3. **构建 Switch 版**：
   ```bash
   scripts/build_switch.sh gt4 --log
   ```
   然后把 `build/switch-gt4/sd/` 里的内容复制到 SD 卡根目录，再把光盘镜像复制成 `switch/ps2recomp/gt4.iso`。GT4 是双层 DVD，镜像超过 4 GiB，SD 卡要用 exFAT 格式。
4. **启动**：在任意游戏上按住 R 进入自制程序菜单（应用模式），再打开 ps2recomp。从相册进入的话内存不够。

## 已知风险
- 这一阶段的 Switch 代码还没有用真正的 devkitPro 编译过，这边没有 libnx 和 switch-sdl2。第一次编译时可能要修几处小问题。
- GCC 15 的 `musttail` 在 AArch64 上对生成代码的函数签名是否都能满足，要到第一次构建时才知道。不满足的话，换成 clang 加 devkitPro 的 sysroot 来编译。
- 原作者用的是 GT4 PAL 版；其他版本在电脑上跑通了再上 Switch。
