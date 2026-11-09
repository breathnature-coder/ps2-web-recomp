# Switch 移植（GT4 优先）

目标：把 ps2-web-recomp 为 Gran Turismo 4 生成的 C++ 编译成 Switch 自制程序（.nro），用你自己的光盘镜像在自己的机器上运行。生成的代码和游戏数据都属于游戏版权方，只能留在自己的设备上，不要发布。

## 架构：哪些能直接用，哪些要换

| 部分 | 代码 | Switch 上的情况 |
|---|---|---|
| EE 重编译器 `eerecomp`、`vurecomp` | `recomp/` | 在 PC 上运行，输出可移植的 C++，不用改 |
| 运行时：内核 HLE、IOP、DMAC、VIF、VU、SPU2、IPU、软件 GS | `runtime/` | 纯 C++20，已能用 GCC 编译，也能交叉编译成 AArch64 |
| 窗口、输入、音频 | `host/window.cpp`（SDL2） | 复用 devkitPro 的 switch-sdl2，已加 `__SWITCH__` 分支 |
| GPU 渲染器 | `runtime/src/gs/gpu/`（WebGPU + WGSL，约 3000 行） | **要重写**，见第 2 阶段 |
| VU 即时编译 | `vu_jit*.cpp`（生成 WebAssembly） | 只有网页版用。Switch 用预编译的 VU 程序，没见过的程序走解释器 |

## 阶段

### 第 0 阶段：可移植性（已完成，在 `switch-port` 分支）
- `vu_ops.hpp` 里两个 clang 专用内建函数加了 GCC 回退（devkitA64 用的是 GCC）。clang 和 GCC、-O0 和 -O2 的结果逐位一致，新测试 `vu_ops_bits` 会持续检查这一点。
- 生成代码里的 `[[clang::musttail]]` 改成 `PS2_MUSTTAIL` 宏。clang 用 `clang::musttail`，GCC 15+ 用 `gnu::musttail`；更旧的编译器只在生成代码里报错，不会悄悄生成可能爆栈的程序。
- 运行时全部 27 个源文件都能交叉编译成 AArch64 目标文件。用的是借来的 x86 头文件，所以这只能说明代码里没有平台相关的写法，不等于在 Switch 上编译通过。
- 新增 `host/switch.cpp`（libnx 入口：从 SD 卡读启动参数，检查是否是应用模式，可选 CPU 超频和日志）、CMake 里的 `ps2nx` 目标、`scripts/build_switch.sh`。
- 所有合成测试在 clang 和 GCC 下都通过。

### 第 1 阶段：在 Switch 上用软件渲染启动 GT4
目标是验证正确性，不追求速度：能开机，进菜单，有声音，能读写存档。
- 预期很慢。原作者在 PC 上用软件 GS 加 VU 解释器只有大约 3 FPS；这里 VU 是预编译的，但 Switch 的 CPU 弱很多。
- 要验证：线程栈（16 MiB）、内存（必须用应用模式启动）、光盘流式读取、手柄映射、音频节奏。
- 调试：在 `args.txt` 里加 `@log`，或者用 nxlink。

### 第 2 阶段：GPU 渲染器（决定能不能玩）
实现 `runtime/src/gs/gpu/gpu.hpp` 里大约 20 个接口函数，内部改用 Switch 能用的图形接口。
- **推荐先做 OpenGL 4.3 计算着色器**（devkitPro 的 switch-mesa，通过 EGL）。现有渲染器本来就是"计算着色器光栅化，GS 显存放在 GPU 上"的设计，和 GL 计算着色器一一对应。
- 着色器可以先试 naga（wgpu 的着色器编译器），把 `gs.wgsl` 自动转成 GLSL，再手工修补，省得从头翻译 1150 行。
- 以后要压榨性能再考虑 deko3d（Switch 的原生底层 API）。
- 原生运行器的软件渲染器可以当参照，用来逐帧比对 GPU 输出。

### 第 3 阶段：性能
- 线程绑核：EE、VU1、GS 线程各占一个 Switch 应用核（0 到 2 号）。
- 用 1 倍内部分辨率，打开 `@boost`（1785 MHz），视情况调节 GPU 频率。
- 探索阶段要尽量覆盖赛道和车辆，让更多 VU 程序被预编译，少走解释器。
- 先实测再定目标；在跑出第 1 和第 2 阶段之前，帧率估计没有意义。

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
2. **装 devkitPro**：`dkp-pacman -S switch-dev switch-sdl2`。devkitA64 的 GCC 必须是 15 以上（`aarch64-none-elf-gcc --version`）。
3. **构建 Switch 版**：
   ```bash
   scripts/build_switch.sh gt4 --boost --log
   ```
   然后把 `build/switch-gt4/sd/` 里的内容复制到 SD 卡根目录，再把光盘镜像复制成 `switch/ps2recomp/gt4.iso`。GT4 是双层 DVD，镜像超过 4 GiB，SD 卡要用 exFAT 格式。
4. **启动**：在任意游戏上按住 R 进入自制程序菜单（应用模式），再打开 ps2recomp。从相册进入的话内存不够。

## 已知风险
- 这一阶段的 Switch 代码还没有用真正的 devkitPro 编译过，这边没有 libnx 和 switch-sdl2。第一次编译时可能要修几处小问题。
- GCC 15 的 `musttail` 在 AArch64 上对生成代码的函数签名是否都能满足，要到第一次构建时才知道。不满足的话，换成 clang 加 devkitPro 的 sysroot 来编译。
- 原作者用的是 GT4 PAL 版；其他版本在电脑上跑通了再上 Switch。
