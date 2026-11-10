#include "glapi.hpp"

#include <cstdio>

namespace ps2::gs::gpu::gl {

#define PS2_GL_DEFINE(ret, name, args) ret(*gl##name) args = nullptr;
PS2_GL_FUNCTIONS(PS2_GL_DEFINE)
#undef PS2_GL_DEFINE

bool load_functions() {
    bool ok = true;
#define PS2_GL_LOAD(ret, name, args)                                                                  \
    gl##name = reinterpret_cast<ret(*) args>(eglGetProcAddress("gl" #name));                          \
    if (!gl##name) {                                                                                  \
        std::fprintf(stderr, "[gl] missing gl%s\n", #name);                                           \
        ok = false;                                                                                   \
    }
    PS2_GL_FUNCTIONS(PS2_GL_LOAD)
#undef PS2_GL_LOAD
    return ok;
}

}
