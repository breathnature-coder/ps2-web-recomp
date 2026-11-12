#pragma once

// The small part of EGL and OpenGL 4.3 core the GL renderer uses, declared
// here so it builds without system GL headers: on the desktop against Mesa's
// libEGL, on the Switch against devkitPro's switch-mesa. GL entry points are
// loaded through eglGetProcAddress.

#include <cstddef>
#include <cstdint>

extern "C" {
using EGLDisplay = void *;
using EGLConfig = void *;
using EGLContext = void *;
using EGLSurface = void *;
using EGLNativeWindowType = void *;
using EGLint = std::int32_t;
using EGLBoolean = unsigned int;
using EGLenum = unsigned int;

EGLDisplay eglGetDisplay(void *display_id);
EGLBoolean eglInitialize(EGLDisplay dpy, EGLint *major, EGLint *minor);
EGLBoolean eglBindAPI(EGLenum api);
EGLBoolean eglChooseConfig(EGLDisplay dpy, const EGLint *attrib_list, EGLConfig *configs, EGLint config_size, EGLint *num_config);
EGLContext eglCreateContext(EGLDisplay dpy, EGLConfig config, EGLContext share_context, const EGLint *attrib_list);
EGLSurface eglCreatePbufferSurface(EGLDisplay dpy, EGLConfig config, const EGLint *attrib_list);
EGLSurface eglCreateWindowSurface(EGLDisplay dpy, EGLConfig config, EGLNativeWindowType win, const EGLint *attrib_list);
EGLBoolean eglDestroySurface(EGLDisplay dpy, EGLSurface surface);
EGLBoolean eglMakeCurrent(EGLDisplay dpy, EGLSurface draw, EGLSurface read, EGLContext ctx);
EGLContext eglGetCurrentContext();
EGLBoolean eglSwapBuffers(EGLDisplay dpy, EGLSurface surface);
EGLBoolean eglSwapInterval(EGLDisplay dpy, EGLint interval);
EGLint eglGetError();
const char *eglQueryString(EGLDisplay dpy, EGLint name);
void (*eglGetProcAddress(const char *procname))();
}

namespace ps2::gs::gpu::gl {

inline constexpr EGLint EGL_NONE = 0x3038, EGL_RENDERABLE_TYPE = 0x3040, EGL_SURFACE_TYPE = 0x3033, EGL_PBUFFER_BIT = 0x0001, EGL_WINDOW_BIT = 0x0004,
                        EGL_OPENGL_BIT = 0x0008, EGL_RED_SIZE = 0x3024, EGL_GREEN_SIZE = 0x3023, EGL_BLUE_SIZE = 0x3022, EGL_WIDTH = 0x3057,
                        EGL_HEIGHT = 0x3056, EGL_CONTEXT_MAJOR_VERSION = 0x3098, EGL_CONTEXT_MINOR_VERSION = 0x30FB,
                        EGL_CONTEXT_OPENGL_PROFILE_MASK = 0x30FD, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT = 0x0001, EGL_EXTENSIONS = 0x3055;
inline constexpr EGLenum EGL_OPENGL_API = 0x30A2, EGL_PLATFORM_SURFACELESS_MESA = 0x31DD;

using GLenum = unsigned int;
using GLuint = unsigned int;
using GLint = int;
using GLsizei = int;
using GLboolean = unsigned char;
using GLbitfield = unsigned int;
using GLchar = char;
using GLfloat = float;
using GLintptr = std::intptr_t;
using GLsizeiptr = std::intptr_t;
using GLuint64 = std::uint64_t;
using GLsync = struct __GLsync *;

inline constexpr GLenum GL_NO_ERROR = 0, GL_VENDOR = 0x1F00, GL_RENDERER = 0x1F01, GL_VERSION = 0x1F02, GL_SHADER_STORAGE_BUFFER = 0x90D2,
                        GL_UNIFORM_BUFFER = 0x8A11, GL_COPY_READ_BUFFER = 0x8F36, GL_COPY_WRITE_BUFFER = 0x8F37, GL_STATIC_DRAW = 0x88E4,
                        GL_DYNAMIC_DRAW = 0x88E8, GL_STREAM_READ = 0x88E1, GL_COMPUTE_SHADER = 0x91B9, GL_VERTEX_SHADER = 0x8B31,
                        GL_FRAGMENT_SHADER = 0x8B30, GL_COMPILE_STATUS = 0x8B81, GL_LINK_STATUS = 0x8B82, GL_INFO_LOG_LENGTH = 0x8B84,
                        GL_SYNC_GPU_COMMANDS_COMPLETE = 0x9117, GL_ALREADY_SIGNALED = 0x911A, GL_TIMEOUT_EXPIRED = 0x911B,
                        GL_CONDITION_SATISFIED = 0x911C, GL_WAIT_FAILED = 0x911D, GL_TRIANGLES = 0x0004, GL_FRAMEBUFFER = 0x8D40,
                        GL_COLOR_BUFFER_BIT = 0x4000, GL_DEPTH_TEST = 0x0B71, GL_BLEND = 0x0BE2, GL_SCISSOR_TEST = 0x0C11,
                        GL_CULL_FACE = 0x0B44, GL_MAX_SHADER_STORAGE_BLOCK_SIZE = 0x90DE, GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT = 0x8A34,
                        GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT = 0x90DF, GL_TIME_ELAPSED = 0x88BF, GL_QUERY_RESULT = 0x8866,
                        GL_QUERY_RESULT_AVAILABLE = 0x8867, GL_TEXTURE_2D = 0x0DE1, GL_RGBA8 = 0x8058, GL_DEPTH_COMPONENT32F = 0x8CAC,
                        GL_TEXTURE0 = 0x84C0, GL_COLOR_ATTACHMENT0 = 0x8CE0, GL_DEPTH_ATTACHMENT = 0x8D00, GL_FRAMEBUFFER_COMPLETE = 0x8CD5,
                        GL_NONE = 0, GL_TEXTURE_MAG_FILTER = 0x2800, GL_TEXTURE_MIN_FILTER = 0x2801, GL_NEAREST = 0x2600, GL_ZERO = 0, GL_ONE = 1,
                        GL_CONSTANT_COLOR = 0x8001, GL_ONE_MINUS_CONSTANT_COLOR = 0x8002, GL_SRC1_ALPHA = 0x8589,
                        GL_ONE_MINUS_SRC1_ALPHA = 0x88FB, GL_FUNC_ADD = 0x8006, GL_FUNC_SUBTRACT = 0x800A, GL_FUNC_REVERSE_SUBTRACT = 0x800B,
                        GL_NEVER = 0x0200, GL_GREATER = 0x0204, GL_GEQUAL = 0x0206, GL_ALWAYS = 0x0207, GL_LOWER_LEFT = 0x8CA1,
                        GL_ZERO_TO_ONE = 0x935F, GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS = 0x90D6, GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS = 0x90DA;
inline constexpr GLbitfield GL_SHADER_STORAGE_BARRIER_BIT = 0x00002000, GL_BUFFER_UPDATE_BARRIER_BIT = 0x00000200,
                            GL_UNIFORM_BARRIER_BIT = 0x00000004, GL_SYNC_FLUSH_COMMANDS_BIT = 0x00000001;
inline constexpr GLuint64 GL_TIMEOUT_IGNORED = ~0ull;

#define PS2_GL_FUNCTIONS(X)                                                                                                                  \
    X(const unsigned char *, GetString, (GLenum name))                                                                                       \
    X(void, GetIntegerv, (GLenum pname, GLint * data))                                                                                       \
    X(GLenum, GetError, ())                                                                                                                  \
    X(void, GenBuffers, (GLsizei n, GLuint * buffers))                                                                                       \
    X(void, DeleteBuffers, (GLsizei n, const GLuint *buffers))                                                                               \
    X(void, BindBuffer, (GLenum target, GLuint buffer))                                                                                      \
    X(void, BufferData, (GLenum target, GLsizeiptr size, const void *data, GLenum usage))                                                    \
    X(void, BufferSubData, (GLenum target, GLintptr offset, GLsizeiptr size, const void *data))                                              \
    X(void, GetBufferSubData, (GLenum target, GLintptr offset, GLsizeiptr size, void *data))                                                 \
    X(void, CopyBufferSubData, (GLenum read, GLenum write, GLintptr read_offset, GLintptr write_offset, GLsizeiptr size))                    \
    X(void, BindBufferBase, (GLenum target, GLuint index, GLuint buffer))                                                                    \
    X(void, BindBufferRange, (GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size))                                 \
    X(GLuint, CreateShader, (GLenum type))                                                                                                   \
    X(void, ShaderSource, (GLuint shader, GLsizei count, const GLchar *const *string, const GLint *length))                                  \
    X(void, CompileShader, (GLuint shader))                                                                                                  \
    X(void, GetShaderiv, (GLuint shader, GLenum pname, GLint * params))                                                                      \
    X(void, GetShaderInfoLog, (GLuint shader, GLsizei size, GLsizei * length, GLchar * log))                                                 \
    X(void, DeleteShader, (GLuint shader))                                                                                                   \
    X(GLuint, CreateProgram, ())                                                                                                             \
    X(void, AttachShader, (GLuint program, GLuint shader))                                                                                   \
    X(void, LinkProgram, (GLuint program))                                                                                                   \
    X(void, GetProgramiv, (GLuint program, GLenum pname, GLint * params))                                                                    \
    X(void, GetProgramInfoLog, (GLuint program, GLsizei size, GLsizei * length, GLchar * log))                                               \
    X(void, DeleteProgram, (GLuint program))                                                                                                 \
    X(void, UseProgram, (GLuint program))                                                                                                    \
    X(void, DispatchCompute, (GLuint x, GLuint y, GLuint z))                                                                                 \
    X(void, MemoryBarrier, (GLbitfield barriers))                                                                                            \
    X(GLsync, FenceSync, (GLenum condition, GLbitfield flags))                                                                               \
    X(GLenum, ClientWaitSync, (GLsync sync, GLbitfield flags, GLuint64 timeout))                                                             \
    X(void, DeleteSync, (GLsync sync))                                                                                                       \
    X(void, Flush, ())                                                                                                                       \
    X(void, Finish, ())                                                                                                                      \
    X(void, GenVertexArrays, (GLsizei n, GLuint * arrays))                                                                                   \
    X(void, BindVertexArray, (GLuint array))                                                                                                 \
    X(void, DrawArrays, (GLenum mode, GLint first, GLsizei count))                                                                           \
    X(void, Viewport, (GLint x, GLint y, GLsizei width, GLsizei height))                                                                     \
    X(void, ClearColor, (GLfloat r, GLfloat g, GLfloat b, GLfloat a))                                                                        \
    X(void, Clear, (GLbitfield mask))                                                                                                        \
    X(void, BindFramebuffer, (GLenum target, GLuint framebuffer))                                                                            \
    X(void, Disable, (GLenum cap))                                                                                                           \
    X(void, GenQueries, (GLsizei n, GLuint * ids))                                                                                           \
    X(void, BeginQuery, (GLenum target, GLuint id))                                                                                          \
    X(void, EndQuery, (GLenum target))                                                                                                       \
    X(void, GetQueryObjectiv, (GLuint id, GLenum pname, GLint * params))                                                                     \
    X(void, GetQueryObjectui64v, (GLuint id, GLenum pname, GLuint64 * params))                                                               \
    X(void, GenTextures, (GLsizei n, GLuint * textures))                                                                                     \
    X(void, DeleteTextures, (GLsizei n, const GLuint *textures))                                                                             \
    X(void, BindTexture, (GLenum target, GLuint texture))                                                                                    \
    X(void, TexStorage2D, (GLenum target, GLsizei levels, GLenum format, GLsizei width, GLsizei height))                                     \
    X(void, CopyImageSubData, (GLuint src, GLenum src_target, GLint src_level, GLint src_x, GLint src_y, GLint src_z, GLuint dst, GLenum dst_target, GLint dst_level, GLint dst_x, GLint dst_y, GLint dst_z, GLsizei width, GLsizei height, GLsizei depth)) \
    X(void, TexParameteri, (GLenum target, GLenum pname, GLint param))                                                                       \
    X(void, ActiveTexture, (GLenum texture))                                                                                                 \
    X(void, GenFramebuffers, (GLsizei n, GLuint * framebuffers))                                                                             \
    X(void, DeleteFramebuffers, (GLsizei n, const GLuint *framebuffers))                                                                     \
    X(void, FramebufferTexture2D, (GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level))                         \
    X(GLenum, CheckFramebufferStatus, (GLenum target))                                                                                       \
    X(void, DrawBuffers, (GLsizei n, const GLenum *buffers))                                                                                 \
    X(void, Enable, (GLenum cap))                                                                                                            \
    X(void, BlendFuncSeparate, (GLenum src_rgb, GLenum dst_rgb, GLenum src_alpha, GLenum dst_alpha))                                         \
    X(void, BlendEquationSeparate, (GLenum rgb, GLenum alpha))                                                                               \
    X(void, BlendColor, (GLfloat r, GLfloat g, GLfloat b, GLfloat a))                                                                        \
    X(void, ColorMask, (GLboolean r, GLboolean g, GLboolean b, GLboolean a))                                                                 \
    X(void, DepthFunc, (GLenum func))                                                                                                        \
    X(void, DepthMask, (GLboolean flag))                                                                                                     \
    X(void, Scissor, (GLint x, GLint y, GLsizei width, GLsizei height))

#define PS2_GL_DECLARE(ret, name, args) extern ret(*gl##name) args;
PS2_GL_FUNCTIONS(PS2_GL_DECLARE)
#undef PS2_GL_DECLARE

// Loads every function above; false (with the missing name printed) when the
// driver lacks one.
bool load_functions();

}
