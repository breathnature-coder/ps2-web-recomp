// Nintendo Switch entry point (homebrew, libnx).
//
// The Switch has no command line, so the arguments the desktop runner takes
// are read from a text file next to the .nro instead:
//
//   sdmc:/switch/ps2recomp/args.txt
//
// one argument per line (or several separated by spaces; quote a path that
// contains spaces). Lines starting with '#' are comments. A typical file:
//
//   core.elf
//   --disc gt4.iso
//   --memcard gt4.ps2
//   --patches patches.txt
//
// Relative paths are taken from sdmc:/switch/ps2recomp. scripts/build_switch.sh
// writes this file and lays out the folder.
//
// A few lines are understood by this launcher rather than the runner:
//   @boost            run the CPU at 1785 MHz (the "fast load" boost mode)
//   @log              send stdout/stderr to log.txt in the same folder
//   @env NAME=VALUE   set an environment variable the runtime reads, such as
//                     PS2_GS=cpu for the software renderer or PS2_SCALE=2
#include <switch.h>

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int ps2_main(int argc, char **argv);

namespace {

constexpr const char *kHome = "sdmc:/switch/ps2recomp";

std::vector<std::string> split(const std::string &line) {
    std::vector<std::string> out;
    std::string cur;
    bool quoted = false, any = false;
    for (char ch : line) {
        if (ch == '"') {
            quoted = !quoted;
            any = true;
        } else if (!quoted && (ch == ' ' || ch == '\t' || ch == '\r')) {
            if (any) out.push_back(cur);
            cur.clear();
            any = false;
        } else {
            cur.push_back(ch);
            any = true;
        }
    }
    if (any) out.push_back(cur);
    return out;
}

bool read_args(std::vector<std::string> &args, bool &boost, bool &log) {
    std::FILE *f = std::fopen("args.txt", "r");
    if (!f) return false;
    char buf[1024];
    while (std::fgets(buf, sizeof buf, f)) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        const auto first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#') continue;
        const std::vector<std::string> words = split(line);
        for (std::size_t i = 0; i < words.size(); ++i) {
            const std::string &word = words[i];
            if (word == "@boost") boost = true;
            else if (word == "@log") log = true;
            else if (word == "@env" && i + 1 < words.size()) {
                const std::string &pair = words[++i];
                const auto eq = pair.find('=');
                if (eq != std::string::npos) setenv(pair.substr(0, eq).c_str(), pair.substr(eq + 1).c_str(), 1);
            } else {
                args.push_back(word);
            }
        }
    }
    std::fclose(f);
    return true;
}

// Shown when something is wrong before the game starts, until + is pressed.
void fail(const char *message) {
    consoleInit(nullptr);
    padConfigureInput(1, HidNpadStyleSet_NpadStandard);
    PadState pad;
    padInitializeDefault(&pad);
    std::printf("ps2recomp\n\n%s\n\nPress + to exit.\n", message);
    while (appletMainLoop()) {
        padUpdate(&pad);
        if (padGetButtonsDown(&pad) & HidNpadButton_Plus) break;
        consoleUpdate(nullptr);
    }
    consoleExit(nullptr);
}

bool g_boost = false;

}  // namespace

// Called by host/window.cpp right before it exits the process, which skips the
// end of main below.
void ps2_switch_before_exit() {
    appletSetMediaPlaybackState(false);
    if (g_boost) appletSetCpuBoostMode(ApmCpuBoostMode_Normal);
}

int main(int, char **) {
    if (chdir(kHome) != 0) {
        fail("Folder sdmc:/switch/ps2recomp not found.\nCopy the build's sd/ folder to the root of the SD card.");
        return 1;
    }

    // The 32 MiB of PS2 RAM, the GS memory, the generated code and the
    // renderer need far more than the few hundred MiB an applet (the album)
    // gets, so the game has to run as an application: launch it from the
    // homebrew menu while holding R over a game.
    const AppletType type = appletGetAppletType();
    if (type != AppletType_Application && type != AppletType_SystemApplication) {
        fail("Running in applet mode, which does not have enough memory.\n"
             "Start the homebrew menu by holding R while launching a game, then open this again.");
        return 1;
    }

    std::vector<std::string> args{"ps2nx"};
    bool boost = false, log = false;
    if (!read_args(args, boost, log) || args.size() < 2) {
        fail("sdmc:/switch/ps2recomp/args.txt is missing or names no executable.");
        return 1;
    }
    if (log) {
        std::freopen("log.txt", "w", stdout);
        std::freopen("log.txt", "a", stderr);
        std::setvbuf(stderr, nullptr, _IOLBF, 0);
    }
    g_boost = boost;
    if (boost) appletSetCpuBoostMode(ApmCpuBoostMode_FastLoad);
    // Keep the console awake while a race is on and nobody touches the stick.
    appletSetMediaPlaybackState(true);

    args.push_back("--fullscreen");
    std::vector<char *> argv;
    for (std::string &a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    const int result = ps2_main(static_cast<int>(argv.size()) - 1, argv.data());
    ps2_switch_before_exit();
    return result;
}
