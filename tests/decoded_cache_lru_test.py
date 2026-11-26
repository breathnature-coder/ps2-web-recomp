#!/usr/bin/env python3
"""Compile the real decode allocator in isolation and check it against a word oracle.

Run from any directory with Python 3 and a C++20 compiler (CXX defaults to c++).
The extracted section keeps this test independent of GPU SDK/device availability.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'runtime/src/gs/gpu/gpu.cpp').read_text()
allocator = source[source.index('struct DecodeKey {'):source.index('// Admission history')]
mix = source[source.index('std::uint64_t mix('):source.index('// Indexed sampling')]
prefix = r'''
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <unordered_map>
#include <vector>
#include <random>
#include "gs/gpu/decoded_arena.h"
using ps2::gs::decoded_arena_can_append;
constexpr std::uint32_t kDecodedWords = 1024;
std::uint32_t g_decoded_used = 0;
'''
test = r'''
void require(bool ok) {
    if (!ok) { std::fputs("decode LRU invariant failed\n", stderr); std::abort(); }
}
DecodeKey key(unsigned n) { DecodeKey k{}; k.tbp = n; return k; }
void verify() {
    std::array<unsigned, kDecodedWords> owners{};
    auto mark = [&](unsigned at, unsigned words) {
        require(words && at + words <= g_decoded_used);
        for (unsigned i = at; i < at + words; ++i) require(++owners[i] == 1);
    };
    for (auto &[k,d] : g_decoded_cache) mark(d.offset,d.words);
    for (auto &[s,b] : g_decode_retired) mark(b.offset,b.words);
    for (auto &b : g_decode_free) mark(b.offset,b.words);
    for (unsigned i = 0; i < g_decoded_used; ++i) require(owners[i] == 1);
    std::size_t count = 0;
    Decoded *previous = nullptr;
    for (auto *d = g_decode_oldest; d; d = d->newer) {
        require(++count <= g_decoded_cache.size());
        require(d->older == previous);
        require(d->key && &g_decoded_cache.at(*d->key) == d);
        if (previous) require(previous->last_use_serial <= d->last_use_serial);
        previous = d;
    }
    require(count == g_decoded_cache.size() && previous == g_decode_newest);
    require((g_decode_oldest == nullptr) == g_decoded_cache.empty());
}
void reset() {
    decode_cache_clear(); g_decode_retired.clear(); g_decode_free.clear();
    g_decoded_used = 0; g_decode_reusable_serial = 0; verify();
}
int main() {
    unsigned offset = 0;
    // Same-batch recency must survive rehash and choose the untouched oldest.
    for (unsigned i = 0; i < 4; ++i) {
        require(decode_allocate(256, offset));
        decode_cache_store(key(i), {offset, 0, 256, 1});
    }
    decode_lru_touch(g_decoded_cache.at(key(0)));
    g_decoded_cache.rehash(1009); verify();
    require(!decode_allocate(256, offset)); // batch one is still protected
    g_decode_reusable_serial = 1;
    require(decode_allocate(256, offset));
    require(!g_decoded_cache.contains(key(1)) && g_decoded_cache.contains(key(0)));
    decode_cache_store(key(4), {offset, 0, 256, 2}); verify();
    reset();
    // Fragmented safe regions cannot justify evicting any entries.
    for (unsigned i = 0; i < 8; ++i) {
        require(decode_allocate(128, offset));
        decode_cache_store(key(i), {offset, 0, 128, 1});
    }
    for (unsigned i = 1; i < 8; i += 2) {
        auto &d = g_decoded_cache.at(key(i)); d.last_use_serial = 2; decode_lru_touch(d);
    }
    g_decode_reusable_serial = 1;
    require(!decode_allocate(256, offset) && g_decoded_cache.size() == 8); verify();
    reset();
    std::mt19937 rng(42);
    std::uint64_t serial = 1;
    unsigned next_key = 20;
    for (unsigned step = 0; step < 100000; ++step) {
        const unsigned op = rng() % 6;
        if (op == 0) {
            ++serial;
            // Exercise both immediate queue ordering and delayed completion.
            const auto lag = rng() % 5;
            if (serial > lag + 1) g_decode_reusable_serial = std::max(g_decode_reusable_serial, serial - lag - 1);
            decode_collect();
        } else if (op == 1 && !g_decoded_cache.empty()) {
            auto it = g_decoded_cache.begin(); std::advance(it, rng() % g_decoded_cache.size());
            decode_cache_erase(it);
        } else if (op == 2 && !g_decoded_cache.empty()) {
            auto it = g_decoded_cache.begin(); std::advance(it, rng() % g_decoded_cache.size());
            it->second.last_use_serial = serial; decode_lru_touch(it->second);
        } else {
            const unsigned words = 1 + rng() % 256;
            std::array<bool, kDecodedWords> busy{};
            std::vector<DecodeKey> protected_keys;
            auto protect = [&](unsigned at, unsigned n) { for (unsigned i = at; i < at + n; ++i) busy[i] = true; };
            for (auto &[k,d] : g_decoded_cache) if (d.last_use_serial > g_decode_reusable_serial) {
                protected_keys.push_back(k); protect(d.offset,d.words);
            }
            for (auto &[s,b] : g_decode_retired) if (s > g_decode_reusable_serial) protect(b.offset,b.words);
            unsigned run = 0, best = 0;
            for (bool b : busy) { run = b ? 0 : run + 1; best = std::max(best, run); }
            const auto evictions = g_decode_evictions;
            const bool success = decode_allocate(words, offset);
            require(success == (best >= words));
            for (auto k : protected_keys) require(g_decoded_cache.contains(k));
            if (success) decode_cache_store(key(next_key++), {offset, 0, words, serial});
            else require(evictions == g_decode_evictions);
        }
        if (step % 127 == 0) g_decoded_cache.rehash(17 + rng() % 1024);
        verify();
        if (step % 10007 == 0) reset();
    }
    reset();
    std::puts("PASS: LRU recency, rehash/reset safety, fragmented requests, and 100000 operations against a per-word oracle");
}
'''
with tempfile.TemporaryDirectory(prefix='gt4-lru-') as temp:
    cpp = Path(temp) / 'test.cpp'
    exe = Path(temp) / 'test'
    cpp.write_text(prefix + mix + allocator + test)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    flags = shlex.split(os.environ.get('CXXFLAGS', '-O2'))
    subprocess.run(compiler + ['-std=c++20', *flags, '-I' + str(ROOT / 'runtime/src'), str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
