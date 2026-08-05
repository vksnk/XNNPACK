// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/arch.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "ynnpack/base/base.h"
#include "ynnpack/base/log.h"

#ifdef YNN_ENABLE_CPUINFO
#include <cpuinfo.h>
#endif

namespace ynn {

namespace {

// Debug knob: YNN_DISABLE_ARCH is a comma-separated list of arch flag names
// (e.g. "avx512vnni,amxint8") to remove from the detected features.
uint64_t get_disabled_arch_flags() {
  const char* disabled = getenv("YNN_DISABLE_ARCH");
  if (!disabled) return 0;
  struct named_flag {
    const char* name;
    uint64_t flags;
  };
  static constexpr named_flag named_flags[] = {
#ifdef YNN_ARCH_X86
      {"sse2", arch_flag::sse2},
      {"ssse3", arch_flag::ssse3},
      {"sse41", arch_flag::sse41},
      {"avx", arch_flag::avx},
      {"f16c", arch_flag::f16c},
      {"avx2", arch_flag::avx2},
      {"fma3", arch_flag::fma3},
      {"avx512f", arch_flag::avx512f},
      {"avx512bw", arch_flag::avx512bw},
      {"avx512vl", arch_flag::avx512vl},
      {"avx512dq", arch_flag::avx512dq},
      {"avx512bf16", arch_flag::avx512bf16},
      {"avx512fp16", arch_flag::avx512fp16},
      {"avx512vnni", arch_flag::avx512vnni},
      {"avx512", arch_flag::avx512},
      {"amxbf16", arch_flag::amxbf16},
      {"amxfp16", arch_flag::amxfp16},
      {"amxint8", arch_flag::amxint8},
#endif  // YNN_ARCH_X86
#ifdef YNN_ARCH_ARM
      {"neon", arch_flag::neon},
      {"neonfma", arch_flag::neonfma},
      {"neondot", arch_flag::neondot},
      {"neonfp16", arch_flag::neonfp16},
      {"neonfp16arith", arch_flag::neonfp16arith},
      {"neonbf16", arch_flag::neonbf16},
      {"neonfp8", arch_flag::neonfp8},
      {"neonfp8dot4", arch_flag::neonfp8dot4},
      {"neoni8mm", arch_flag::neoni8mm},
      {"sme", arch_flag::sme},
      {"sme2", arch_flag::sme2},
      {"sve", arch_flag::sve},
#endif  // YNN_ARCH_ARM
  };
  uint64_t result = 0;
  const char* token = disabled;
  while (*token) {
    const char* end = strchr(token, ',');
    const size_t len = end ? static_cast<size_t>(end - token) : strlen(token);
    for (const named_flag& f : named_flags) {
      if (strlen(f.name) == len && strncmp(f.name, token, len) == 0) {
        result |= f.flags;
      }
    }
    token = end ? end + 1 : token + len;
  }
  YNN_LOG_WARNING() << "YNN_DISABLE_ARCH is set, disabling arch flags "
                    << result;
  return result;
}

}  // namespace

#if defined(YNN_ARCH_X86_64) && defined(__linux__) && !defined(CHROMIUM)
#include <sys/syscall.h>
#include <sys/types.h>

#define XFEATURE_XTILEDATA 18
#define ARCH_REQ_XCOMP_PERM 0x1023

ssize_t ynn_syscall(size_t rax, size_t rdi, size_t rsi, size_t rdx) {
  __asm("syscall"
        : "+a"(rax)
        : "D"(rdi), "S"(rsi), "d"(rdx)
        : "rcx", "r11", "memory");
  return rax;
}

bool can_use_amx_tile() {
  return ynn_syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA,
                     0) == 0;
}

#else
bool can_use_amx_tile() { return false; }
#endif

uint64_t get_supported_arch_flags() {
  static uint64_t flags = []() -> uint64_t {
    uint64_t result = 0;
#ifdef YNN_ENABLE_CPUINFO
    if (!cpuinfo_initialize()) {
      YNN_LOG_WARNING() << "Failed to initialize cpuinfo";
      return 0;
    }

#ifdef YNN_ARCH_X86
    result |= arch_flag::sse2;
    result |= arch_flag::sse2_fma;
    if (cpuinfo_has_x86_ssse3()) result |= arch_flag::ssse3;
    if (cpuinfo_has_x86_sse4_1()) result |= arch_flag::sse41;
    if (cpuinfo_has_x86_avx()) result |= arch_flag::avx;
    if (cpuinfo_has_x86_f16c()) result |= arch_flag::f16c;
    if (cpuinfo_has_x86_avx2()) result |= arch_flag::avx2;
    if (cpuinfo_has_x86_fma3()) result |= arch_flag::fma3;
    if (cpuinfo_has_x86_avx512f()) result |= arch_flag::avx512f;
    if (cpuinfo_has_x86_avx512bw()) result |= arch_flag::avx512bw;
    if (cpuinfo_has_x86_avx512vl()) result |= arch_flag::avx512vl;
    if (cpuinfo_has_x86_avx512dq()) result |= arch_flag::avx512dq;
    if (cpuinfo_has_x86_avx512bf16()) result |= arch_flag::avx512bf16;
    if (cpuinfo_has_x86_avx512fp16()) result |= arch_flag::avx512fp16;
    if (cpuinfo_has_x86_avx512vnni()) result |= arch_flag::avx512vnni;
    if (cpuinfo_has_x86_amx_tile() && can_use_amx_tile()) {
      if (cpuinfo_has_x86_amx_bf16()) result |= arch_flag::amxbf16;
      if (cpuinfo_has_x86_amx_fp16()) result |= arch_flag::amxfp16;
      if (cpuinfo_has_x86_amx_int8()) result |= arch_flag::amxint8;
    }
#endif  // YNN_ARCH_X86
#ifdef YNN_ARCH_ARM
    if (cpuinfo_has_arm_neon()) result |= arch_flag::neon;
    if (cpuinfo_has_arm_neon_fma()) result |= arch_flag::neonfma;
    if (cpuinfo_has_arm_neon_dot()) result |= arch_flag::neondot;
    if (cpuinfo_has_arm_neon_fp16()) result |= arch_flag::neonfp16;
    if (cpuinfo_has_arm_neon_fp16_arith()) result |= arch_flag::neonfp16arith;
    if (cpuinfo_has_arm_neon_bf16()) result |= arch_flag::neonbf16;
    if (cpuinfo_has_arm_i8mm()) result |= arch_flag::neoni8mm;
    if (cpuinfo_has_arm_fp8()) result |= arch_flag::neonfp8;
    if (cpuinfo_has_arm_f8dot()) result |= arch_flag::neonfp8dot4;
#if !YNN_COMPILER_HAS_FEATURE(memory_sanitizer)
    // msan (understandably) does not support SVE/SME (b/494230133).
    if (cpuinfo_has_arm_sme()) result |= arch_flag::sme;
    if (cpuinfo_has_arm_sme2()) result |= arch_flag::sme2;
    if (cpuinfo_has_arm_sve()) result |= arch_flag::sve;
#endif  //  YNN_COMPILER_HAS_FEATURE(memory_sanitizer)
#endif  // YNN_ARCH_ARM
#endif  // YNN_ENABLE_CPUINFO
#ifdef YNN_ARCH_HEXAGON
    result |= arch_flag::hvx;
#endif  // YNN_ARCH_HEXAGON
#ifdef YNN_ARCH_WASM
    result |= arch_flag::simd128;
#endif  // YNN_ARCH_WASM
    result &= ~get_disabled_arch_flags();
    return result;
  }();
  return flags;
}

}  // namespace ynn
