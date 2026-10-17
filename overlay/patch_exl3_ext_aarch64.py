#!/usr/bin/env python3
"""e8 (exllamav3 1.6.0, 151539c): guard-aware aarch64 verifier (no shim needed).

Upstream 1.6.0 made the x86 CPU TUs aarch64-safe at the SOURCE level:
  - cpu/moe_mul1.cpp wraps the whole TU in
    `#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))`
    and provides, under `#else  // !x86: EXL3_AARCH64_STUB`, a complete aarch64
    stub set for every exl3_moe_cpu_* symbol it owns (TORCH_CHECK(false) on
    reach; GB10 serving uses the fused GPU exl3_moe path);
  - parallel/all_reduce_cpu_avx2.cpp / avx512.cpp carry the same line-1 guard;
  - avx2_target.h / avx512_target.h gate the target-attribute macros behind
    `#if defined(__linux__) && (defined(__x86_64__) ...)` (marked EXL3_AARCH64_STUB)
    and exclude <intrin.h> on Linux;
  - the spin pauses carry EXL3_AARCH64_PAUSE markers (yield on aarch64).

The old kit stub patcher (wholesale TU replacement, pause rewrites, avx2_target
rewrites) is therefore RETIRED, and so is the interim link shim: upstream's own
#else stub branch already satisfies the unguarded bindings.cpp registrations
(a home-grown shim duplicates those definitions and fails the link with
"multiple definition"). This installer only VERIFIES, fail closed:

  1. every upstream guard listed above is present,
  2. every exl3_moe_cpu_* binding registered in bindings.cpp has a definition
     site in moe_mul1.cpp's !x86 stub branch or in the moe_handoff files,
  3. every file carrying an x86 ISA marker is one of the guard-covered files.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

X86_GUARD = "#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))"
GUARDED_TUS = ("cpu/moe_mul1.cpp", "parallel/all_reduce_cpu_avx2.cpp", "parallel/all_reduce_cpu_avx512.cpp")
STUB_GUARDED_HEADERS = ("avx2_target.h", "avx512_target.h")

X86_ISA_MARKERS = (
    "immintrin.h",
    "__builtin_ia32_pause",
    "_mm_pause",
    '__attribute__((target("avx',
    'target_clones("avx',
    "__builtin_cpu_supports",
    "__cpuidex",
    "intrin.h",
)


def fail(msg: str) -> None:
    raise SystemExit(f"e8 aarch64 verifier FATAL: {msg}")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/exllamav3/exllamav3/exllamav3_ext")

    # 1. Upstream x86 guards must be present and exact.
    for rel in GUARDED_TUS:
        head = (root / rel).read_text()[:220]
        if X86_GUARD not in head:
            fail(f"{rel} lost the upstream line-1 x86 guard")
    for rel in STUB_GUARDED_HEADERS:
        text = (root / rel).read_text()
        if "EXL3_AARCH64_STUB" not in text or "defined(__x86_64__)" not in text:
            fail(f"{rel} lost the upstream target-macro guard (EXL3_AARCH64_STUB)")

    # 2. moe_mul1.cpp's !x86 stub branch (EXL3_AARCH64_STUB) is the definition
    #    site for its exl3_moe_cpu_* symbols on aarch64.
    mul1 = (root / "cpu" / "moe_mul1.cpp").read_text()
    lines = mul1.splitlines()
    stub_line = next(
        (i for i, ln in enumerate(lines) if ln.startswith("#else") and "EXL3_AARCH64_STUB" in ln),
        -1,
    )
    if stub_line < 0:
        fail("moe_mul1.cpp lost the upstream !x86 EXL3_AARCH64_STUB stub branch")
    stub_region = "\n".join(lines[stub_line:])
    stub_syms = set(re.findall(r"\b(exl3_moe_cpu_[A-Za-z0-9_]+)", stub_region))

    # 3. pybind contract: every registered exl3_moe_cpu_* symbol defined on
    #    aarch64 (stub branch or handoff files).
    bindings = root / "bindings.cpp"
    if not bindings.is_file():
        fail("bindings.cpp missing")
    registered = sorted(
        set(re.findall(r'm\.def\(\s*"(exl3_moe_cpu_[A-Za-z0-9_]+)"', bindings.read_text()))
    )
    if not registered:
        fail("no exl3_moe_cpu_* bindings found - anchor drifted")
    handoff = (root / "cpu" / "moe_handoff.cu").read_text() + (root / "cpu" / "moe_handoff.h").read_text()
    missing = []
    for name in registered:
        if name in stub_syms:
            continue
        if re.search(rf"\b{name}\b", handoff):
            continue
        missing.append(name)
    if missing:
        fail(
            "registered exl3_moe_cpu_* symbols have no aarch64 definition site: "
            + ", ".join(missing)
        )

    # 4. x86-marker containment: every file with an x86 ISA marker must be one of
    #    the guard-covered files.
    covered = set(GUARDED_TUS) | set(STUB_GUARDED_HEADERS) | {
        "avx2_target.cpp", "avx512_target.cpp", "cpu/moe_handoff.cu", "cpu/moe_handoff.h",
        "cpu/moe_mul1.h", "parallel/all_reduce_cpu.cu",
    }
    strays = []
    for path in root.rglob("*"):
        if path.suffix not in {".c", ".cpp", ".h", ".hpp", ".cu", ".cuh"}:
            continue
        rel = str(path.relative_to(root))
        data = path.read_bytes()
        if any(m.encode() in data for m in X86_ISA_MARKERS) and rel not in covered:
            strays.append(rel)
    if strays:
        fail("x86 ISA markers outside the guard-covered set: " + ", ".join(sorted(strays)))

    print(
        f"e8 aarch64 verifier OK: upstream guards intact, moe_mul1 !x86 stub branch covers "
        f"{len(stub_syms)} symbols ({len(registered)} registered cpu symbols all defined), "
        f"x86 markers contained"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
