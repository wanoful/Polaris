#!/bin/bash
# Generate rust-project.json for the POLARIS out-of-tree kernel module.
#
# Usage: ./scripts/gen-rust-project.sh [KOBJTREE=/path/to/kernel-build] [KSRCTREE=/path/to/kernel-source]
#
# This invokes the kernel's generate_rust_analyzer.py with exttree pointing
# to our module source, then patches the output to add OBJTREE and source
# include directories for proper rust-analyzer resolution.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXTTREE="$PROJECT_ROOT/kernel"

KREL="$(uname -r)"
KBASE="${KREL%%-*}"

# Resolve the kernel object/build tree. This must contain generated files such
# as include/generated/rustc_cfg and the generated Rust bindings.
KOBJTREE="${KOBJTREE:-${KDIR:-${1:-}}}"
if [ -z "$KOBJTREE" ]; then
    for candidate in \
        "/lib/modules/$KREL/build" \
        "/lib/modules/$KREL/source"; do
        if [ -f "$candidate/Kbuild" ] || [ -f "$candidate/Makefile" ]; then
            KOBJTREE="$candidate"
            break
        fi
    done
fi

if [ -z "$KOBJTREE" ] || [ ! -d "$KOBJTREE" ]; then
    echo "Error: Cannot find kernel object/build tree." >&2
    echo "Set KOBJTREE or KDIR to the kernel build directory." >&2
    exit 1
fi
KOBJTREE="$(realpath "$KOBJTREE")"

if [ ! -f "$KOBJTREE/include/generated/rustc_cfg" ]; then
    echo "Error: $KOBJTREE is missing include/generated/rustc_cfg." >&2
    echo "Use a configured Rust-enabled kernel build tree for KOBJTREE/KDIR." >&2
    exit 1
fi

# Resolve the kernel source tree. Ubuntu's headers package provides the object
# tree and generated Rust files, but the actual rust/kernel/*.rs sources live in
# linux-source-*.
KSRCTREE="${KSRCTREE:-}"
if [ -z "$KSRCTREE" ]; then
    for candidate in \
        "$KOBJTREE" \
        "/lib/modules/$KREL/source" \
        "/usr/src/linux-source-$KBASE" \
        /usr/src/linux-source-*; do
        if [ -f "$candidate/rust/kernel/lib.rs" ] && \
           [ -f "$candidate/scripts/generate_rust_analyzer.py" ]; then
            KSRCTREE="$candidate"
            break
        fi
    done
fi

if [ -z "$KSRCTREE" ] || [ ! -d "$KSRCTREE" ]; then
    echo "Error: Cannot find kernel source tree with rust/kernel/lib.rs." >&2
    echo "Install the matching kernel source package or set KSRCTREE." >&2
    exit 1
fi
KSRCTREE="$(realpath "$KSRCTREE")"

GENERATOR="$KOBJTREE/scripts/generate_rust_analyzer.py"
if [ ! -f "$GENERATOR" ]; then
    GENERATOR="$KSRCTREE/scripts/generate_rust_analyzer.py"
fi
if [ ! -f "$GENERATOR" ]; then
    echo "Error: Cannot find scripts/generate_rust_analyzer.py." >&2
    exit 1
fi

echo "POLARIS rust-analyzer config generator"
echo "  KSRCTREE: $KSRCTREE"
echo "  KOBJTREE: $KOBJTREE"
echo "  Module  : $EXTTREE"

export RUSTC="${RUSTC:-$(which rustc)}"
RUSTC_SYSROOT="$($RUSTC --print sysroot)"
RUST_LIB_SRC="${RUST_LIB_SRC:-}"
if [ -z "$RUST_LIB_SRC" ]; then
    for candidate in \
        "$RUSTC_SYSROOT/lib/rustlib/src/rust/library" \
        "$RUSTC_SYSROOT/lib/rustlib/src/rust" \
        "/usr/lib/rustlib/src/rust/library" \
        "/usr/lib/rustlib/src/rust"; do
        if [ -f "$candidate/core/src/lib.rs" ]; then
            RUST_LIB_SRC="$candidate"
            break
        fi
    done
fi

if [ -z "$RUST_LIB_SRC" ] || [ ! -f "$RUST_LIB_SRC/core/src/lib.rs" ]; then
    echo "Error: Cannot find Rust library sources." >&2
    echo "Install rust-src or set RUST_LIB_SRC to the directory containing core/src/lib.rs." >&2
    exit 1
fi
RUST_LIB_SRC="$(realpath "$RUST_LIB_SRC")"
CORE_CFGS="--cfg no_fp_fmt_parse"
CORE_EDITION="2024"

echo "  Sysroot : $RUSTC_SYSROOT"
echo "  Rust src: $RUST_LIB_SRC"

# Generate base rust-project.json via kernel script
python3 "$GENERATOR" \
    --cfgs "core=$CORE_CFGS" \
    --cfgs 'proc_macro2=feature="proc-macro" wrap_proc_macro proc_macro_span_file proc_macro_span_location' \
    --cfgs 'quote=feature="proc-macro"' \
    --cfgs 'syn=feature="clone-impls" feature="derive" feature="full" feature="parsing" feature="printing" feature="proc-macro" feature="visit-mut"' \
    "$CORE_EDITION" \
    "$KSRCTREE" \
    "$KOBJTREE" \
    "$RUSTC_SYSROOT" \
    "$RUST_LIB_SRC" \
    "$EXTTREE" \
    > "$PROJECT_ROOT/rust-project.json"

# Patch the polaris crate entry with OBJTREE and source include dirs
PROJECT_ROOT="$PROJECT_ROOT" EXTTREE="$EXTTREE" KSRCTREE="$KSRCTREE" KOBJTREE="$KOBJTREE" python3 - <<'PY'
import json
import os
import sys

project_root = os.environ["PROJECT_ROOT"]
exttree = os.environ["EXTTREE"]
ksrctree = os.environ["KSRCTREE"]
kobjtree = os.environ["KOBJTREE"]
project_file = os.path.join(project_root, "rust-project.json")

with open(project_file) as f:
    data = json.load(f)

found = False
for crate in data["crates"]:
    if crate["display_name"] == "polaris":
        source = crate.setdefault("source", {})
        source.setdefault("include_dirs", [])
        source.setdefault("exclude_dirs", [])
        source["include_dirs"].extend([
            exttree,
            os.path.join(ksrctree, "rust"),
            os.path.join(kobjtree, "rust"),
        ])
        crate["env"]["OBJTREE"] = kobjtree

        def dedup(paths):
            seen = set()
            out = []
            for path in paths:
                if path not in seen:
                    seen.add(path)
                    out.append(path)
            return out

        source["include_dirs"] = dedup(source["include_dirs"])
        source["exclude_dirs"] = dedup(source["exclude_dirs"])
        found = True
        break

if not found:
    print("Error: generated rust-project.json does not contain a polaris crate", file=sys.stderr)
    sys.exit(1)

with open(project_file, "w") as f:
    json.dump(data, f, sort_keys=True, indent=4)

print("  Patched polaris crate with OBJTREE and source includes")
PY

echo "Written: $PROJECT_ROOT/rust-project.json"
echo "Done. rust-analyzer will auto-detect this file."
